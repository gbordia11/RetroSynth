"""Agent 5: validates a reaction path (a route tree from agent 2) with LLM tool calling.

Each step of the route (reactants -> product) is checked in three stages:
1. chemistry gate: the reactant atoms cover the product, and the leftover is a known small
   by-product. A polymer step (product with [*] end points) must also be a valid repeat unit.
2. thermodynamics: each molecule is relaxed and its electronic energy computed with GFN2-xTB
   (default) or ORCA. The step passes only if its energy change dE is below zero.
3. LLM review (only when 1 and 2 pass): a dspy.ReAct agent reads the path and can call tools.
   It can reject a route, but it cannot approve one that fails the thermodynamic check.

A failure at any stage rejects the route, with the failing steps named.
"""

import hashlib
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import dspy
from rdkit import Chem
from rdkit.Chem import AllChem

from config import ValidatorConfig
from gfn2 import Gfn2Error, Gfn2Runner

HARTREE_TO_KCAL = 627.5095

# Small molecules that can appear as by-products, as element counts and SMILES.
KNOWN_BYPRODUCTS = {
    "H2": (Counter(H=2), "[H][H]"),
    "H2O": (Counter(H=2, O=1), "O"),
    "NH3": (Counter(N=1, H=3), "N"),
    "HCl": (Counter(H=1, Cl=1), "Cl"),
    "HBr": (Counter(H=1, Br=1), "Br"),
    "HF": (Counter(H=1, F=1), "F"),
    "CH3OH": (Counter(C=1, H=4, O=1), "CO"),
    "C2H5OH": (Counter(C=2, H=6, O=1), "CCO"),
    "CH3COOH": (Counter(C=2, H=4, O=2), "CC(=O)O"),
    "CO2": (Counter(C=1, O=2), "O=C=O"),
}


class OrcaError(RuntimeError):
    pass


ENERGY_ERRORS = (OrcaError, Gfn2Error, ValueError)  # what a failed energy calculation raises


@dataclass
class StepCheck:
    product: str
    reactants: list  # SMILES, one entry per reactant molecule
    gate_ok: bool
    message: str
    delta: float | None = None  # kcal/mol for the criterion (dG or dE), None if no energy
    note: str = ""  # why delta is missing


@dataclass
class ValidationResult:
    valid: bool
    rationale: str
    steps: list = field(default_factory=list)  # StepCheck, in route order from the target down
    favourable: bool = False  # every step has dE < 0


# --- deterministic chemistry ----------------------------------------------

def parse_species(text):
    """'C=C, OC(=O)C:2' -> {'C=C': 1, 'OC(=O)C': 2}. Each item is SMILES with an optional ':count'."""
    species = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        smi, _, count = item.partition(":")
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None or "*" in smi:
            raise ValueError(f"reactant is not a valid SMILES without [*]: {smi!r}")
        key = Chem.MolToSmiles(mol)
        species[key] = species.get(key, 0) + int(count or 1)
    return species


def formula(smi):
    """Element counts including hydrogens. [*] end points are not counted."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    return Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > 0)


def formula_str(counts):
    """Hill order: C, H, then the rest alphabetically. Empty counts give '(none)'."""
    if not +counts:
        return "(none)"
    order = [e for e in ("C", "H") if counts[e]] + sorted(e for e in counts if e not in "CH" and counts[e])
    return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order)


def check_repeat_unit(polymer):
    """Return (ok, message) for a PSMILES repeat unit with two [*] end points."""
    mol = Chem.MolFromSmiles(polymer)
    if mol is None:
        return False, f"repeat unit is not valid SMILES: {polymer}"
    stars = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    if stars != 2:
        return False, f"repeat unit needs exactly two [*] end points, found {stars}"
    return True, "repeat unit is valid PSMILES with two [*] end points"


def balance(reactants, product):
    """Return (ok, byproduct Counter, message). `reactants` is a {smiles: count} dict."""
    supplied = Counter()
    for smi, n in reactants.items():
        supplied += Counter({e: c * n for e, c in formula(smi).items()})
    needed = formula(product)
    lacking = needed - supplied
    if lacking:
        return False, Counter(), f"product needs atoms the reactants do not supply: {formula_str(lacking)}"
    byproduct = supplied - needed
    return True, byproduct, f"atoms balance; by-product {formula_str(byproduct)}"


def decompose(byproduct):
    """Split a by-product formula into known small molecules. Return {name: count} or None.
    Backtracks, because a greedy choice can strand atoms (H4O2 must be 2 x H2O, not H2 + ...)."""
    rest = +Counter(byproduct)
    if not rest:
        return {}
    for name, (fml, _) in KNOWN_BYPRODUCTS.items():
        if all(rest[e] >= c for e, c in fml.items()):
            sub = decompose(rest - fml)
            if sub is not None:
                return {**sub, name: sub.get(name, 0) + 1}
    return None


def check_step_gate(product, reactants):
    """Gate for one step, reactants -> product. `reactants` is a {smiles: count} dict.
    Return (ok, byproduct Counter, message)."""
    if "[*]" in product:
        ok, message = check_repeat_unit(product)
        if not ok:
            return False, Counter(), message
    ok, byproduct, message = balance(reactants, product)
    if not ok:
        return False, Counter(), message
    if decompose(byproduct) is None:
        return False, byproduct, f"by-product {formula_str(byproduct)} is not a known small molecule"
    return True, byproduct, message


def capped(polymer):
    """Repeat unit with each [*] replaced by H, giving the repeat unit plus H2."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(polymer.replace("[*]", "[H]")))


def route_steps(root):
    """Each reaction step of a route as (product, [reactant SMILES]), target first."""
    return [(step.smiles, [child.smiles for child in step.children]) for step in root.steps()]


def describe_path(steps, label):
    """One line per step in forward order: reactants -> product, with gate result and energy change."""
    lines = []
    for i, s in enumerate(reversed(steps), 1):
        energy = f"{label}={s.delta:.1f} kcal/mol" if s.delta is not None else f"{label} not computed ({s.note})"
        gate = "gate ok" if s.gate_ok else f"gate FAILED: {s.message}"
        lines.append(f"step {i}: {' + '.join(s.reactants)} -> {s.product} | {gate} | {energy}")
    return "\n".join(lines)


# --- ORCA ------------------------------------------------------------------

def parse_energy(text):
    """Final single-point energy (Hartree) from ORCA output. Raise OrcaError if the run failed."""
    if "ORCA TERMINATED NORMALLY" not in text:
        raise OrcaError("ORCA did not terminate normally; see the .out file in the work directory")
    matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
    if not matches:
        raise OrcaError("no final energy in ORCA output")
    return float(matches[-1])


class OrcaRunner:
    """Writes ORCA input files from RDKit geometries and reads back the energies."""

    def __init__(self, config):
        self.config = config
        self.workdir = Path(config.workdir)
        self._energies = {}

    def available(self):
        return shutil.which(self.config.orca_path) is not None

    def input_text(self, smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            raise OrcaError(f"could not generate a 3D geometry for {smiles}")
        AllChem.MMFFOptimizeMolecule(mol)
        charge = Chem.GetFormalCharge(mol)
        multiplicity = 1 + sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        conf = mol.GetConformer()
        lines = [f"! {self.config.method}", f"* xyz {charge} {multiplicity}"]
        for atom in mol.GetAtoms():
            x, y, z = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol()} {x:.6f} {y:.6f} {z:.6f}")
        lines.append("*")
        return "\n".join(lines) + "\n"

    def energy(self, smiles):
        """Energy of a molecule without [*] end points, in Hartree. Results are cached."""
        key = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        if key in self._energies:
            return self._energies[key]
        if not self.available():
            raise OrcaError(f"ORCA not found at '{self.config.orca_path}'")
        self.workdir.mkdir(parents=True, exist_ok=True)
        stem = hashlib.sha1(key.encode()).hexdigest()[:12]
        inp = self.workdir / f"{stem}.inp"
        inp.write_text(self.input_text(key))
        try:
            proc = subprocess.run(
                [self.config.orca_path, inp.name], cwd=self.workdir,
                capture_output=True, text=True, timeout=self.config.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise OrcaError(f"ORCA timed out after {self.config.timeout} s") from exc
        inp.with_suffix(".out").write_text(proc.stdout)
        self._energies[key] = parse_energy(proc.stdout)
        return self._energies[key]


# --- agent -----------------------------------------------------------------

class ValidateRoute(dspy.Signature):
    """Judge whether this reaction path is a sound synthesis from the starting materials to the target.
    Every step has passed the atom-balance gate. dE is the energy change of a step in kcal/mol;
    a step without dE could not be computed. Use the tools to inspect a step or an energy if you
    need more detail. Give a verdict that cites the steps."""

    path: str = dspy.InputField(desc="one line per step, in forward order: reactants -> product")
    valid: bool = dspy.OutputField(desc="true if the reaction path is chemically sound")
    rationale: str = dspy.OutputField(desc="short justification that cites the steps and energies")


class RouteValidator(dspy.Module):
    """Agent 5: validates a reaction path from a route tree, with the LLM calling the tools."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ValidatorConfig()
        if self.config.backend == "gfn2":
            self.energies = Gfn2Runner()
        elif self.config.backend == "orca":
            self.energies = OrcaRunner(self.config)
        else:
            raise ValueError(f"unknown validator backend {self.config.backend!r}; use 'gfn2' or 'orca'")
        if self.config.criterion not in ("dg", "de"):
            raise ValueError(f"unknown criterion {self.config.criterion!r}; use 'dg' or 'de'")
        if self.config.criterion == "dg" and self.config.backend != "gfn2":
            raise ValueError("criterion 'dg' needs backend 'gfn2'; ORCA free energies are not implemented")
        self.lm = dspy.LM(self.config.model, max_tokens=self.config.max_tokens)
        self.agent = dspy.ReAct(
            ValidateRoute,
            tools=[self.check_step, self.molecule_energy, self.step_energy],
            max_iters=self.config.max_iters,
        )

    def forward(self, route):
        """`route` is the root TreeNode from the search. Its leaves are the starting materials."""
        steps = []
        for product, reactants in route_steps(route):
            ok, _, message = check_step_gate(product, Counter(reactants))
            steps.append(StepCheck(product, reactants, ok, message))
        if not steps:
            return ValidationResult(False, "route has no reaction steps")
        failed = [s for s in steps if not s.gate_ok]
        if failed:
            reasons = "; ".join(f"{s.product}: {s.message}" for s in failed)
            return ValidationResult(False, f"rejected by the chemistry gate: {reasons}", steps)
        for s in steps:
            try:
                s.delta = self.step_delta(s.product, Counter(s.reactants))
            except ENERGY_ERRORS as exc:
                s.note = str(exc)
        label = self.criterion_label
        unfavourable = [s for s in steps if s.delta is None or s.delta >= 0]
        if unfavourable:
            reasons = "; ".join(
                f"{s.product}: " + (f"{label}={s.delta:+.1f} kcal/mol" if s.delta is not None else f"no energy ({s.note})")
                for s in unfavourable
            )
            return ValidationResult(False, f"not thermodynamically favourable ({label}): {reasons}", steps, favourable=False)
        with dspy.context(lm=self.lm):
            pred = self.agent(path=describe_path(steps, label))
        # The LLM can only reject a route that passed the energy check, never approve one that failed it.
        return ValidationResult(_as_bool(pred.valid), pred.rationale, steps, favourable=True)

    @property
    def criterion_label(self):
        return "dG" if self.config.criterion == "dg" else "dE"

    def step_delta(self, product, reactants):
        """The change the validator checks, dG or dE per the config, in kcal/mol."""
        if self.config.criterion == "dg":
            return self.step_delta_g(product, reactants)
        return self.step_delta_e(product, reactants)

    def step_delta_e(self, product, reactants):
        """Electronic-energy change dE of one step, reactants -> product + by-product, kcal/mol. Negative is favourable."""
        return self._step_change(product, reactants, self.energies.energy)

    def step_delta_g(self, product, reactants):
        """Free-energy change dG of one step at 298 K and 1 atm, kcal/mol. Negative is favourable. Needs backend gfn2."""
        if not hasattr(self.energies, "free_energy"):
            raise OrcaError("free energies are implemented for backend gfn2 only")
        return self._step_change(product, reactants, self.energies.free_energy)

    @staticmethod
    def _step_change(product, reactants, quantity):
        """Change of `quantity` (energy or free energy, Hartree per molecule) over one step, in kcal/mol."""
        ok, byproduct, message = check_step_gate(product, reactants)
        if not ok:
            raise ValueError(message)
        parts = decompose(byproduct)
        q_h2 = quantity(KNOWN_BYPRODUCTS["H2"][1])
        if "[*]" in product:
            q_product = quantity(capped(product)) - q_h2  # capped repeat unit = RU + H2
        else:
            q_product = quantity(product)
        q_reactants = sum(n * quantity(smi) for smi, n in reactants.items())
        q_byproduct = sum(n * quantity(KNOWN_BYPRODUCTS[name][1]) for name, n in parts.items())
        return (q_product + q_byproduct - q_reactants) * HARTREE_TO_KCAL

    # Tools called by the LLM. They return text, and errors come back as text too,
    # so the agent can read them and try something else.

    def check_step(self, product: str, reactants: str) -> str:
        """Run the chemistry gate on one step. reactants: comma-separated SMILES, each with an optional ':count'."""
        try:
            ok, _, message = check_step_gate(product, parse_species(reactants))
        except ValueError as exc:
            return str(exc)
        return ("gate ok: " if ok else "gate FAILED: ") + message

    def molecule_energy(self, smiles: str) -> str:
        """Electronic energy of one molecule without [*], relaxed with the validator's backend. Returns Hartree."""
        try:
            return f"{self.energies.energy(smiles):.6f} Hartree"
        except ENERGY_ERRORS as exc:
            return f"error: {exc}"

    def step_energy(self, product: str, reactants: str) -> str:
        """Energy change of one step in kcal/mol, dG or dE per the validator's criterion.
        reactants: comma-separated SMILES with optional ':count'."""
        try:
            return f"{self.step_delta(product, parse_species(reactants)):.2f} kcal/mol ({self.criterion_label})"
        except ENERGY_ERRORS as exc:
            return f"error: {exc}"


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)
