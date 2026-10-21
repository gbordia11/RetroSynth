"""Agent 5: validates a polymer route (monomers -> repeat unit) with LLM tool calling.

Deterministic checks run first and gate the verdict: the repeat unit must be valid PSMILES,
and the monomer atoms must cover it with a by-product made of known small molecules.
If they fail, the route is rejected without calling the LLM.

Otherwise a dspy.ReAct agent chooses which tools to call:
- check_repeat_unit(polymer)            PSMILES parses and has exactly two [*] end points
- check_mass_balance(monomers, polymer) monomer atoms = repeat unit + known by-product
- orca_energy(smiles)                    ORCA geometry optimisation + single point, Hartree
- reaction_energy(monomers, polymer)     energy change of monomers -> repeat unit + by-product

ORCA is optional. Without the binary the energy tools return an explanation instead of failing.
"""

import hashlib
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import dspy
from rdkit import Chem
from rdkit.Chem import AllChem

from config import ValidatorConfig

HARTREE_TO_KCAL = 627.5095

# Small molecules that can appear as by-products, as element counts and SMILES.
KNOWN_BYPRODUCTS = {
    "H2": (Counter(H=2), "[H][H]"),
    "H2O": (Counter(H=2, O=1), "O"),
    "NH3": (Counter(N=1, H=3), "N"),
    "HCl": (Counter(H=1, Cl=1), "Cl"),
    "CH3OH": (Counter(C=1, H=4, O=1), "CO"),
    "C2H5OH": (Counter(C=2, H=6, O=1), "CCO"),
    "CO2": (Counter(C=1, O=2), "O=C=O"),
}


class OrcaError(RuntimeError):
    pass


@dataclass
class ValidationResult:
    valid: bool
    rationale: str
    delta_e: float | None = None  # kcal/mol per repeat unit, set when ORCA energies were computed


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
            raise ValueError(f"monomer is not a valid SMILES without [*]: {smi!r}")
        key = Chem.MolToSmiles(mol)
        species[key] = species.get(key, 0) + int(count or 1)
    return species


def format_species(species):
    """Inverse of parse_species, for passing counts to the LLM as text."""
    return ", ".join(f"{smi}:{n}" if n > 1 else smi for smi, n in species.items())


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


def balance(monomers, polymer):
    """Return (ok, byproduct Counter, message). `monomers` is a {smiles: count} dict."""
    supplied = Counter()
    for smi, n in monomers.items():
        supplied += Counter({e: c * n for e, c in formula(smi).items()})
    needed = formula(polymer)
    lacking = needed - supplied
    if lacking:
        return False, Counter(), f"repeat unit needs atoms the monomers do not supply: {formula_str(lacking)}"
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


def capped(polymer):
    """Repeat unit with each [*] replaced by H, giving the repeat unit plus H2."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(polymer.replace("[*]", "[H]")))


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
    """Decide whether the monomers can form the polymer repeat unit by a sound route.
    Use the tools: check the repeat unit and the mass balance first, then the reaction energy
    when ORCA is available. Base the verdict on the tool results and say which ones you used."""

    monomers: str = dspy.InputField(desc="comma-separated monomer SMILES with optional ':count', e.g. 'NCCCCCCN, OC(=O)CCCCC(=O)O'")
    polymer: str = dspy.InputField(desc="repeat unit as PSMILES with two [*] end points")
    valid: bool = dspy.OutputField(desc="true if the route is chemically sound")
    rationale: str = dspy.OutputField(desc="short justification that cites the tool results")


class RouteValidator(dspy.Module):
    """Agent 5: validates monomers -> polymer repeat unit, with the LLM calling the tools."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ValidatorConfig()
        self.orca = OrcaRunner(self.config)
        self.lm = dspy.LM(self.config.model, max_tokens=self.config.max_tokens)
        self.agent = dspy.ReAct(
            ValidateRoute,
            tools=[self.check_repeat_unit, self.check_mass_balance, self.orca_energy, self.reaction_energy],
            max_iters=self.config.max_iters,
        )
        self._delta_e = None

    def forward(self, monomers, polymer):
        self._delta_e = None
        try:
            species = parse_species(monomers)
        except ValueError as exc:
            return ValidationResult(False, str(exc))
        ok, message = self._gate(species, polymer)
        if not ok:
            return ValidationResult(False, message)
        with dspy.context(lm=self.lm):
            pred = self.agent(monomers=monomers, polymer=polymer)
        return ValidationResult(_as_bool(pred.valid), pred.rationale, self._delta_e)

    @staticmethod
    def _gate(species, polymer):
        ok, message = check_repeat_unit(polymer)
        if not ok:
            return False, message
        ok, byproduct, message = balance(species, polymer)
        if not ok:
            return False, message
        if decompose(byproduct) is None:
            return False, f"by-product {formula_str(byproduct)} is not a known small molecule"
        return True, message

    # Tools called by the LLM. They return text, and errors come back as text too,
    # so the agent can read them and try something else.

    def check_repeat_unit(self, polymer: str) -> str:
        """Check that a repeat unit parses as PSMILES with exactly two [*] end points."""
        return check_repeat_unit(polymer)[1]

    def check_mass_balance(self, monomers: str, polymer: str) -> str:
        """Check that the monomer atoms cover the repeat unit and that the excess is a known by-product.
        monomers: comma-separated SMILES, each with an optional ':count'."""
        try:
            ok, byproduct, message = balance(parse_species(monomers), polymer)
        except ValueError as exc:
            return str(exc)
        if not ok:
            return f"not balanced: {message}"
        parts = decompose(byproduct)
        if parts is None:
            return f"not balanced: by-product {formula_str(byproduct)} is not a known small molecule"
        return f"balanced: {message} ({', '.join(f'{n} x {name}' for name, n in parts.items()) or 'no by-product'})"

    def orca_energy(self, smiles: str) -> str:
        """ORCA geometry optimisation and single point for one molecule without [*]. Returns Hartree."""
        try:
            return f"{self.orca.energy(smiles):.6f} Hartree"
        except (OrcaError, ValueError) as exc:
            return f"error: {exc}"

    def reaction_energy(self, monomers: str, polymer: str) -> str:
        """Energy change of monomers -> repeat unit + by-product, from ORCA, in kcal/mol per repeat unit.
        Negative means favourable. Run check_mass_balance first."""
        try:
            delta = self.reaction_energy_kcal(monomers, polymer)
        except (OrcaError, ValueError) as exc:
            return f"error: {exc}"
        self._delta_e = delta
        return f"{delta:.2f} kcal/mol per repeat unit"

    def reaction_energy_kcal(self, monomers, polymer):
        species = parse_species(monomers)
        ok, byproduct, message = balance(species, polymer)
        if not ok:
            raise ValueError(message)
        parts = decompose(byproduct)
        if parts is None:
            raise ValueError(f"by-product {formula_str(byproduct)} is not a known small molecule")
        # The capped repeat unit is the repeat unit plus H2, so subtract one H2.
        e_h2 = self.orca.energy(KNOWN_BYPRODUCTS["H2"][1])
        e_ru = self.orca.energy(capped(polymer)) - e_h2
        e_monomers = sum(n * self.orca.energy(smi) for smi, n in species.items())
        e_byproduct = sum(n * self.orca.energy(KNOWN_BYPRODUCTS[name][1]) for name, n in parts.items())
        return (e_ru + e_byproduct - e_monomers) * HARTREE_TO_KCAL


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)
