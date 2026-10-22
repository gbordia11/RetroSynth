"""Claude as a tool-calling orchestrator for polymer (or small-molecule) retrosynthesis.

Claude reads the target, proposes monomers and a forward reaction path, and checks each step with
tools that wrap the validator (agent 5): the chemistry gate, then the energy changes from the
[validator] backend and criterion (dG and dH from GFN2-xTB by default). It revises steps that fail
and returns the path as reaction SMILES.

Claude's answer is then re-checked without the LLM: every step must pass the gate and have every
configured energy change below zero, and every starting material must be in stock. Claude can
propose a route, but only the tools can make it valid.

Usage:
    python orchestrator.py "[*]CC[*]" --stock stock.smi [--context TEXT] [--config config.ini]
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field

import dspy

from config import DEFAULT_CONFIG_PATH, Config, load_config
from molecules import TreeNode, canonical, load_stock
from ranker import RouteRanker
from renderer import RouteRenderer
from validator import (
    ENERGY_ERRORS,
    KNOWN_BYPRODUCTS,
    RouteValidator,
    StepCheck,
    check_step_gate,
    format_deltas,
    parse_species,
    route_steps,
)


class PlanSynthesis(dspy.Signature):
    """Plan a synthesis of the target from purchasable starting materials.
    The target is a molecule, or a polymer repeat unit in PSMILES with two [*] end points.
    For a polymer, the last step turns the monomers of one repeat unit into the repeat unit,
    for example C=C>>[*]CC[*] or Nc1ccc(N)cc1.O=C(Cl)c1ccc(C(=O)Cl)cc1>>[*]Nc1ccc(NC(=O)c2ccc(C([*])=O)cc2)cc1.
    Write reactants only; by-products are found by atom balance and must be small molecules
    from this list: {byproducts}.
    Check every step with validate_step before you return it; it runs the atom-balance gate and
    computes the energy changes, and a step passes only if all of them are below zero. Check
    starting materials with in_stock. If a step fails, revise the monomers or the step and validate
    again. Return only steps that passed validate_step."""

    target: str = dspy.InputField(desc="SMILES of the target molecule, or PSMILES of the polymer repeat unit")
    context: str = dspy.InputField(desc="extra instructions from the chemist, may be empty")
    steps: list[str] = dspy.OutputField(
        desc="reaction SMILES in forward order, 'reactant1.reactant2>>product'; the last product is the target"
    )
    rationale: str = dspy.OutputField(desc="why these monomers and steps, citing the energy changes")


# dspy reads the instructions when the class is built, so the by-product list is filled in here.
PlanSynthesis = PlanSynthesis.with_instructions(PlanSynthesis.__doc__.format(byproducts=", ".join(KNOWN_BYPRODUCTS)))


@dataclass
class OrchestratorResult:
    valid: bool  # every step passed the gate and the energy criteria, and every leaf is in stock
    rationale: str  # Claude's explanation
    steps: list = field(default_factory=list)  # StepCheck, in forward order (starting materials first)
    issues: list = field(default_factory=list)  # why the route is not valid
    root: TreeNode | None = None  # the route as a tree, as the search agent would return it
    trajectory: dict = field(default_factory=dict)  # Claude's thoughts and tool calls


def parse_reaction(text):
    """'A.B>>P' -> ([canonical A, canonical B], canonical P). Raise ValueError if it cannot be read."""
    left, sep, right = text.partition(">>")
    if not sep:
        raise ValueError(f"not a reaction SMILES (needs '>>'): {text!r}")
    reactants = [canonical(r) for r in left.split(".") if r.strip()]
    product = canonical(right.strip())
    if not reactants or None in reactants or product is None:
        raise ValueError(f"reaction SMILES has an unreadable molecule: {text!r}")
    return reactants, product


def route_from_steps(steps, target):
    """Route tree from reaction SMILES in forward order. Return (root or None, issues).
    A reactant made by an earlier step becomes that step's subtree; any other reactant is a leaf."""
    made, issues = {}, []
    for text in steps:
        try:
            reactants, product = parse_reaction(text)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        children = [made.pop(r) if r in made else TreeNode(r) for r in reactants]
        made[product] = TreeNode(product, 1.0, children)
    root = made.pop(canonical(target), None)
    if root is None:
        issues.append(f"no step makes the target {target}")
    if made:
        issues.append(f"intermediates made but not used: {', '.join(made)}")
    return root, issues


class RetroOrchestrator(dspy.Module):
    """Claude proposes monomers and steps, validates them with tools, and the result is re-checked."""

    def __init__(self, stock, config=None):
        super().__init__()
        self.config = config or Config()
        self.stock = stock
        self.validator = RouteValidator(self.config.validator)  # gate and energies; its LLM review is not used
        self.lm = dspy.LM(self.config.orchestrator.model, max_tokens=self.config.orchestrator.max_tokens)
        self.agent = dspy.ReAct(
            PlanSynthesis,
            tools=[self.validate_step, self.in_stock],
            max_iters=self.config.orchestrator.max_iters,
        )

    def forward(self, target, context=""):
        with dspy.context(lm=self.lm):
            pred = self.agent(target=target, context=context)
        steps = pred.steps if isinstance(pred.steps, list) else str(pred.steps).splitlines()
        result = self.check(target, [s for s in steps if s.strip()])
        result.rationale = pred.rationale
        result.trajectory = dict(pred.trajectory)
        return result

    def check(self, target, steps):
        """Re-check a proposed path without the LLM: gate, energy criteria, and stock, for every step."""
        root, issues = route_from_steps(steps, target)
        checks = []
        if root is not None:
            for product, reactants in reversed(route_steps(root)):
                checks.append(self._check_step(product, Counter(reactants)))
            for s in checks:
                if not s.gate_ok:
                    issues.append(f"{s.product}: gate failed: {s.message}")
                elif not s.deltas:
                    issues.append(f"{s.product}: no energy ({s.note})")
                elif any(v >= 0 for v in s.deltas.values()):
                    issues.append(f"{s.product}: not favourable: {format_deltas(s.deltas)}")
            issues += [f"not in stock: {leaf.smiles}" for leaf in root.leaves() if leaf.smiles not in self.stock]
        return OrchestratorResult(valid=root is not None and not issues, rationale="", steps=checks, issues=issues, root=root)

    def _check_step(self, product, reactants):
        """StepCheck for one step; `reactants` is a {smiles: count} dict. Energies only if the gate passes."""
        ok, _, message = check_step_gate(product, reactants)
        step = StepCheck(product, list(reactants.elements()), ok, message)
        if ok:
            try:
                step.deltas = self.validator.step_deltas(product, reactants)
            except ENERGY_ERRORS as exc:
                step.note = str(exc)
        return step

    # Tools called by Claude. They return text, and errors come back as text too,
    # so Claude can read them and revise the step.

    def validate_step(self, product: str, reactants: str) -> str:
        """Validate one step, reactants -> product: atom-balance gate, then the energy changes in kcal/mol.
        product: SMILES, or PSMILES with [*] end points for a polymer repeat unit.
        reactants: comma-separated SMILES, each with an optional ':count', e.g. 'C=C' or 'Nc1ccc(N)cc1, O=C(Cl)c1ccc(C(=O)Cl)cc1'."""
        if canonical(product) is None:
            return f"FAIL: product is not valid SMILES: {product!r}"
        try:
            species = parse_species(reactants)
        except ValueError as exc:
            return f"FAIL: {exc}"
        step = self._check_step(product, Counter(species))
        if not step.gate_ok:
            return f"FAIL: gate failed: {step.message}"
        if not step.deltas:
            return f"FAIL: gate ok ({step.message}), but no energy: {step.note}"
        verdict = "PASS" if all(v < 0 for v in step.deltas.values()) else "FAIL: not favourable"
        text = f"{verdict}: gate ok ({step.message}); {format_deltas(step.deltas)}"
        if step.ceiling_temperature is not None:
            text += f"; ceiling temperature {step.ceiling_temperature:.0f} K"
        return text

    def in_stock(self, smiles: str) -> str:
        """Whether starting materials are purchasable. smiles: comma-separated SMILES."""
        lines = []
        for smi in (s.strip() for s in smiles.split(",") if s.strip()):
            can = canonical(smi)
            status = "invalid SMILES" if can is None else ("in stock" if can in self.stock else "NOT in stock")
            lines.append(f"{smi}: {status}")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Claude orchestrator: monomers and validated reaction steps")
    parser.add_argument("target", help="target SMILES, or polymer PSMILES with [*] end points")
    parser.add_argument("--stock", required=True, help="building blocks file, one SMILES per line")
    parser.add_argument("--context", default="", help="extra instructions for Claude")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="settings file (INI)")
    args = parser.parse_args()

    config = load_config(args.config)
    stock = load_stock(args.stock)
    result = RetroOrchestrator(stock, config)(args.target, context=args.context)

    n = sum(1 for key in result.trajectory if key.startswith("tool_name_"))
    print(f"Claude made {n} tool call(s):")
    for i in range(n):
        args_i = result.trajectory.get(f"tool_args_{i}", {})
        print(f"  {result.trajectory[f'tool_name_{i}']}({args_i})")
        observation = str(result.trajectory.get(f"observation_{i}", "")).splitlines()
        print(f"    -> {observation[0] if observation else ''}")
    print(f"Route valid={result.valid}")
    for i, s in enumerate(result.steps, 1):
        gate = "ok" if s.gate_ok else "FAIL"
        energy = f" {format_deltas(s.deltas)}" if s.deltas else ""
        tc = f" Tc={s.ceiling_temperature:.0f} K" if s.ceiling_temperature is not None else ""
        print(f"  step {i}: {' + '.join(s.reactants)} -> {s.product}: gate={gate}{energy}{tc}")
    for issue in result.issues:
        print(f"  ! {issue}")
    print(f"Claude: {result.rationale}")
    if result.root is not None and result.root.children:
        for path in RouteRenderer(config.renderer)(RouteRanker(stock)([result.root])):
            print(f"Image: {path}")


if __name__ == "__main__":
    main()
