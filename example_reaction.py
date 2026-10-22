"""Example: validate a reaction and report its energies.

Two modes:

1. Default, addition of HCl to propene. Each candidate is a protonation site, and the script
   reports ΔG, the barrier Ea from a restrained scan, and regioselectivity.

2. A row from a polymerization dataset (--csv FILE --pid PID). The reaction is
   reactant1 + reactant2 -> Polymer_smiles, with the by-product found by atom balance.
   The script reports the gate, ΔG per repeat unit and the validator verdict. Barrier and
   regioselectivity are not computed: for symmetric step-growth monomers every reacting group
   is equivalent, so there is no site choice to compare.

Usage:
    python example_reaction.py                                  # propene + HCl, needs MISTRAL_API_KEY
    python example_reaction.py --no-llm                         # energies only
    python example_reaction.py --csv data.csv --pid PT00001     # one dataset row
"""

import argparse
import csv

from config import load_config
from kinetics import boltzmann_fractions, combined_geometry, hydrogen_on, scan_barrier
from molecules import TreeNode, canonical
from validator import RouteValidator

REACTANTS = "C=CC.Cl"  # propene + HCl
DONOR = "Cl"  # the element that carries the transferred proton (HCl)

CANDIDATES = [
    {"name": "protonate C1 (Markovnikov)", "site": 0, "product": "CC(C)Cl"},
    {"name": "protonate C2 (anti-Markovnikov)", "site": 1, "product": "CCCCl"},
]


def evaluate(reactants, candidates, donor, config, use_llm=True):
    """Return one result dict per candidate: name, product, dG, Ea, ts_distance, at_end, fraction, valid."""
    validator = RouteValidator(config.validator)
    mol, _, _ = combined_geometry(reactants)
    donor_index = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == donor)
    proton = hydrogen_on(mol, donor_index)
    reactant_smiles = [canonical(s) for s in reactants.split(".")]

    results = []
    for cand in candidates:
        pair = (cand["site"], proton)
        product = canonical(cand["product"])
        route = TreeNode(product, 1.0, [TreeNode(s) for s in reactant_smiles])

        if use_llm:
            verdict = validator(route)
            step = verdict.steps[0]
            dg, valid = step.delta, verdict.valid
        else:
            step_reactants = {s: reactant_smiles.count(s) for s in set(reactant_smiles)}
            dg = validator.step_delta(product, step_reactants)
            valid = dg < 0

        scan = scan_barrier(validator.energies, reactants, pair)
        results.append({
            "name": cand["name"],
            "product": product,
            "dG": dg,
            "Ea": scan["ea"],
            "ts_distance": scan["ts_distance"],
            "at_end": scan["at_end"],
            "valid": valid,
        })

    fractions = boltzmann_fractions([r["Ea"] for r in results])
    for r, frac in zip(results, fractions):
        r["fraction"] = frac
    return results


def print_table(results, criterion):
    print(f"{'candidate':<32}{'product':<14}{'dG' if criterion == 'dg' else 'dE':>10}{'Ea':>10}"
          f"{'TS (A)':>9}{'ratio':>9}  valid")
    for r in results:
        note = " *" if r["at_end"] else ""
        print(f"{r['name']:<32}{r['product']:<14}{r['dG']:>10.2f}{r['Ea']:>10.2f}"
              f"{r['ts_distance']:>9.2f}{100 * r['fraction']:>8.1f}%  {r['valid']}{note}")
    print("kcal/mol. ratio = Boltzmann fraction of product at 298 K from the barriers.")
    if any(r["at_end"] for r in results):
        print("* The barrier maximum is at the end of the scan, so the true barrier may be higher.")


def route_from_row(row):
    """Route tree for one dataset row: the product is Polymer_smiles (PSMILES), the reactants are reactant1 and reactant2."""
    reactants = [canonical(row["reactant1"]), canonical(row["reactant2"])]
    return TreeNode(canonical(row["Polymer_smiles"]), 1.0, [TreeNode(s) for s in reactants])


def find_row(path, pid):
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["PID"] == pid:
                return row
    raise SystemExit(f"PID {pid} not found in {path}")


def report_row(row, config, use_llm):
    validator = RouteValidator(config.validator)
    route = route_from_row(row)
    print(f"PID {row['PID']}: {row['reactant1_name']} + {row['reactant2_name']} -> {row['common_name'] or row['Polymer_smiles']}")
    print(f"  product (PSMILES): {row['Polymer_smiles']}")
    if use_llm:
        verdict = validator(route)
        step = verdict.steps[0]
        print(f"  gate: {'ok' if step.gate_ok else 'FAIL'} ({step.message})")
        print(f"  dG = {step.delta:.2f} kcal/mol per repeat unit" if step.delta is not None
              else f"  dG not computed: {step.note}")
        print(f"  validator: valid={verdict.valid}")
        print(f"  {verdict.rationale}")
    else:
        step_reactants = {}
        for node in route.children:
            step_reactants[node.smiles] = step_reactants.get(node.smiles, 0) + 1
        dg = validator.step_delta(route.smiles, step_reactants)
        print(f"  dG = {dg:.2f} kcal/mol per repeat unit ({'favourable' if dg < 0 else 'not favourable'})")
    print("  barrier and regioselectivity: not computed for this step-growth reaction (see module docstring)")


def main():
    parser = argparse.ArgumentParser(description="Validate a reaction: ΔG, and Ea and regioselectivity for propene + HCl")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM review, energies only")
    parser.add_argument("--csv", help="dataset CSV; with --pid, validates one row instead of the propene example")
    parser.add_argument("--pid", help="PID of the row to validate")
    args = parser.parse_args()

    config = load_config()
    if config.validator.backend != "gfn2" or config.validator.criterion != "dg":
        raise SystemExit("this example needs backend = gfn2 and criterion = dg in [validator]")

    if args.csv:
        if not args.pid:
            raise SystemExit("--csv needs --pid")
        report_row(find_row(args.csv, args.pid), config, use_llm=not args.no_llm)
        return
    results = evaluate(REACTANTS, CANDIDATES, DONOR, config, use_llm=not args.no_llm)
    print_table(results, config.validator.criterion)


if __name__ == "__main__":
    main()
