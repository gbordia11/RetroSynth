"""Validate one polymerization reaction from a dataset row: gate, ΔG per repeat unit, and the validator verdict.

The reaction is reactant1 + reactant2 -> Polymer_smiles. The by-product (for example 2 HCl) is
found by atom balance. The product is passed as written in the CSV, because canonical() turns
[*] end points into *, which the validator does not recognise as end points.

Usage:
    python example_polymer.py --csv data.csv --pid PT00001              # LLM review included
    python example_polymer.py --csv data.csv --pid PT00001 --no-llm     # energies only
"""

import argparse

from config import load_config
from dataset import find_row, row_reactants
from molecules import TreeNode
from validator import RouteValidator


def main():
    parser = argparse.ArgumentParser(description="Validate a polymerization reaction from a dataset row")
    parser.add_argument("--csv", required=True, help="dataset CSV")
    parser.add_argument("--pid", required=True, help="PID of the row to validate")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM review, energies only")
    args = parser.parse_args()

    config = load_config()
    if config.validator.backend != "gfn2" or config.validator.criterion != "dg":
        raise SystemExit("this example needs backend = gfn2 and criterion = dg in [validator]")

    try:
        row = find_row(args.csv, args.pid)
    except KeyError as exc:
        raise SystemExit(str(exc))
    counts = row_reactants(row)
    route = TreeNode(row["Polymer_smiles"], 1.0, [TreeNode(s) for s in counts for _ in range(counts[s])])
    validator = RouteValidator(config.validator)

    print(f"PID {row['PID']}: {row['reactant1_name']} + {row['reactant2_name']} -> {row['common_name'] or row['Polymer_smiles']}")
    print(f"  repeat unit (PSMILES): {row['Polymer_smiles']}")

    if args.no_llm:
        dg = validator.step_delta(route.smiles, counts)
        verdict_text = "favourable" if dg < 0 else "not favourable"
        print(f"  dG = {dg:.2f} kcal/mol per repeat unit ({verdict_text})")
        return

    verdict = validator(route)
    step = verdict.steps[0]
    print(f"  gate: {'ok' if step.gate_ok else 'FAIL'} ({step.message})")
    if step.delta is not None:
        print(f"  dG = {step.delta:.2f} kcal/mol per repeat unit")
    else:
        print(f"  dG not computed: {step.note}")
    print(f"  validator: valid={verdict.valid}, favourable={verdict.favourable}")
    print(f"  {verdict.rationale}")


if __name__ == "__main__":
    main()
