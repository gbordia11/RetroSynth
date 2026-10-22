"""Validate one polymerization reaction from a dataset row: gate, ΔG and ΔH per repeat unit, and the validator verdict.

The reaction is reactant1 + reactant2 -> Polymer_smiles. The by-product (for example 2 HCl) is
found by atom balance.

Usage:
    python example_polymer.py --csv data.csv --pid PT00001              # LLM review included
    python example_polymer.py --csv data.csv --pid PT00001 --no-llm     # energies only
"""

import argparse

from config import load_config
from dataset import find_row, row_reactants
from molecules import TreeNode
from validator import RouteValidator, format_deltas


def main():
    parser = argparse.ArgumentParser(description="Validate a polymerization reaction from a dataset row")
    parser.add_argument("--csv", required=True, help="dataset CSV")
    parser.add_argument("--pid", required=True, help="PID of the row to validate")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM review, energies only")
    args = parser.parse_args()

    config = load_config()
    if config.validator.backend != "gfn2":
        raise SystemExit("this example needs backend = gfn2 in [validator]")

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
        deltas = validator.step_deltas(route.smiles, counts)
        verdict_text = "favourable" if all(v < 0 for v in deltas.values()) else "not favourable"
        print(f"  {format_deltas(deltas)} per repeat unit ({verdict_text})")
        return

    verdict = validator(route)
    step = verdict.steps[0]
    print(f"  gate: {'ok' if step.gate_ok else 'FAIL'} ({step.message})")
    if step.deltas:
        print(f"  {format_deltas(step.deltas)} per repeat unit")
    else:
        print(f"  energies not computed: {step.note}")
    if step.ceiling_temperature is not None:
        print(f"  ceiling temperature {step.ceiling_temperature:.0f} K")
    print(f"  validator: valid={verdict.valid}, favourable={verdict.favourable}")
    print(f"  {verdict.rationale}")


if __name__ == "__main__":
    main()
