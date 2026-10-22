"""Reading polymerization reactions from the dataset CSV (for example polymer_templates.csv).

Each row is one reaction: reactant1 + reactant2 -> Polymer_smiles (PSMILES with two [*] end points).
A homopolymer lists the same monomer twice. Its repeat unit takes one monomer, so the monomer is
counted once.
"""

import csv

from molecules import canonical


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def find_row(path, pid):
    for row in load_rows(path):
        if row["PID"] == pid:
            return row
    raise KeyError(f"PID {pid} not found in {path}")


def row_reactants(row):
    """{canonical SMILES: count} of the monomers that make one repeat unit."""
    r1, r2 = canonical(row["reactant1"]), canonical(row["reactant2"])
    return {r1: 1} if r1 == r2 else {r1: 1, r2: 1}
