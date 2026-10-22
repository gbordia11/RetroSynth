from pathlib import Path

import pytest

from dataset import find_row, load_rows, row_reactants
from molecules import canonical
from validator import check_step_gate, decompose, formula
from collections import Counter

DATASET = Path(__file__).resolve().parent.parent / "polymer_templates.csv"
requires_dataset = pytest.mark.skipif(not DATASET.exists(), reason="polymer_templates.csv is not in the project")


@requires_dataset
def test_every_row_passes_the_chemistry_gate():
    """All 2000 reactions balance: the by-products are known small molecules, and homopolymers count the monomer once."""
    failures = []
    for row in load_rows(DATASET):
        ok, _, message = check_step_gate(row["Polymer_smiles"], row_reactants(row))
        if not ok:
            failures.append((row["PID"], message))
    assert failures == []


@requires_dataset
def test_homopolymer_counts_its_monomer_once():
    row = find_row(DATASET, "PT00352")  # 2,3-dimethylbutadiene + itself
    assert row_reactants(row) == {canonical(row["reactant1"]): 1}


@requires_dataset
def test_copolymer_counts_both_monomers():
    row = find_row(DATASET, "PT00001")  # TFMB + the diacid chloride
    assert sorted(row_reactants(row).values()) == [1, 1]


@requires_dataset
def test_hf_and_hbr_by_products_are_recognised():
    assert decompose(Counter(H=2, F=2)) == {"HF": 2}
    assert decompose(Counter(H=2, Br=2)) == {"HBr": 2}


def test_formula_counts_a_known_monomer():
    assert formula("C=C(C)C(=C)C") == Counter(C=6, H=10)
