import numpy as np
import pytest

from gfn2 import Gfn2Runner
from kinetics import FRAGMENT_SPACING, boltzmann_fractions, combined_geometry, hydrogen_on, scan_barrier

requires_tblite = pytest.mark.skipif(not Gfn2Runner.available(), reason="tblite is not installed")
PROPENE_HCL = "C=CC.Cl"


def proton_on_cl(mol):
    cl = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl")
    return hydrogen_on(mol, cl)


def test_fragments_start_apart():
    mol, numbers, positions = combined_geometry(PROPENE_HCL)
    propene, hcl = positions[: mol.GetNumAtoms() - 2], positions[-2:]
    gap = np.linalg.norm(propene.mean(axis=0) - hcl.mean(axis=0))
    assert gap == pytest.approx(FRAGMENT_SPACING / 0.52917721092, rel=0.2)  # Bohr


def test_hydrogen_on_finds_the_proton_of_hcl():
    mol, _, _ = combined_geometry(PROPENE_HCL)
    h = proton_on_cl(mol)
    assert mol.GetAtomWithIdx(h).GetSymbol() == "H"
    assert any(n.GetSymbol() == "Cl" for n in mol.GetAtomWithIdx(h).GetNeighbors())


def test_hydrogen_on_rejects_a_carbon_without_hydrogen():
    mol, _, _ = combined_geometry("CC(C)(C)C")  # neopentane: the central carbon has no hydrogen
    with pytest.raises(ValueError, match="has no hydrogen"):
        hydrogen_on(mol, 1)


def test_boltzmann_equal_barriers_split_evenly():
    assert boltzmann_fractions([10.0, 10.0]) == pytest.approx([0.5, 0.5])


def test_lower_barrier_gets_the_larger_fraction_and_fractions_sum_to_one():
    fractions = boltzmann_fractions([3.0, 6.7, 10.0])
    assert sum(fractions) == pytest.approx(1.0)
    assert fractions[0] > fractions[1] > fractions[2]


@requires_tblite
def test_markovnikov_barrier_is_lower_than_anti_markovnikov():
    """Protonation at the terminal CH2 gives the secondary carbocation, so its barrier is lower."""
    runner = Gfn2Runner()
    mol, _, _ = combined_geometry(PROPENE_HCL)
    h = proton_on_cl(mol)
    markovnikov = scan_barrier(runner, PROPENE_HCL, (0, h))
    anti = scan_barrier(runner, PROPENE_HCL, (1, h))
    assert markovnikov["ea"] < anti["ea"]
    assert markovnikov["profile"][0][1] == pytest.approx(0.0, abs=0.5)  # starts at the separated reactants
