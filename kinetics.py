"""Activation energy by a restrained scan, and regioselectivity from competing barriers.

The scan pulls one forming bond, atom i to atom j, from its starting distance to r_end. At each
distance the rest of the structure is relaxed at GFN2-xTB with a harmonic restraint on that
distance. The barrier is the highest electronic energy on the path, minus the energy of the
separated reactants.

This is a scan estimate, not a transition-state search. The highest point on a scan is usually
above the true saddle point, so Ea is an upper estimate on a fixed reaction coordinate.
"""

import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from gfn2 import BOHR_PER_ANGSTROM, Gfn2Error

HARTREE_TO_KCAL = 627.5095
RESTRAINT_K = 0.5  # Hartree/Bohr^2, harmonic restraint on the scanned distance
R_KCAL = 1.987204e-3  # kcal/(mol K)
T_STD = 298.15


FRAGMENT_SPACING = 10.0  # Angstrom between the centres of separate reactant molecules


def combined_geometry(reactants_smiles):
    """Reactant molecules in one system, each embedded separately and placed apart along x.
    Returns (mol, numbers, positions in Bohr). The atoms are in the order of the dot-separated SMILES."""
    mol = None
    blocks = []
    for k, smi in enumerate(reactants_smiles.split(".")):
        frag = Chem.AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(frag, randomSeed=42) != 0:
            raise Gfn2Error(f"could not generate a 3D geometry for {smi}")
        AllChem.MMFFOptimizeMolecule(frag)
        xyz = frag.GetConformer().GetPositions()
        xyz = xyz - xyz.mean(axis=0) + np.array([k * FRAGMENT_SPACING, 0.0, 0.0])
        blocks.append(xyz)
        mol = frag if mol is None else Chem.CombineMols(mol, frag)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    positions = np.vstack(blocks) * BOHR_PER_ANGSTROM
    return mol, numbers, positions


def hydrogen_on(mol, heavy_index):
    """Index of a hydrogen bonded to a heavy atom. Used to pick the proton that is transferred."""
    for nbr in mol.GetAtomWithIdx(heavy_index).GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            return nbr.GetIdx()
    raise ValueError(f"atom {heavy_index} has no hydrogen")


def _restrained(runner, numbers, positions, charge, unpaired, i, j, target):
    """Restrained energy and gradient at one geometry: E + k (|r_i - r_j| - target)^2."""
    shape = positions.shape

    def objective(x):
        x3 = x.reshape(shape)
        res = runner.singlepoint(numbers, x3, charge, unpaired)
        energy = float(res.get("energy"))
        grad = np.asarray(res.get("gradient")).copy()
        diff = x3[i] - x3[j]
        d = np.linalg.norm(diff)
        energy += RESTRAINT_K * (d - target) ** 2
        if d > 1e-12:
            unit = diff / d
            push = 2 * RESTRAINT_K * (d - target)
            grad[i] += push * unit
            grad[j] -= push * unit
        return energy, grad.ravel()

    from scipy.optimize import minimize

    result = minimize(objective, positions.ravel(), jac=True, method="L-BFGS-B",
                      options={"gtol": 1e-4, "maxiter": 300})
    return float(result.fun), result.x.reshape(shape)


def scan_barrier(runner, reactants_smiles, pair, r_end=1.0, steps=16):
    """Electronic barrier Ea (kcal/mol) for the forming bond `pair` = (i, j) on a restrained scan.

    `runner` is a Gfn2Runner. The scan starts from the embedded structure, where the pair
    distance is d0, and ends at r_end (Angstrom). 1.0 A is needed to see the rise on the scan. Returns dict with ea, ts_distance (Angstrom)
    and the profile of (distance, energy relative to reactants) in kcal/mol.
    """
    mol, numbers, positions = combined_geometry(reactants_smiles)
    charge = Chem.GetFormalCharge(mol)
    unpaired = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    i, j = pair
    d0 = float(np.linalg.norm(positions[i] - positions[j]))
    r_end_bohr = r_end * BOHR_PER_ANGSTROM
    reference = sum(runner.energy(frag) for frag in reactants_smiles.split("."))

    profile = []
    current = positions
    for target in np.linspace(d0, r_end_bohr, steps):
        energy, current = _restrained(runner, numbers, current, charge, unpaired, i, j, target)
        profile.append((target / BOHR_PER_ANGSTROM, (energy - reference) * HARTREE_TO_KCAL))
    top = max(range(len(profile)), key=lambda k: profile[k][1])
    return {
        "ea": profile[top][1],
        "ts_distance": profile[top][0],
        "profile": profile,
        "at_end": top == len(profile) - 1,  # the maximum is at r_end, so the true barrier may be higher
    }


def boltzmann_fractions(barriers_kcal, T=T_STD):
    """Product fractions from the barriers of competing pathways, from Boltzmann weights exp(-Ea/RT)."""
    lowest = min(barriers_kcal)
    weights = [math.exp(-(b - lowest) / (R_KCAL * T)) for b in barriers_kcal]
    total = sum(weights)
    return [w / total for w in weights]
