"""GFN2-xTB energies and free energies through the tblite library. No external binary is needed.

For each molecule: RDKit embeds a 3D geometry, GFN2-xTB relaxes it with L-BFGS, and the
electronic energy of the relaxed structure is returned in Hartree.

The enthalpy and free energy add the thermal corrections from a numerical Hessian (finite
differences of the GFN2 gradient) and rigid-rotor, harmonic-oscillator thermochemistry at 298.15 K.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import minimize

from thermo import (
    HARTREE_J,
    IMAGINARY_CM,
    NA,
    inertia_eigenvalues,
    is_linear,
    mass_weighted_frequencies,
    rrho_thermo,
)

BOHR_PER_ANGSTROM = 1 / 0.52917721092
HESSIAN_STEP = 0.005  # Bohr, for the finite-difference Hessian


class Gfn2Error(RuntimeError):
    pass


class Gfn2Runner:
    """Relaxes molecules at GFN2-xTB and returns their electronic energies, enthalpies and free energies."""

    def __init__(self):
        self._relaxed = {}
        self._thermo = {}

    @staticmethod
    def available():
        try:
            import tblite.interface  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def geometry(smiles):
        """RDKit 3D starting structure: numbers (Z), positions (Bohr), charge, unpaired electrons."""
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            raise Gfn2Error(f"could not generate a 3D geometry for {smiles}")
        AllChem.MMFFOptimizeMolecule(mol)
        numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
        positions = mol.GetConformer().GetPositions() * BOHR_PER_ANGSTROM
        charge = Chem.GetFormalCharge(mol)
        unpaired = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        return numbers, positions, charge, unpaired

    @staticmethod
    def singlepoint(numbers, positions, charge, unpaired):
        """GFN2-xTB result (energy in Hartree, gradient in Hartree/Bohr) at one geometry."""
        from tblite.interface import Calculator

        try:
            calc = Calculator("GFN2-xTB", numbers, positions, charge=float(charge), uhf=unpaired)
            calc.set("verbosity", 0)
            return calc.singlepoint()
        except Exception as exc:  # tblite raises its own error types; re-raise them uniformly
            raise Gfn2Error(f"GFN2-xTB failed: {exc}") from exc

    def relax(self, smiles):
        """Relaxed structure of a molecule: (numbers, positions in Bohr, charge, unpaired, energy). Cached."""
        key = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        if key in self._relaxed:
            return self._relaxed[key]
        numbers, positions, charge, unpaired = self.geometry(key)
        shape = positions.shape

        def objective(x):
            res = self.singlepoint(numbers, x.reshape(shape), charge, unpaired)
            return float(res.get("energy")), np.asarray(res.get("gradient")).ravel()

        result = minimize(
            objective, positions.ravel(), jac=True, method="L-BFGS-B",
            options={"gtol": 1e-4, "maxiter": 500},
        )
        if not result.success:
            raise Gfn2Error(f"GFN2-xTB geometry optimisation did not converge: {result.message}")
        self._relaxed[key] = (numbers, result.x.reshape(shape), charge, unpaired, float(result.fun))
        return self._relaxed[key]

    def energy(self, smiles):
        """Electronic energy (Hartree) of the GFN2-xTB relaxed structure of a molecule."""
        return self.relax(smiles)[4]

    def hessian(self, numbers, positions, charge, unpaired):
        """Cartesian Hessian (Hartree/Bohr^2) by central differences of the GFN2 gradient."""
        flat = positions.ravel()
        n = flat.size
        shape = positions.shape

        def gradient(x):
            res = self.singlepoint(numbers, x.reshape(shape), charge, unpaired)
            return np.asarray(res.get("gradient")).ravel()

        hess = np.zeros((n, n))
        for i in range(n):
            plus, minus = flat.copy(), flat.copy()
            plus[i] += HESSIAN_STEP
            minus[i] -= HESSIAN_STEP
            hess[:, i] = (gradient(plus) - gradient(minus)) / (2 * HESSIAN_STEP)
        return (hess + hess.T) / 2

    def thermochemistry(self, smiles):
        """(electronic energy in Hartree, Thermo) of the relaxed structure. One Hessian per molecule, cached.
        Raises Gfn2Error if the structure has an imaginary mode, because it is not a minimum."""
        key = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        if key in self._thermo:
            return self._thermo[key]
        numbers, positions, charge, unpaired, energy = self.relax(key)
        masses = np.array([Chem.GetPeriodicTable().GetAtomicWeight(int(z)) for z in numbers])
        hess = self.hessian(numbers, positions, charge, unpaired)
        positions3 = positions.reshape(-1, 3)
        linear = is_linear(inertia_eigenvalues(positions3, masses))
        freqs = mass_weighted_frequencies(hess, masses, linear)
        thermo = rrho_thermo(positions3, masses, freqs)
        if thermo.n_imaginary:
            raise Gfn2Error(
                f"{key} has an imaginary mode below {IMAGINARY_CM:.0f} cm^-1; the structure is not a minimum"
            )
        self._thermo[key] = (energy, thermo)
        return self._thermo[key]

    def free_energy(self, smiles):
        """Free energy G (Hartree) at 298.15 K and 1 atm: electronic energy plus thermal correction."""
        energy, thermo = self.thermochemistry(smiles)
        return energy + thermo.g_corr / (NA * HARTREE_J)

    def enthalpy(self, smiles):
        """Enthalpy H (Hartree) at 298.15 K: electronic energy plus ZPE, thermal energy and PV."""
        energy, thermo = self.thermochemistry(smiles)
        return energy + thermo.h_corr / (NA * HARTREE_J)
