import numpy as np
import pytest

from thermo import T_STD, inertia_eigenvalues, is_linear, rrho_thermo

# H2 at its experimental bond length, with its harmonic stretch frequency.
H2_BOND_BOHR = 0.7414 / 0.52917721092
H2_POSITIONS = np.array([[0.0, 0.0, -H2_BOND_BOHR / 2], [0.0, 0.0, H2_BOND_BOHR / 2]])
H2_MASSES = np.array([1.00783, 1.00783])
H2_NU = np.array([4401.0])  # cm^-1


def test_h2_standard_entropy_matches_reference():
    # Reference: S(298.15 K, 1 bar) = 130.68 J/(mol K), which is 130.57 at 1 atm.
    thermo = rrho_thermo(H2_POSITIONS, H2_MASSES, H2_NU, sigma=2)
    assert thermo.entropy == pytest.approx(130.6, abs=1.0)


def test_h2_zero_point_energy_is_harmonic_value():
    # ZPE = 1/2 h c nu N_A = 26.3 kJ/mol for 4401 cm^-1.
    thermo = rrho_thermo(H2_POSITIONS, H2_MASSES, H2_NU, sigma=2)
    assert thermo.zpe / 1000 == pytest.approx(26.3, abs=0.1)


def test_gibbs_correction_combines_enthalpy_and_entropy():
    thermo = rrho_thermo(H2_POSITIONS, H2_MASSES, H2_NU, sigma=2)
    assert thermo.g_corr < thermo.zpe  # the TS term outweighs the thermal enthalpy for H2


def test_enthalpy_correction_is_gibbs_correction_plus_ts():
    thermo = rrho_thermo(H2_POSITIONS, H2_MASSES, H2_NU, sigma=2)
    assert thermo.h_corr == pytest.approx(thermo.g_corr + T_STD * thermo.entropy)


def test_h2_is_linear_and_nonlinear_water_is_not():
    assert is_linear(inertia_eigenvalues(H2_POSITIONS, H2_MASSES))
    water = np.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]])  # Bohr, bent
    assert not is_linear(inertia_eigenvalues(water, np.array([15.999, 1.008, 1.008])))


def test_imaginary_modes_are_counted():
    thermo = rrho_thermo(H2_POSITIONS, H2_MASSES, np.array([4401.0, -300.0]), sigma=2)
    assert thermo.n_imaginary == 1


def test_noise_frequencies_are_ignored():
    with_noise = rrho_thermo(H2_POSITIONS, H2_MASSES, np.array([4401.0, 2.0]), sigma=2)
    without = rrho_thermo(H2_POSITIONS, H2_MASSES, np.array([4401.0]), sigma=2)
    assert with_noise.g_corr == pytest.approx(without.g_corr)
