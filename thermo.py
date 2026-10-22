"""Ideal-gas thermochemistry (rigid rotor, harmonic oscillator) at 298.15 K and 1 atm.

Input: a relaxed geometry, its atomic masses, and its harmonic vibrational frequencies.
Output: the thermal correction to the Gibbs energy, G_corr = H_corr - T*S, in J/mol.
Adding G_corr to the electronic energy gives the free energy G.

Known approximation: the rotational symmetry number is taken as 1 by default. A symmetric
molecule's entropy is then too high by R*ln(sigma), about 0.6 kcal/mol at 298 K for sigma = 4.
"""

import math
from dataclasses import dataclass

import numpy as np

H_PLANCK = 6.62607015e-34  # J s
KB = 1.380649e-23  # J/K
NA = 6.02214076e23  # 1/mol
R = KB * NA  # J/(mol K)
C_CM = 2.99792458e10  # speed of light, cm/s
AMU = 1.66053906660e-27  # kg
BOHR = 5.29177210903e-11  # m
HARTREE_J = 4.3597447222071e-18  # J

T_STD = 298.15  # K
P_STD = 101325.0  # Pa, 1 atm
IMAGINARY_CM = -50.0  # modes below this are imaginary: the geometry is not a minimum
NOISE_CM = 10.0  # modes between this and zero are numerical noise and are ignored


@dataclass
class Thermo:
    zpe: float  # zero-point energy, J/mol
    entropy: float  # S, J/(mol K)
    h_corr: float  # H_corr, J/mol (the enthalpy term inside G_corr)
    g_corr: float  # G_corr = H_corr - T*S, J/mol
    n_imaginary: int


def inertia_eigenvalues(positions, masses):
    """Principal moments of inertia, ascending, in amu*Bohr^2. positions in Bohr, masses in amu."""
    com = np.average(positions, axis=0, weights=masses)
    r = positions - com
    tensor = np.zeros((3, 3))
    for ri, mi in zip(r, masses):
        tensor += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    return np.linalg.eigvalsh(tensor)


def is_linear(inertia):
    return inertia[0] < 1e-6 * max(inertia[2], 1e-12)


def mass_weighted_frequencies(hessian, masses, linear):
    """Harmonic frequencies in cm^-1 from a Cartesian Hessian (Hartree/Bohr^2).
    Imaginary modes are negative. The 5 (linear) or 6 lowest-magnitude modes are the
    translations and rotations, and are dropped."""
    m = np.repeat(masses, 3)
    mass_weighted = hessian / np.sqrt(np.outer(m, m))
    eigenvalues = np.linalg.eigvalsh(mass_weighted)  # Hartree/(Bohr^2 amu)
    eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))]
    vib = eigenvalues[5 if linear else 6:]
    conversion = HARTREE_J / (BOHR**2 * AMU)  # to 1/s^2
    omega = np.sign(vib) * np.sqrt(np.abs(vib) * conversion)  # rad/s
    return omega / (2 * math.pi * C_CM)  # cm^-1


def rrho_thermo(positions, masses, frequencies, sigma=1, T=T_STD, P=P_STD):
    """Gibbs energy correction from translation, rotation and harmonic vibration."""
    n_imaginary = int(np.sum(frequencies < IMAGINARY_CM))
    vib = frequencies[(frequencies > NOISE_CM)]
    inertia = inertia_eigenvalues(positions, masses)
    linear = is_linear(inertia)

    # Translation (Sackur-Tetrode)
    mass_kg = float(np.sum(masses)) * AMU
    lam = H_PLANCK / math.sqrt(2 * math.pi * mass_kg * KB * T)  # thermal wavelength, m
    s_trans = R * (math.log(T * KB / (P * lam**3)) + 2.5)
    h_trans = 2.5 * R * T

    # Rotation: theta = h^2 / (8 pi^2 I k) for each principal moment
    thetas = [
        H_PLANCK**2 / (8 * math.pi**2 * (i * AMU * BOHR**2) * KB) for i in inertia if i > 1e-12
    ]
    if linear:
        s_rot = R * (math.log(T / (sigma * thetas[-1])) + 1.0)
        h_rot = R * T
    else:
        s_rot = R * (math.log(math.sqrt(math.pi) / sigma * math.sqrt(T**3 / math.prod(thetas))) + 1.5)
        h_rot = 1.5 * R * T

    # Harmonic vibration
    zpe = 0.0
    h_vib = 0.0
    s_vib = 0.0
    for nu in vib:
        theta_v = H_PLANCK * C_CM * nu / KB  # vibrational temperature, K
        x = theta_v / T
        zpe += 0.5 * R * theta_v
        h_vib += R * theta_v / math.expm1(x)
        s_vib += R * (x / math.expm1(x) - math.log(1 - math.exp(-x)))

    h_corr = zpe + h_vib + h_trans + h_rot + R * T  # RT is the PV term of an ideal gas
    entropy = s_trans + s_rot + s_vib
    return Thermo(zpe=zpe, entropy=entropy, h_corr=h_corr, g_corr=h_corr - T * entropy, n_imaginary=n_imaginary)
