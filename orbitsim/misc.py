from typing import Tuple

import numpy as np

from orbit.utils.consts import classical_proton_radius
from orbit.utils.consts import speed_of_light
from orbit.core.orbit_utils import Matrix


def orbit_matrix_to_numpy(matrix: Matrix) -> np.ndarray:
    array = np.zeros(matrix.size())
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] = matrix.get(i, j)
    return array


def lorentz_factors(mass: float, kin_energy: float) -> Tuple[float]:
    gamma = 1.0 + (kin_energy / mass)
    beta = np.sqrt(gamma**2 - 1.0) / gamma
    return (gamma, beta)


def momentum(mass: float, kin_energy: float) -> float:
    return np.sqrt(kin_energy * (kin_energy + 2.0 * mass))


def magnetic_rigidity(mass: float, kin_energy: float) -> float:
    pc = get_pc(mass, kin_energy)
    brho = 1.00e+09 * (pc / speed_of_light)
    return brho


def space_charge_perveance(mass: float, kin_energy: float, line_density: float) -> float:
    gamma, beta = lorentz_factors(mass, kin_energy)
    perveance = (2.0 * classical_proton_radius * line_density) / (beta**2 * gamma**3)
    return perveance


def intensity_from_perveance(perveance: float, mass: float, kin_energy: float, length: float):
    gamma, beta = lorentz_factors(mass, kin_energy)
    intensity = (beta**2 * gamma**3 * perveance * bunch_length) / (2.0 * classical_proton_radius)
    return intensity