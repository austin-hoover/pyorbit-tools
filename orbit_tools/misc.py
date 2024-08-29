from typing import Tuple

import numpy as np

from orbit.utils.consts import speed_of_light


CLASSICAL_PROTON_RADIUS = 1.53469e-18  # [m]



def get_lorentz_factors(mass: float, kin_energy: float) -> Tuple[float]:
    gamma = 1.0 + (kin_energy / mass)
    beta = np.sqrt(gamma**2 - 1.0) / gamma
    return (gamma, beta)


def get_momentum(mass: float, kin_energy: float) -> float:
    return np.sqrt(kin_energy * (kin_energy + 2.0 * mass))


def get_magnetic_rigidity(mass: float, kin_energy: float) -> float:
    brho = 1.00e+09 * get_momentum(mass=mass, kin_energy=kin_energy) / speed_of_light
    return brho


def get_perveance(mass: float, kin_energy: float, line_density: float) -> float:
    gamma, beta = get_lorentz_factors(mass, kin_energy)
    classical_proton_radius = CLASSICAL_PROTON_RADIUS
    perveance = (2.0 * classical_proton_radius * line_density) / (beta**2 * gamma**3)
    return perveance


def get_intensity_from_perveance(perveance: float, mass: float, kin_energy: float, length: float):
    gamma, beta = get_lorentz_factors(mass, kin_energy)
    classical_proton_radius = CLASSICAL_PROTON_RADIUS
    intensity = (beta**2 * gamma**3 * perveance * bunch_length) / (2.0 * classical_proton_radius)
    return intensity