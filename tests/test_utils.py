import numpy as np
import pytest

from orbit.core import orbit_mpi
from orbit.core.orbit_utils import Matrix

import orbit_tools as ot


def test_get_lorentz_factors():
    ot.utils.get_lorentz_factors(mass=0.938, kin_energy=1.000)


def test_get_momentum():
    ot.utils.get_momentum(mass=0.938, kin_energy=1.000)



def test_get_magnetic_rigidity():
    ot.utils.get_magnetic_rigidity(mass=0.938, kin_energy=1.000)


def test_get_perveance():
    ot.utils.get_perveance(mass=0.938, kin_energy=1.000, line_density=1.0)


def test_get_intensity_from_perveance():
    ot.utils.get_intensity_from_perveance(perveance=1.0e-4, mass=0.938, kin_energy=1.000, length=1.0)


def test_orbit_matrix_to_numpy():
    matrix = Matrix(2, 2)
    matrix.set(0, 0, 1.0)
    matrix.set(1, 1, 1.0)
    matrix_np = ot.utils.orbit_matrix_to_numpy(matrix)
    assert matrix.get(0, 0) == matrix_np[0, 0]
