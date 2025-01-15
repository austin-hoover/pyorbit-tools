import numpy as np
import pytest

import orbit_tools as ot


def test_get_lorentz_factors():
    ot.misc.get_lorentz_factors(mass=0.938, kin_energy=1.000)


def test_get_momentum():
    ot.misc.get_momentum(mass=0.938, kin_energy=1.000)



def test_get_magnetic_rigidity():
    ot.misc.get_magnetic_rigidity(mass=0.938, kin_energy=1.000)


def test_get_perveance():
    ot.misc.get_perveance(mass=0.938, kin_energy=1.000, line_density=1.0)


def test_get_intensity_from_perveance():
    ot.misc.get_intensity_from_perveance(perveance=1.0e-4, mass=0.938, kin_energy=1.000, length=1.0)