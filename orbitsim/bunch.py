import math
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import orbit.bunch_generators
import orbit.lattice
import orbit.teapot
import orbit.utils
from orbit.core.bunch import Bunch
from orbit.core.orbit_mpi import mpi_comm
from orbit.core.orbit_mpi import mpi_datatype
from orbit.core.orbit_mpi import mpi_op
from orbit.core.orbit_utils import Matrix


def get_intensity(current: float, frequency: float, charge: float) -> float:
    """Return the bunch intensity from beam current [A] and bunch frequency [Hz]."""
    return (current / frequency) / (abs(charge) * orbit.utils.consts.charge_electron)


def get_z_to_phase_coeff(bunch: Bunch, frequency: float) -> float:
    """Return coefficient to calculate rf phase [degrees] from z [m]."""
    wavelength = orbit.utils.consts.speed_of_light / frequency
    coeff = -360.0 / (bunch.getSyncParticle().beta() * wavelength)
    return coeff


def get_z_rms_deg(bunch: Bunch, frequency: float, z_rms: float) -> float:
    """Convert <zz> [m] to <\phi\phi> [deg]."""
    return -get_z_to_phase_coeff(bunch, frequency) * z_rms


def set_current(bunch: Bunch, current: float, frequency: float) -> Bunch:
    """Set macroparticle size from beam current [A] and bunch frequency [Hz]."""
    intensity = get_intensity(current=current, frequency=frequency, charge=bunch.charge())
    size = bunch.getSizeGlobal()
    macro_size = intensity / size
    bunch.macroSize(macro_size)
    return bunch


def get_coords(bunch: Bunch) -> np.ndarray:
    """Extract the phase space coordinates from the bunch.

    If using MPI, this function will return the particles on one MPI node.
    """
    X = np.zeros((bunch.getSize(), 6))
    for i in range(bunch.getSize()):
        X[i, 0] = bunch.x(i)
        X[i, 1] = bunch.xp(i)
        X[i, 2] = bunch.y(i)
        X[i, 3] = bunch.yp(i)
        X[i, 4] = bunch.z(i)
        X[i, 5] = bunch.dE(i)
    return X


def decorrelate_x_y_z(bunch: Bunch, verbose: bool = False) -> Bunch:
    """Decorrelate x-y-z by randomly permuting particle indices.
    
    Does not work with MPI.
    """
    size = bunch.getSizeGlobal()
    idx_x = np.random.permutation(np.arange(size))
    idx_y = np.random.permutation(np.arange(size))
    idx_z = np.random.permutation(np.arange(size))

    if verbose:
        print("Building decorrelated bunch")
    
    bunch_temp = Bunch()
    bunch.copyEmptyBunchTo(bunch_temp)

    indices = zip(idx_x, idx_y, idx_z)
    if verbose:
        indices = tqdm(indices)
    
    for i, j, k in indices:
        bunch_temp.addParticle(
            bunch.x(i), bunch.xp(i), bunch.y(j), bunch.yp(j), bunch.z(k), bunch.dE(k)
        )
        
    bunch_temp.copyBunchTo(bunch)
    bunch_temp.deleteAllParticles()
    return bunch


def decorrelate_xy_z(bunch: Bunch, verbose: bool = False) -> Bunch:
    """Decorrelate xy-z by randomly permuting particle indices.
    
    Does not work with MPI.
    """
    size = bunch.getSizeGlobal()
    idx_xy = np.random.permutation(np.arange(size))
    idx_z = np.random.permutation(np.arange(size))
    
    if verbose:
        print("Building decorrelated bunch")

    bunch_temp = Bunch()
    bunch.copyEmptyBunchTo(bunch_temp)

    indices = zip(idx_xy, idx_z)
    if verbose:
        indices = tqdm(indices)

    for i, j in indices:
        bunch_temp.addParticle(
            bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i), bunch.z(j), bunch.dE(j)
        )
        
    bunch_temp.copyBunchTo(bunch)
    bunch_temp.deleteAllParticles()
    return bunch



