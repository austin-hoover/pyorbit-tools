import math
import os
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional

import numpy as np
from tqdm import tqdm

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.bunch_generators import TwissContainer
from orbit.utils.consts import speed_of_light
from orbit.utils.consts import charge_electron

from .cov import normalization_matrix


def get_part_coords(bunch: Bunch, index: int) -> list[float]:
    x = bunch.x(index)
    y = bunch.y(index)
    z = bunch.z(index)
    xp = bunch.xp(index)
    yp = bunch.yp(index)
    de = bunch.dE(index)
    return [x, xp, y, yp, z, de]


def set_part_coords(bunch: Bunch, index: int, coords: list[float]) -> Bunch:
    (x, xp, y, yp, z, de) = coords
    bunch.x(index, x)
    bunch.y(index, y)
    bunch.z(index, z)
    bunch.xp(index, xp)
    bunch.yp(index, yp)
    bunch.dE(index, de)
    return bunch


def get_bunch_coords(bunch: Bunch, axis: tuple[int, ...] = None) -> np.ndarray:
    if axis is None:
        axis = tuple(range(6))

    X = np.zeros((bunch.getSize(), 6))
    for i in range(bunch.getSize()):
        X[i, 0] = bunch.x(i)
        X[i, 1] = bunch.xp(i)
        X[i, 2] = bunch.y(i)
        X[i, 3] = bunch.yp(i)
        X[i, 4] = bunch.z(i)
        X[i, 5] = bunch.dE(i)
    return X[:, axis]


def set_bunch_coords(
    bunch: Bunch, X: np.ndarray, axis: tuple[int, ...] = None
) -> Bunch:
    if axis is None:
        axis = tuple(range(6))

    if X.shape[0] != bunch.getSize():
        bunch = resize_bunch(bunch, X.shape[0])

    X_all = np.zeros((X.shape[0], 6))
    X_all[:, axis] = X

    coords = np.zeros(6)
    for i in range(X_all.shape[0]):
        set_part_coords(bunch, i, X_all[i, :])
    return bunch


def resize_bunch(bunch: Bunch, size: int) -> Bunch:
    X_old = get_bunch_coords(bunch)
    X_new = np.zeros((size, 6))

    if X_old.shape[0] <= X_new.shape[0]:
        X_new[: X_old.shape[0]] = X_old
    else:
        X_new = X_old[: X_new.shape[0]]

    bunch.deleteAllParticles()
    for i in range(X_new.shape[0]):
        bunch.addParticle(*X_new[i, :])
    return bunch


def reverse_bunch(bunch: Bunch) -> Bunch:
    size = bunch.getSize()
    for i in range(size):
        bunch.xp(i, -bunch.xp(i))
        bunch.yp(i, -bunch.yp(i))
        bunch.z(i, -bunch.z(i))
    return bunch


def get_bunch_centroid(bunch: Bunch) -> np.ndarray:
    calc = BunchTwissAnalysis()
    calc.analyzeBunch(bunch)
    return np.array([calc.getAverage(i) for i in range(6)])


def set_bunch_centroid(bunch: Bunch, centroid: np.ndarray) -> Bunch:
    offset = centroid - get_bunch_centroid(bunch)
    bunch = shift_bunch_centroid(bunch, offset)
    return bunch


def shift_bunch_centroid(bunch: Bunch, offset: np.ndarray) -> Bunch:
    for index in range(bunch.getSize()):
        coords = np.array(get_part_coords(bunch, index))
        set_part_coords(bunch, index, coords + offset)
    return bunch


def set_bunch_cov(
    bunch: Bunch, covariance_matrix: np.ndarray, block_diag: bool = True
) -> Bunch:
    X_old = get_bunch_coords(bunch)
    S_old = np.cov(X_old.T)

    # Assume block-diagonal covariance matrix
    V_old_inv = normalization_matrix(S_old, scale=True, block_diag=block_diag)
    V_old = np.linalg.inv(V_old_inv)

    S_new = np.copy(covariance_matrix)
    V_new_inv = normalization_matrix(S_new, scale=True, block_diag=block_diag)
    V_new = np.linalg.inv(V_new_inv)

    M = np.matmul(V_new, V_old_inv)
    X_new = np.matmul(X_old, M.T)

    bunch = set_bunch_coords(bunch, X_new)
    return bunch


def transform_bunch(
    bunch: Bunch, transform: Callable, axis: tuple[int, ...] = None
) -> Bunch:
    if axis is None:
        axis = tuple(range(6))

    X = get_bunch_coords(bunch)
    X[:, axis] = transform(X[:, axis])
    return set_bunch_coords(bunch, X)


def transform_bunch_linear(
    bunch: Bunch, matrix: np.ndarray, axis: tuple[int, ...] = None
) -> Bunch:
    return transform_bunch(bunch, lambda x: np.matmul(x, matrix.T), axis=axis)


def get_z_to_phase_coefficient(bunch: Bunch, frequency: float) -> float:
    velocity = bunch.getSyncParticle().beta() * speed_of_light
    wavelength = velocity / frequency
    coefficient = -360.0 / wavelength
    return coefficient


def current_to_intensity(current: float, frequency: float, charge: float) -> float:
    """Return the bunch intensity from beam current [A] and bunch frequency [Hz]."""
    return (current / frequency) / (abs(charge) * charge_electron)


def set_bunch_current(bunch: Bunch, current: float, frequency: float) -> Bunch:
    """Set bunch macroparticle size from current and bunch frequency.

    Assumes bunch charge is already set.
    """
    intensity = current_to_intensity(
        current=current, frequency=frequency, charge=bunch.charge()
    )
    bunch_size_global = bunch.getSizeGlobal()
    if bunch_size_global > 0:
        macro_size = intensity / bunch_size_global
        bunch.macroSize(macro_size)
    return bunch


def decorrelate_bunch_x_y_z(bunch: Bunch, verbose: bool = False) -> Bunch:
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


def decorrelate_bunch_xy_z(bunch: Bunch, verbose: bool = False) -> Bunch:
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


def downsample_bunch(
    bunch: Bunch,
    new_size: int,
    method: str = "first",
    conserve_intensity: bool = True,
    verbose: bool = True,
):
    """Delete a subset of the bunch particles.

    Parameters
    ----------
    new_size : int or float
        The number of particles to keep. (The new global bunch size.)
    method : str
        "first": Keep the first `size` particles in the bunch. (Or keep the
        first (n / mpi_size) particles on each processor.) This method
        assumes the particles were originally randomly generated.
    conserve_intensity : bool
        Whether to increase the macrosize after downsampling to conserve
        the bunch intensity.
    """
    if new_size is None:
        return bunch

    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

    macro_size = bunch.macroSize()
    bunch_size_global = bunch.getSizeGlobal()
    intensity = macro_size * bunch_size_global

    size_proc = int(new_size / _mpi_size)
    size_proc_old = bunch.getSize()
    if size_proc >= size_proc_old:
        return bunch

    if method == "first":
        if verbose:
            print(f"(rank {_mpi_rank}) size_old={size_proc_old}, size_new={size_proc}")
        for i in reversed(range(size_proc, size_proc_old)):
            bunch.deleteParticleFast(i)
        bunch.compress()
    else:
        raise ValueError("Invalid method")

    if conserve_intensity:
        bunch_size_global = bunch.getSizeGlobal()
        macro_size = intensity / bunch_size_global
        bunch.macroSize(macro_size)

    if verbose:
        print("Downsampling complete.")
    return bunch


def get_bunch_cov(
    bunch: Bunch, dispersion_flag: bool = False, emit_norm_flag: bool = False
) -> np.ndarray:
    order = 2
    twiss_calc = BunchTwissAnalysis()
    twiss_calc.computeBunchMoments(
        bunch,
        order,
        int(dispersion_flag),
        int(emit_norm_flag),
    )

    S = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1):
            S[i, j] = twiss_calc.getCorrelation(j, i)
            S[j, i] = S[i, j]
    return S


def generate_bunch(
    sample: Callable, size: int, bunch: Bunch = None, verbose: bool = True
) -> Bunch:
    """Generate bunch from particle sampler..

    Parameters
    ----------
    sample : callable
        Samples (x, xp, y, yp, z, dE) from the distribution: `(x, xp, y, yp, z, dE) = sample()`.
    size : int
        The number of particles to generate.
    bunch : Bunch
        The bunch is repopulated if provided. Otherwise a new bunch is created.
    verbose : bool
        Whether to show progress bar.

    Returns
    -------
    Bunch
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)
    _mpi_dtype = orbit_mpi.mpi_datatype.MPI_DOUBLE
    _mpi_main_rank = 0

    if bunch is None:
        bunch = Bunch()
    else:
        bunch.deleteAllParticles()

    _range = range(size)
    if verbose:
        _range = tqdm(_range)

    for i in _range:
        (x, xp, y, yp, z, dE) = sample()
        (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast(
            (x, xp, y, yp, z, dE),
            _mpi_dtype,
            _mpi_main_rank,
            _mpi_comm,
        )
        if i % _mpi_size == _mpi_rank:
            bunch.addParticle(x, xp, y, yp, z, dE)
    return bunch


def get_bunch_twiss_containers(bunch: Bunch) -> list[TwissContainer]:
    """Compute covariance matrix and return x/y/z TwissContainers from bunch.

    We return [twiss_x, twiss_y, twiss_z], where each entry is a TwissContainer
    containing [alpha, beta, emittance].

    This is helpful for generating rms-equivalent bunches.
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

    order = 2
    dispersion_flag = 0
    emit_norm_flag = 0

    twiss_analysis = BunchTwissAnalysis()
    twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)
    eps_x = twiss_analysis.getEffectiveEmittance(0)
    eps_y = twiss_analysis.getEffectiveEmittance(1)
    eps_z = twiss_analysis.getEffectiveEmittance(2)
    beta_x = twiss_analysis.getEffectiveBeta(0)
    beta_y = twiss_analysis.getEffectiveBeta(1)
    beta_z = twiss_analysis.getEffectiveBeta(2)
    alpha_x = twiss_analysis.getEffectiveAlpha(0)
    alpha_y = twiss_analysis.getEffectiveAlpha(1)
    alpha_z = twiss_analysis.getEffectiveAlpha(2)
    twiss_x = TwissContainer(alpha_x, beta_x, eps_x)
    twiss_y = TwissContainer(alpha_y, beta_y, eps_y)
    twiss_z = TwissContainer(alpha_z, beta_z, eps_z)
    return [twiss_x, twiss_y, twiss_z]


class WrappedBunch:
    def __init__(
        self,
        bunch: Bunch,
        current: float = None,
        frequency: float = None,
    ) -> None:
        self.bunch = bunch
        self.current = current
        self.frequency = frequency

    def update_macrosize(self) -> None:
        if self.current is not None and self.frequency is not None:
            set_current(self.bunch, self.current, self.frequency)

    def set_current(self, current: float) -> None:
        self.current = current
        self.update_macrosize()

    def set_frequency(self, frequency: float) -> None:
        self.frequency = frequency
        self.update_macrosize()
