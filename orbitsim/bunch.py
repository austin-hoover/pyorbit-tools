import math
import os
from typing import Any
from typing import Callable
from typing import List
from typing import Iterable
from typing import Tuple
from typing import Union

import numpy as np
from tqdm import tqdm

import orbit.bunch_generators
import orbit.core
import orbit.diagnostics
import orbit.lattice
import orbit.teapot
import orbit.utils
from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis


def get_intensity(current: float, frequency: float, charge: float) -> float:
    """Return the bunch intensity from beam current [A] and bunch frequency [Hz]."""
    return (current / frequency) / (abs(charge) * orbit.utils.consts.charge_electron)


def get_z_to_phase_coeff(bunch: Bunch, frequency: float) -> float:
    """Return coefficient to calculate rf phase [degrees] from z [m]."""
    wavelength = orbit.utils.consts.speed_of_light / frequency
    coeff = -360.0 / (bunch.getSyncParticle().beta() * wavelength)
    return coeff


def get_z_rms_deg(bunch: Bunch, frequency: float, z_rms: float) -> float:
    """Convert <zz> [m] to <phi phi> [deg]."""
    return -get_z_to_phase_coeff(bunch, frequency) * z_rms


def set_current(bunch: Bunch, current: float, frequency: float) -> Bunch:
    """Set bunch macroparticle size from current and bunch frequency.

    Assumes bunch charge is already set.
    """
    intensity = get_intensity(current=current, frequency=frequency, charge=bunch.charge())
    bunch_size_global = bunch.getSizeGlobal()
    if bunch_size_global > 0:
        macro_size = intensity / bunch_size_global
        bunch.macroSize(macro_size)
    return bunch


def get_coords(bunch: Bunch, size: int = None) -> np.ndarray:
    """Extract the phase space coordinates from the bunch.

    If using MPI, this function will return the particles on one MPI node.
    """
    if size is None:
        size = bunch.getSize()
    size = min(size, bunch.getSize())
    
    coords = np.zeros((size, 6))
    for i in range(size):
        coords[i, 0] = bunch.x(i)
        coords[i, 1] = bunch.xp(i)
        coords[i, 2] = bunch.y(i)
        coords[i, 3] = bunch.yp(i)
        coords[i, 4] = bunch.z(i)
        coords[i, 5] = bunch.dE(i)
    return coords


def set_coords(bunch: Bunch, coords: np.ndarray, verbose: bool = True) -> Bunch:
    """Assign phase space coordinate array to bunch.

    bunch : Bunch
        The bunch is resized if space needs to be made for the new particles.
    coords : ndarray, shape (k, 6)
        The phase space coordinates to add (columns: x, xp, y, yp, z, dE).
    verbose: bool
        Whether to use progress bar.

    Returns
    -------
    bunch : Bunch
        The modified bunch.
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)
    _mpi_dtype = orbit_mpi.mpi_datatype.MPI_DOUBLE
    _main_rank = 0

    _range = range(coords.shape[0])
    if verbose:
        _range = tqdm(_range)

    for i in _range:
        (x, xp, y, yp, z, dE) = coords[i, :]
        (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast(
            (x, xp, y, yp, z, dE),
            _mpi_dtype,
            _main_rank,
            _mpi_comm,
        )
        if i % _mpi_size == _mpi_rank:
            bunch.addParticle(x, xp, y, yp, z, dE)
    return bunch


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


def downsample(
    bunch: Bunch,
    new_size: int,
    method: str = "first",
    conserve_intensity: bool =True,
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
        print('Downsampling complete.')
    return bunch


def reverse(bunch: Bunch) -> Bunch:
    """Reverse the bunch propagation direction.

    Since the tail becomes the head of the bunch, the sign of z
    changes but the sign of dE does not change.
    """
    for i in range(bunch.getSize()):
        bunch.xp(i, -bunch.xp(i))
        bunch.yp(i, -bunch.yp(i))
        bunch.z(i, -bunch.z(i))
    return bunch


def transform(bunch: Bunch, transform: Callable, axis: tuple[int] = None) -> Bunch:
    if axis is None:
        axis = list(range(6))
    axis = list(axis)
    
    for i in range(bunch.getSize()):
        (x, xp) = (bunch.x(i), bunch.xp(i))
        (y, yp) = (bunch.y(i), bunch.yp(i))
        (z, de) = (bunch.z(i), bunch.dE(i))
        
        vector = np.array([x, xp, y, yp, z, de])
        vector[axis] = transform(vector[axis])
        
        (x, xp, y, yp, z, de) = vector
        bunch.x(i, x)
        bunch.y(i, y)
        bunch.z(i, z)
        bunch.xp(i, xp)
        bunch.yp(i, yp)
        bunch.dE(i, de)
    return bunch


def linear_transform(bunch: Bunch, matrix: np.ndarray, axis: tuple[int] = None) -> Bunch:
    assert matrix.shape[0] == matrix.shape[1]
    if axis is None:
        axis = list(range(matrix.shape[0]))
    return transform(bunch, lambda x: np.matmul(matrix, x), axis=axis)


def get_centroid(bunch: Bunch) -> np.array:
    twiss_analysis = BunchTwissAnalysis()
    twiss_analysis.analyzeBunch(bunch)
    return np.array([twiss_analysis.getAverage(i) for i in range(6)])


def shift_centroid(bunch: Bunch, delta: np.ndarray, verbose: bool = False) -> Bunch:
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

    if _mpi_rank == 0 and verbose:
        print("Shifting bunch centroid...")
        print(f" delta x = {delta[0]:.3e} [m]")
        print(f" delta y = {delta[2]:.3e} [m]")
        print(f" delta z = {delta[2]:.3e} [m]")
        print(f" delta xp = {delta[1]:.3e} [mrad]")
        print(f" delta yp = {delta[3]:.3e} [mrad]")
        print(f" delta dE = {delta[5]:.3e} [GeV]")
    for i in range(bunch.getSize()):
        bunch.x(i, bunch.x(i) + delta[0])
        bunch.y(i, bunch.y(i) + delta[2])
        bunch.z(i, bunch.z(i) + delta[4])
        bunch.xp(i, bunch.xp(i) + delta[1])
        bunch.yp(i, bunch.yp(i) + delta[3])
        bunch.dE(i, bunch.dE(i) + delta[5])
    if verbose:
        centroid = get_centroid(bunch)
        print("New centroid:")
        print("<x>  = {} [m]".format(centroid[0]))
        print("<y>  = {} [m]".format(centroid[2]))
        print("<z>  = {} [m]".format(centroid[4]))
        print("<xp> = {} [rad]".format(centroid[1]))
        print("<yp> = {} [rad]".format(centroid[3]))
        print("<de> = {} [GeV]".format(centroid[5]))
    return bunch


def set_centroid(bunch, centroid: np.ndarray = 0.0, verbose: bool = False) -> Bunch:
    if centroid is None:
        return bunch

    if np.ndim(centroid) == 0:
        centroid = 6 * [centroid]

    if all([x is None for x in centroid]):
        return bunch

    old_centroid = get_centroid(bunch)
    for i in range(6):
        if centroid[i] is None:
            centroid[i] = old_centroid[i]
    delta = np.subtract(centroid, old_centroid)
    return shift_centroid(bunch, delta=delta, verbose=verbose)


def get_mean(bunch: Bunch) -> np.array:
    return centroid(bunch)


def get_mean_and_covariance(bunch: Bunch, dispersion_flag: bool = False, emit_norm_flag: bool = False):
    order = 2
    bunch_twiss_analysis = BunchTwissAnalysis()
    bunch_twiss_analysis.computeBunchMoments(
        bunch,
        order,
        int(dispersion_flag),
        int(emit_norm_flag)
    )

    mean = np.zeros(6)
    for i in range(6):
        mean[i] = bunch_twiss_analysis.getAverage(i)

    cov = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1):
            value = bunch_twiss_analysis.getCorrelation(j, i)
            cov[i, j] = cov[j, i] = value

    return mean, cov


def get_covariance(bunch: Bunch, dispersion_flag: bool = False, emit_norm_flag: bool = False):
    mean, cov = get_mean_and_covariance(bunch, dispersion_flag, emit_norm_flag)
    return cov


def load(
    filename: str = None,
    format: str = "pyorbit",
    bunch: Bunch = None,
    verbose: bool = True,
):
    """
    Parameters
    ----------
    filename : str
        Path the file.
    format : str
        "pyorbit":
            The expected header format is:
        "parmteq":
            The expected header format is:
                Number of particles    =
                Beam current           =
                RF Frequency           =
                The input file particle coordinates were written in double precision.
                x(cm)             xpr(=dx/ds)       y(cm)             ypr(=dy/ds)       phi(radian)        W(MeV)
    bunch : Bunch
        Create a new bunch if None, otherwise modify this bunch.

    Returns
    -------
    Bunch
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

    if verbose:
        print("(rank {}) loading bunch from file {}".format(_mpi_rank, filename))

    if bunch is None:
        bunch = Bunch()

    if filename is None:
        print("No filename provided")
        return bunch

    if not os.path.isfile(filename):
        raise ValueError("File '{}' does not exist.".format(filename))

    if format == "pyorbit":
        bunch.readBunch(filename)
        return bunch
    elif format == "parmteq":
        # Read data.
        header = np.genfromtxt(filename, max_rows=3, usecols=[0, 1, 2, 3, 4], dtype=str)
        size = int(header[0, 4])
        current = np.float(header[1, 3])
        frequency = np.float(header[2, 3]) * 1e6  # MHz to Hz
        coords = np.loadtxt(filename, skiprows=5)

        # Unit conversion.
        coords[:, 0] *= 0.01  # [cm] -> [m]
        coords[:, 2] *= 0.01  # [cm] -> [m]
        coords[:, 4] = np.rad2deg(coords[:, 4]) / get_z_to_phase_coeff(bunch, frequency)  # [rad] -> [m]
        coords[:, 5] *= 0.001  # [MeV} -> [GeV]

        # Subtract synchronous particle energy.
        kin_energy = np.mean(coords[:, 5])
        coords[:, 5] -= kin_energy

        # Set coordinates.
        bunch.getSyncParticle().kinEnergy(kin_energy)
        set_coords(bunch, coords)
        set_current(bunch, current, frequency)
    else:
        raise KeyError(f"Unrecognized format '{format}'")

    if verbose:
        print(
            "(rank {}) bunch loaded (size={}, macrosize={})".format(
                _mpi_rank,
                bunch.getSize(),
                bunch.macroSize()
            )
        )
    return bunch


def generate(sample: Callable, size: int, bunch: Bunch = None, verbose: bool = True):
    """Generate bunch from distribution generator.

    Parameters
    ----------
    sample : callable
        Samples (x, xp, y, yp, z, dE) from the distribution.
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


def get_twiss_containers(bunch: Bunch) -> List[orbit.bunch_generators.TwissContainer]:
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
    twiss_x = orbit.bunch_generators.TwissContainer(alpha_x, beta_x, eps_x)
    twiss_y = orbit.bunch_generators.TwissContainer(alpha_y, beta_y, eps_y)
    twiss_z = orbit.bunch_generators.TwissContainer(alpha_z, beta_z, eps_z)
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
