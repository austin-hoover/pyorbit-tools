import os
import sys
import time

import numpy as np
import pandas as pd

import orbit.lattice
from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.lattice import AccLattice
from orbit.core.orbit_utils import BunchExtremaCalculator

import orbit_tools.bunch
import orbit_tools.lattice
import orbit_tools.stats
import orbit_tools.utils


def get_transfer_matrix(lattice: AccLattice, mass: float, kin_energy: float, ndim: int = 6) -> np.ndarray:
    matrix_lattice = orbit_tools.lattice.get_matrix_lattice(lattice, mass, kin_energy)
    M = matrix_lattice.oneTurnMatrix
    M = orbit_tools.utils.orbit_matrix_to_numpy(M)
    M = M[:ndim, :ndim]
    return M
    

def track_twiss(lattice: AccLattice, mass: float, kin_energy: float) -> dict[str, np.ndarray]:
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    
    matrix_lattice = orbit.lattice.TEAPOT_MATRIX_Lattice(lattice, bunch)
    (pos_nu_x, pos_alpha_x, pos_beta_x) = matrix_lattice.getRingTwissDataX()
    (pos_nu_y, pos_alpha_y, pos_beta_y) = matrix_lattice.getRingTwissDataY()
    
    data = dict()
    data["pos"] = np.array(pos_nu_x)[:, 0]
    data["nu_x"] = np.array(pos_nu_x)[:, 1]
    data["nu_y"] = np.array(pos_nu_y)[:, 1]
    data["alpha_x"] = np.array(pos_alpha_x)[:, 1]
    data["alpha_y"] = np.array(pos_alpha_y)[:, 1]
    data["beta_x"] = np.array(pos_beta_x)[:, 1]
    data["beta_y"] = np.array(pos_beta_y)[:, 1]
    return data


def track_dispersion(lattice: AccLattice, mass: float, kin_energy: float) -> dict[str, np.ndarray]:
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    (pos_disp_x, pos_dispp_x) = matrix_lattice.getRingDispersionDataX()
    (pos_disp_y, pos_dispp_y) = matrix_lattice.getRingDispersionDataY()
    
    data = dict()
    data["s"] = np.array(pos_disp_x)[:, 0]
    data["disp_x"] = np.array(pos_disp_x)[:, 1]
    data["disp_y"] = np.array(pos_disp_y)[:, 1]
    data["dispp_x"] = np.array(pos_dispp_x)[:, 1]
    data["dispp_y"] = np.array(pos_dispp_y)[:, 1]
    return data


def unit_symplectic_matrix(ndim: int = 4) -> np.ndarray:
    U = np.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        U[i : i + 2, i : i + 2] = [[0.0, 1.0], [-1.0, 0.0]]
    return U


def normalize_eigenvectors(eigenvectors: np.ndarray) -> np.ndarray:
    ndim = eigenvectors.shape[0]
    U = unit_symplectic_matrix(ndim)
    for i in range(0, ndim, 2):
        v = eigenvectors[:, i]
        val = np.linalg.multi_dot([np.conj(v), U, v]).imag
        if val > 0.0:
            (eigenvectors[:, i], eigenvectors[:, i + 1]) = (
                eigenvectors[:, i + 1],
                eigenvectors[:, i],
            )
        eigenvectors[:, i : i + 2] *= np.sqrt(2.0 / np.abs(val))
    return eigenvectors


def normalization_matrix_from_eigenvectors(eigenvectors: np.ndarray) -> np.ndarray:
    V = np.zeros(eigenvectors.shape)
    for i in range(0, V.shape[1], 2):
        V[:, i] = eigenvectors[:, i].real
        V[:, i + 1] = (1.0j * eigenvectors[:, i]).real
    return V


def normalization_matrix_from_covariance_matrix(cov: np.ndarray) -> np.ndarray:
    Sigma = cov
    U = unit_symplectic_matrix(4)
    SU = np.matmul(Sigma, U)
    eigenvalues, eigenvectors = np.linalg.eig(SU)
    eigenvectors = normalize_eigenvectors(eigenvectors)
    W = normalization_matrix_from_eigenvectors(eigenvectors)
    return W


def match_bunch(bunch: Bunch, transfer_matrix: np.ndarray = None, lattice: AccLattice = None) -> Bunch:
    """Match the bunch covariance matrix to the ringn transfer matrix.
    
    X -> V inv(W) X, where V is the lattice normalization matrix and W is the bunch
    normalization matrix.
    
    W transforms the bunch such that Sigma = diag(eps_1, eps_1, eps_2, eps_2), where
    eps_j is the intrinsic emittance of mode j.
        
    Parameters
    ----------
    bunch: Bunch
        The bunch to normalize.
    transfer_matrix:
        A periodic symplectic transfer matrix.
    lattice : AccLattice
        A periodic lattice. Must be provided if `transfer_matrix=None`.
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

    # Get the lattice transfer matrix if not provided.
    if transfer_matrix is None:
        if lattice is None:
            raise ValueError("Need lattice")
        raise NotImplementedError
    
    # Compute lattice normalization matrix V.
    M = np.copy(transfer_matrix)
    M = M[:4, :4]
    eigenvalues, eigenvectors = np.linalg.eig(M)
    eigenvectors = normalize_eigenvectors(eigenvectors)
    V = normalization_matrix_from_eigenvectors(eigenvectors)

    # Compute bunch normalization matrix W.
    Sigma = orbit_tools.bunch.get_covariance(bunch)    
    Sigma = Sigma[:4, :4]
    U = unit_symplectic_matrix(4)
    SU = np.matmul(Sigma, U)
    eigenvalues, eigenvectors = np.linalg.eig(SU)
    eigenvectors = normalize_eigenvectors(eigenvectors)
    W = normalization_matrix_from_eigenvectors(eigenvectors)

    # Transform the bunch.
    T = np.matmul(V, np.linalg.inv(W))
    bunch = orbit_tools.bunch.linear_transform(bunch, T, axis=(0, 1, 2, 3))        
    return bunch


class Monitor:
    def __init__(self, output_dir: str = None, verbose: bool = True) -> None:
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        self.output_dir = output_dir
        self.verbose = verbose
        self.start_time = None
        self.iteration = 0

        if _mpi_rank == 0:
            keys = ["size", "gamma", "beta", "energy"]
            for dim in ["x", "y", "z"]:
                keys.append("{}_rms".format(dim))
            for dim in ["x", "y", "z"]:
                keys.append("{}_min".format(dim))
                keys.append("{}_max".format(dim))    
            for dim in ["x", "y", "z", "1", "2"]:
                keys.append("eps_{}".format(dim))
            for i in range(6):
                keys.append("mean_{}".format(i))
            for i in range(6):
                for j in range(i + 1):
                    keys.append("cov_{}-{}".format(j, i))
            keys.append("runtime")
                    
            self.history = dict()
            for key in keys:
                self.history[key] = None

            self.file = None
            if output_dir is not None:
                self.file = open(os.path.join(output_dir, "history.dat"), "w")
                line = ",".join(list(self.history))
                line = line[:-1] + "\n"
                self.file.write(line)
    
    def __call__(self, params_dict: dict) -> None:
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
                
        if self.start_time is None:
            self.start_time = time.time()
        
        # Get the bunch.
        bunch = params_dict["bunch"]
        beta = bunch.getSyncParticle().beta()
        gamma = bunch.getSyncParticle().gamma()
        size = bunch.getSizeGlobal()
        if _mpi_rank == 0:
            self.history["size"] = size
            self.history["gamma"] = gamma
            self.history["beta"] = beta
            self.history["energy"] = bunch.getSyncParticle().kinEnergy()

        # Measure centroid
        twiss_analysis = BunchTwissAnalysis()
        order = 2
        dispersion_flag = 0
        emit_norm_flag = 0
        twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)
        for i in range(6):
            key = "mean_{}".format(i)
            value = twiss_analysis.getAverage(i)
            if _mpi_rank == 0:
                self.history[key] = value
                
        # Measure covariance matrix
        Sigma = np.zeros((6, 6))
        for i in range(6):
            for j in range(i + 1):
                key = "cov_{}-{}".format(j, i)
                value = twiss_analysis.getCorrelation(j, i)
                if _mpi_rank == 0:
                    self.history[key] = value
                Sigma[j, i] = Sigma[i, j] = value
          
        # Measure rms emittances
        (eps_x, eps_y) = orbit_tools.stats.apparent_emittances(Sigma[:4, :4])
        (eps_1, eps_2) = orbit_tools.stats.intrinsic_emittances(Sigma[:4, :4])   
        if _mpi_rank == 0:
            self.history["eps_x"] = eps_x
            self.history["eps_y"] = eps_y
            self.history["eps_1"] = eps_1
            self.history["eps_2"] = eps_2
        
        # Measure rms sizes
        if _mpi_rank == 0:
            x_rms = np.sqrt(self.history["cov_0-0"])
            y_rms = np.sqrt(self.history["cov_2-2"])
            z_rms = np.sqrt(self.history["cov_4-4"])
            self.history["x_rms"] = x_rms
            self.history["y_rms"] = y_rms
            self.history["z_rms"] = z_rms
            
        # Measure maximum phase space amplitudes.
        extrema_calculator = BunchExtremaCalculator()
        (x_min, x_max, y_min, y_max, z_min, z_max) = extrema_calculator.extremaXYZ(bunch)
        if _mpi_rank == 0:
            self.history["x_min"] = x_min
            self.history["x_max"] = x_max
            self.history["y_min"] = y_min
            self.history["y_max"] = y_max
            self.history["z_min"] = z_min
            self.history["z_max"] = z_max

        if _mpi_rank == 0:
            runtime = time.time() - self.start_time
            self.history["runtime"] = runtime
                      
        # Print update message.
        if self.verbose and _mpi_rank == 0:
            message = f"turn={self.iteration:05.0f} t={runtime:0.3f} size={size:05.0f}"
            message = "{} xrms={:0.2e} yrms={:0.2e} epsx={:0.2e} epsy={:0.2e} eps1={:0.2e} eps2={:0.2e} ".format(
                message,
                1.00e+03 * x_rms,
                1.00e+03 * y_rms,
                1.00e+06 * eps_x,
                1.00e+06 * eps_y,
                1.00e+06 * eps_1,
                1.00e+06 * eps_2,
            )
            print(message)

        # Add one line to the history file.
        if _mpi_rank == 0 and self.file is not None:
            data = [self.history[key] for key in self.history]
            line = ""
            for i in range(len(data)):
                line += "{},".format(data[i])
            line = line[:-1] + "\n"
            self.file.write(line)

        self.iteration += 1    
