import os
import sys
import time

import numpy as np
import pandas as pd

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.lattice import AccLattice
from orbit.core.orbit_utils import BunchExtremaCalculator

import orbitsim.bunch
import orbitsim.stats


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
    Sigma = orbitsim.bunch.get_covariance(bunch)    
    Sigma = Sigma[:4, :4]
    U = unit_symplectic_matrix(4)
    SU = np.matmul(Sigma, U)
    eigenvalues, eigenvectors = np.linalg.eig(SU)
    eigenvectors = normalize_eigenvectors(eigenvectors)
    W = normalization_matrix_from_eigenvectors(eigenvectors)

    # Transform the bunch.
    T = np.matmul(V, np.linalg.inv(W))
    bunch = orbitsim.bunch.linear_transform(bunch, T, axis=(0, 1, 2, 3))        
    return bunch


class Monitor:
    def __init__(self, path: str = None, verbose: bool = True) -> None:
        self.path = path
        self.verbose = verbose
        self.start_time = None
        self.iteration = 0

        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

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
                    
            self.history = dict()
            for key in keys:
                self.history[key] = None

            self.file = None
            if path is not None:
                self.file = open(path, "w")
                line = ",".join(list(self.history))
                line = line[:-1] + "\n"
                self.file.write(line)
            self.path = path
    
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
        (eps_x, eps_y) = orbitsim.stats.apparent_emittances(Sigma[:4, :4])
        (eps_1, eps_2) = orbitsim.stats.intrinsic_emittances(Sigma[:4, :4])   
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
                      
        # Print update message.
        if self.verbose and _mpi_rank == 0:
            runtime = time.time() - self.start_time
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

