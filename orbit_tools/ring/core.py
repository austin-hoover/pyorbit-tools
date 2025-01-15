import os
import sys
import time
from typing import Callable
from typing import Iterable
from typing import Optional

import numpy as np
from tqdm import tqdm

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.core.orbit_utils import BunchExtremaCalculator
from orbit.lattice import AccLattice
from orbit.lattice import AccActionsContainer
from orbit.matrix_lattice import MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice

from .diag import RingDiagnostic
from ..bunch import get_bunch_cov
from ..bunch import transform_bunch_linear
from ..cov import normalization_matrix
from ..cov import normalization_matrix_from_eigvecs
from ..cov import normalize_eigvecs
from ..lattice import get_matrix_lattice
from ..utils import orbit_matrix_to_numpy


def get_transfer_matrix(lattice: AccLattice, mass: float, kin_energy: float, ndim: int = 6) -> np.ndarray:
    matrix_lattice = get_matrix_lattice(lattice, mass, kin_energy)
    M = matrix_lattice.oneTurnMatrix
    M = orbit_matrix_to_numpy(M)
    M = M[:ndim, :ndim]
    return M
    

def track_twiss(lattice: AccLattice, mass: float, kin_energy: float) -> dict[str, np.ndarray]:
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
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
    M = transfer_matrix
    if M is None:
        if lattice is None:
            raise ValueError("Need lattice")
        M = get_transfer_matrix(
            lattice, mass=bunch.mass(), kin_energy=bunch.getSyncParticle().kinEnergy
        )
    
    # Compute lattice normalization matrix V.
    M = np.copy(M)
    M = M[:4, :4]
    eigenvalues, eigenvectors = np.linalg.eig(M)
    eigenvectors = normalize_eigvecs(eigenvectors)
    V = normalization_matrix_from_eigvecs(eigenvectors)

    # Compute bunch normalization matrix W.
    S = get_bunch_cov(bunch)
    S = S[:4, :4]
    U = unit_symplectic_matrix(4)
    SU = np.matmul(S, U)
    eigenvalues, eigenvectors = np.linalg.eig(SU)
    eigenvectors = normalize_eigvecs(eigenvectors)
    W = normalization_matrix_from_eigvecs(eigenvectors)

    # Transform the bunch.
    T = np.matmul(V, np.linalg.inv(W))
    bunch = transform_bunch_linear(bunch, T, axis=(0, 1, 2, 3))
    return bunch


class Tracker:
    def __init__(
        self,
        lattice: AccLattice,
        bunch: Bunch,
        params_dict: dict,
        diagnostics: list[RingDiagnostic],
        progbar: bool = True,
        verbose: bool = 1,
    ) -> None:
        self.lattice = lattice
        self.bunch = bunch
        self.params_dict = params_dict
        self.diagnostics = diagnostics
        self.progbar = progbar
        self.verbose = verbose

    def get_turns_list(self, nturns: int) -> Iterable:
        turns = range(1, nturns + 1)
        if self.progbar:
            turns = tqdm(turns)
        return turns

    def track(self, nturns: int) -> None:

        action_container = None

        if self.verbose > 1:
            def action(params_dict):
                node = params_dict["node"]
                print(node.getName())

            action_container = AccActionsContainer("monitor")
            action_container.addAction(action, AccActionsContainer.EXIT)

        for turn in self.get_turns_list(nturns):
            self.lattice.trackBunch(self.bunch, self.params_dict, actionContainer=action_container)
            for diagnostic in self.diagnostics:
                diagnostic(self.params_dict)
