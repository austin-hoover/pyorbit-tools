import os
import sys
import time

import numpy as np

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode


class Diagnostic:
    def __init__(self, output_dir: str, verbose: bool = True) -> None:
        self._mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        self._mpi_rank = orbit_mpi.MPI_Comm_rank(self._mpi_comm)
        self.output_dir = output_dir
        self.verbose = verbose

    def track(params_dict: dict) -> None:
        raise NotImplementedError

    def should_skip(self) -> None:
        raise NotImplementedError

    def update(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        return

    def __call__(self, params_dict: dict) -> None:
        if not self.should_skip():
            self.track(params_dict)
        self.update()
