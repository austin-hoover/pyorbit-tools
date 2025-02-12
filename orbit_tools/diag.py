import os
import sys
import time
from typing import Any
from typing import Callable

import numpy as np

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.spacecharge import Grid1D
from orbit.core.spacecharge import Grid2D
from orbit.core.spacecharge import Grid3D
from orbit.lattice import AccLattice
from orbit.lattice import AccNode


def get_grid_points(coords: list[np.ndarray]) -> np.ndarray:
    return np.vstack([C.ravel() for C in np.meshgrid(*coords, indexing="ij")]).T


class Diagnostic:
    def __init__(self, output_dir: str = None, verbose: bool = True) -> None:
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


class BunchHistogram(Diagnostic):
    def __init__(
        self,
        axis: tuple[int, ...],
        shape: tuple[int, ...],
        limits: list[tuple[float, float]],
        transform: Callable = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.axis = axis
        self.ndim = len(axis)

        self.dims = ["x", "xp", "y", "yp", "z", "dE"]
        self.dims = [self.dims[i] for i in self.axis]

        self.shape = shape
        self.limits = limits
        self.edges = [
            np.linspace(self.limits[i][0], self.limits[i][1], self.shape[i] + 1)
            for i in range(self.ndim)
        ]
        self.coords = [0.5 * (e[:-1] + e[1:]) for e in self.edges]
        self.values = np.zeros(shape)
        
        self.points = get_grid_points(self.coords)
        self.cell_volume = np.prod([e[1] - e[0] for e in self.edges])

        self.transform = transform

    def get_filename(self) -> str:
        filename = "hist_" + "-".join([str(i) for i in self.axis])
        filename = "{}_{:04.0f}".format(filename, self.index)
        filename = "{}_{}".format(filename, self.node.getName())
        filename = "{}.nc".format(filename)
        filename = os.path.join(self.output_dir, filename)
        return filename

    def compute_histogram(self, bunch: Bunch) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, params_dict: dict) -> np.ndarray:
        bunch_copy = Bunch()

        bunch = params_dict["bunch"]
        bunch.copyBunchTo(bunch_copy)

        if self.transform is not None:
            bunch_copy = self.transform(bunch_copy)

        self.values = self.compute_histogram(bunch_copy)
        values_sum = np.sum(self.values)
        if values_sum > 0.0:
            self.values = self.values / values_sum
        self.values = self.values / self.cell_volume

        if self.output_dir is not None:
            array = xr.DataArray(values, coords=self.coords, dims=self.dims)
            array.to_netcdf(path=self.get_filename(params_dict))

        return self.values

    def track(self, bunch: Bunch) -> np.ndarray:
        params_dict = {"bunch": bunch}
        return self.__call__(params_dict)


class BunchHistogram2D(BunchHistogram):
    def __init__(self, method: str = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self._grid = Grid2D(
            self.shape[0] + 1,
            self.shape[1] + 1,
            self.limits[0][0],
            self.limits[0][1],
            self.limits[1][0],
            self.limits[1][1],
        )
        self.method = method

    def reset(self) -> None:
        self._grid.setZero()

    def compute_histogram(self, bunch: Bunch) -> np.ndarray:
        # Bin coordinates on grid
        if self.method == "bilinear":
            self._grid.binBunchBilinear(bunch, self.axis[0], self.axis[1])
        else:
            self._grid.binBunch(bunch, self.axis[0], self.axis[1])

        # Synchronize MPI
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        self._grid.synchronizeMPI(comm)

        # Extract grid values as numpy array
        values = np.zeros(self.points.shape[0])
        if self.method == "bilinear":
            for i, point in enumerate(self.points):
                values[i] = self._grid.getValueBilinear(*point)
        elif self.method == "nine-point":
            for i, point in enumerate(self.points):
                values[i] = self._grid.getValue(*point)
        else:
            index = 0
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    values[index] = self._grid.getValueOnGrid(i, j)
                    index += 1

        values = np.reshape(values, self.shape)
        return values


class BunchHistogram3D(BunchHistogram):
    def __init__(self, method: str = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self._grid = Grid3D(
            self.shape[0] + 1,
            self.shape[1] + 1,
            self.shape[2] + 1,
            self.limits[0][0],
            self.limits[0][1],
            self.limits[1][0],
            self.limits[1][1],
            self.limits[2][0],
            self.limits[2][1],
        )
        self.method = method

    def reset(self) -> None:
        self._grid.setZero()

    def compute_histogram(self, bunch: Bunch) -> np.ndarray:
        # Bin coordinates on grid
        if self.method == "bilinear":
            self._grid.binBunchBilinear(bunch, self.axis[0], self.axis[1], self.axis[2])
        else:
            self._grid.binBunch(bunch, self.axis[0], self.axis[1], self.axis[2])

        # Synchronize MPI
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        self._grid.synchronizeMPI(comm)

        # Extract grid values as numpy array
        values = np.zeros(self.points.shape[0])
        if self.method == "bilinear":
            for i, point in enumerate(self.points):
                values[i] = self._grid.getValueBilinear(*point)
        elif self.method == "nine-point":
            for i, point in enumerate(self.points):
                values[i] = self._grid.getValue(*point)
        else:
            index = 0
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    for k in range(self.shape[2]):
                        values[index] = self._grid.getValueOnGrid(i, j, k)
                        index += 1

        values = np.reshape(values, self.shape)
        return values
