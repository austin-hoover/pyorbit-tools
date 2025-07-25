import os
import sys
import time
from typing import Any
from typing import Callable
from typing import Union

import numpy as np
import xarray as xr

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.spacecharge import Grid1D
from orbit.core.spacecharge import Grid2D
from orbit.core.spacecharge import Grid3D
from orbit.lattice import AccLattice
from orbit.lattice import AccNode


def get_grid_points(coords: list[np.ndarray]) -> np.ndarray:
    return np.vstack([C.ravel() for C in np.meshgrid(*coords, indexing="ij")]).T


def make_grid(
    shape: tuple[int, ...], limits: list[tuple[float, float]]
) -> Union[Grid1D, Grid2D, Grid3D]:

    ndim = len(shape)

    grid = None
    if ndim == 1:
        grid = Grid1D(shape[0] + 1, limits[0][0], limits[0][1])
    elif ndim == 2:
        grid = Grid2D(
            shape[0] + 1,
            shape[1] + 1,
            limits[0][0],
            limits[0][1],
            limits[1][0],
            limits[1][1],
        )
    elif ndim == 3:
        grid = Grid3D(
            shape[0] + 1,
            shape[1] + 1,
            shape[2] + 1,
        )
        grid.setGridX(limits[0][0], limits[0][1])
        grid.setGridY(limits[1][0], limits[1][1])
        grid.setGridZ(limits[2][0], limits[2][1])
    else:
        raise ValueError

    return grid


class Diagnostic:
    def __init__(self, output_dir: str = None, verbose: bool = True) -> None:
        self.mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        self.mpi_rank = orbit_mpi.MPI_Comm_rank(self.mpi_comm)
        self.output_dir = output_dir
        self.verbose = verbose

    def track(params_dict: dict) -> None:
        raise NotImplementedError

    def skip(self) -> bool:
        return False

    def update(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def __call__(self, params_dict: dict) -> None:
        if not self.skip():
            self.track(params_dict)
        self.update()


class BunchHistogram(Diagnostic):
    """MPI-compatible bunch histogram."""
    def __init__(
        self,
        axis: tuple[int, ...],
        shape: tuple[int, ...],
        limits: list[tuple[float, float]],
        method: str = None,
        transform: Callable = None,
        normalize: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        
        self.node = None
        self.index =0
        self.axis = axis
        self.ndim = len(axis)

        if self.ndim > 2:
            raise NotImplementedError(
                "BunchHistogram does not yet support 3D grids. See https://github.com/PyORBIT-Collaboration/PyORBIT3/issues/46 and https://github.com/PyORBIT-Collaboration/PyORBIT3/issues/47."
            )

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

        self.grid = make_grid(self.shape, self.limits)
        self.method = method
        self.transform = transform
        self.normalize = normalize

    def reset(self) -> None:
        self.grid.setZero()

    def sync_mpi(self) -> None:
        self.grid.synchronizeMPI(self.mpi_comm)

    def bin_bunch(self, bunch: Bunch) -> None:
        macrosize = bunch.macroSize()
        if macrosize == 0:
            bunch.macroSize(1.0)

        if self.method == "bilinear":
            self.grid.binBunchBilinear(bunch, *self.axis)
        else:
            self.grid.binBunch(bunch, *self.axis)

        bunch.macroSize(macrosize)

    def compute_histogram(self, bunch: Bunch) -> np.ndarray:
        self.bin_bunch(bunch)
        self.sync_mpi()

        values = np.zeros(self.points.shape[0])
        if self.method == "bilinear":
            for i, point in enumerate(self.points):
                values[i] = self.grid.getValueBilinear(*point)
        elif self.method == "nine-point":
            for i, point in enumerate(self.points):
                values[i] = self.grid.getValue(*point)
        else:
            for i, indices in enumerate(np.ndindex(*self.shape)):
                values[i] = self.grid.getValueOnGrid(*indices)
        values = np.reshape(values, self.shape)

        if self.normalize:
            values_sum = np.sum(values)
            if values_sum > 0.0:
                values /= values_sum
            values /= self.cell_volume
        return values

    def track(self, params_dict: dict) -> None:
        bunch_copy = Bunch()
        bunch = params_dict["bunch"]
        node = params_dict["node"]
        bunch.copyBunchTo(bunch_copy)
        self.node = node

        if self.transform is not None:
            bunch_copy = self.transform(bunch_copy)

        self.reset()
        self.values = self.compute_histogram(bunch_copy)

        if self.output_dir is not None:
            array = xr.DataArray(self.values, coords=self.coords, dims=self.dims)
            array.to_netcdf(path=self.get_filename())

        self.index += 1

    def get_filename(self) -> str:
        filename = "hist_" + "-".join([str(i) for i in self.axis])
        filename = "{}_{:04.0f}".format(filename, self.index)
        filename = "{}_{}".format(filename, self.node.getName())
        filename = "{}.nc".format(filename)
        filename = os.path.join(self.output_dir, filename)
        return filename


class BunchHistogram1D(BunchHistogram):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class BunchHistogram2D(BunchHistogram):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class BunchHistogram3D(BunchHistogram):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
