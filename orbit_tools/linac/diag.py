import os
import sys
import time
from typing import Callable

import numpy as np

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode

from ..bunch import get_z_to_phase_coefficient
from ..diag import Diagnostic


class LinacDiagnostic(Diagnostic):
    def __init__(self, stride: float = 0.0, position: float = 0.0, index: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stride = stride
        self.position = position
        self.position_old = position
        self.position_start = position

        self.index = index
        self.index_start = index

        self.params_dict = None
        self.node = None
        self.bunch = None

    def track(params_dict: dict) -> None:
        raise NotImplementedError

    def __call__(self, params_dict: dict, force_update: bool = False) -> None:
        if force_update or not self.should_skip():
            self.track(params_dict)
        self.update()

    def update(self, params_dict: dict) -> None:
        self.position_old = self.position
        self.read_params_dict()
        self.index += 1

    def read_params_dict(self, params_dict) -> None:
        self.params_dict = params_dict
        self.position = params_dict["path_length"] + self.position_start
        self.node = params_dict["node"]

    def should_skip(self, params_dict: dict) -> None:
        if self.index == 0:
            return False
        if self.position - self.position_old >= self.stride:
            return False
        return True

    def reset(self) -> None:
        self.position = self.position_old = self.position_start
        self.index = index_start
        self.node = None


class BunchWriter(LinacDiagnostic):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_filename(self) -> str:
        filename = "bunch"
        filename = "{}_{:04.0f}".format(filename, self.index)
        filename = "{}_{}".format(filename, self.node.getName())
        filename = "{}.dat".format(filename)
        filename = os.path.join(self.output_dir, filename)
        return filename

    def track(self, params_dict: dict) -> None:
        bunch = params_dict["bunch"]
        filename = self.get_filename(params_dict)
        if self.verbose and (self._mpi_rank == 0):
            print("Writing bunch to file {}".format(filename))
        bunch.dumpBunch(filename)


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

    def track(self, params_dict: dict) -> None:
        bunch_copy = Bunch()

        bunch = self.bunch
        bunch.copyBunchTo(bunch_copy)

        if self.transform is not None:
            bunch_copy = self.transform(bunch_copy)

        values = self.compute_histogram(bunch_copy)
        values = values / self.cell_volume
        values_sum = np.sum(values)
        if values_sum > 0.0:
            values = values / values_sum

        if self.output_dir is not None:
            array = xr.DataArray(values, coords=self.coords, dims=self.dims)
            array.to_netcdf(path=self.get_filename(params_dict))

        return values


class BunchHistogram2D(BunchHistogram):
    def __init__(self, method: str, **kwargs) -> None:
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
    def __init__(self, method: str, **kwargs) -> None:
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


class ScalarBunchMonitor(LinacDiagnostic):
    def __init__(self, rf_frequency: float = 402.5e06, **kwargs) -> None:
        kwargs.setdefault("stride", 0.1)
        LinacDiagnostic.__init__(self, **kwargs)

        self.rf_frequency = rf_frequency

        # State
        self.start_time = None
        self.time_ellapsed = 0.0

        # Store scalars only for latest update
        if self._mpi_rank == 0:
            keys = [
                "position",
                "node",
                "nparts",
                "gamma",
                "beta",
                "energy",
                "x_rms",
                "y_rms",
                "z_rms",
                "z_rms_deg",
                "z_to_phase_coeff",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "z_min",
                "z_max",
            ]
            for i in range(6):
                keys.append("mean_{}".format(i))
            for i in range(6):
                for j in range(i + 1):
                    keys.append("cov_{}-{}".format(j, i))

            self.history = {}
            for key in keys:
                self.history[key] = None

            if self.output_dir is not None:
                self.history_file = open(os.path.join(self.output_dir, "history.dat"), "w")
                line = ",".join(list(self.history))
                line = line[:-1] + "\n"
                self.history_file.write(line)

    def reset(self) -> None:
        self.position = self.position_offset
        self.index = 0
        self.start_time = None
        self.time_ellapsed = 0.0

    def measure_scalars(self) -> None:
        beta = self.bunch.getSyncParticle().beta()
        gamma = self.bunch.getSyncParticle().gamma()
        bunch_size_global = self.bunch.getSizeGlobal()

        if self._mpi_rank == 0:
            self.history["position"] = self.position
            self.history["node"] = self.node.getName()
            self.history["nparts"] = bunch_size_global
            self.history["gamma"] = gamma
            self.history["beta"] = beta
            self.history["energy"] = self.bunch.getSyncParticle().kinEnergy()

    def measure_stats(self, params_dict: dict) -> None:
        twiss_calc = BunchTwissAnalysis()
        order = 2
        twiss_calc.computeBunchMoments(self.bunch, order, 0, 0)
        for i in range(6):
            key = "mean_{}".format(i)
            value = twiss_calc.getAverage(i)
            if self._mpi_rank == 0:
                self.history[key] = value
        for i in range(6):
            for j in range(i + 1):
                key = "cov_{}-{}".format(j, i)
                value = twiss_calc.getCorrelation(j, i)
                if self._mpi_rank == 0:
                    self.history[key] = value

        if self._mpi_rank == 0:
            x_rms = np.sqrt(self.history["cov_0-0"])
            y_rms = np.sqrt(self.history["cov_2-2"])
            z_rms = np.sqrt(self.history["cov_4-4"])
            z_to_phase_coeff = get_z_to_phase_coefficient(self.bunch, self.rf_frequency)
            z_rms_deg = -z_to_phase_coeff * z_rms
            self.history["x_rms"] = x_rms
            self.history["y_rms"] = y_rms
            self.history["z_rms"] = z_rms
            self.history["z_rms_deg"] = z_rms_deg
            self.history["z_to_phase_coeff"] = z_to_phase_coeff

    def measure_extrema(self) -> None:
        extrema_calculator = BunchExtremaCalculator()
        (x_min, x_max, y_min, y_max, z_min, z_max) = extrema_calculator.extremaXYZ(self.bunch)
        if self._mpi_rank == 0:
            self.history["x_min"] = x_min
            self.history["x_max"] = x_max
            self.history["y_min"] = y_min
            self.history["y_max"] = y_max
            self.history["z_min"] = z_min
            self.history["z_max"] = z_max

    def track(self, params_dict) -> None:
        if self.start_time is None:
            self.start_time = time.time()
        self.time_ellapsed = time.time() - self.start_time

        self.read_params_dict(params_dict)
        self.measure_scalars()
        self.measure_stats()
        self.measure_extrema()

        if self.verbose and (self._mpi_rank == 0):
            message = "index={:05.0f} t={:0.2f} s={:0.3f} xrms={:0.2f} yrms={:0.2f} zrms={:0.2f} size={} node={}".format(
                self.index,
                self.time_ellapsed,
                self.position,
                self.history["x_rms"] * 1000.0,
                self.history["y_rms"] * 1000.0,
                self.history["z_rms"] * 1000.0,
                self.history["nparts"],
                params_dict["node"].getName(),
            )
            print(message)
            sys.stdout.flush()  # for MPI

        # Write new line to history file
        if (self._mpi_rank == 0) and (self.output_dir is not None):
            data = [self.history[key] for key in self.history]
            line = ""
            for i in range(len(data)):
                line += "{},".format(data[i])
            line = line[:-1] + "\n"
            self.history_file.write(line)

    def get_data(self) -> dict:
        if self.output_dir == None:
            for key in self.data.keys():
                self.data[key] = self.data[key][0: (self.index + 1)]
        return self.data
