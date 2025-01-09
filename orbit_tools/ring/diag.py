import os
import sys
import time

import numpy as np

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.core.orbit_utils import BunchExtremaCalculator
from orbit.lattice import AccLattice
from orbit.lattice import AccNode

from ..diag import Diagnostic
from ..cov import projected_emittances
from ..cov import intrinsic_emittances


class RingDiagnostic(Diagnostic):
    def __init__(self, freq: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.freq = freq
        self.turn = 0

    def track(params_dict: dict) -> None:
        raise NotImplementedError

    def should_skip(self) -> None:
        return self.turn % self.freq != 0

    def update(self) -> None:
        self.turn += 1

    def reset(self) -> None:
        self.turn = 0


class BunchWriter(RingDiagnostic):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_filename(self) -> None:
        filename = f"bunch_{self.turn:05.0f}.dat"
        filename = os.path.join(self.output_dir, filename)
        return filename

    def track(self, params_dict: dict) -> None:
        filename = self.get_filename()
        if self.verbose:
            if self._mpi_rank == 0:
                print(f"Writing {filename}")
                sys.stdout.flush()

        bunch = params_dict["bunch"]
        bunch.dumpBunch(filename)


class BunchMonitor(RingDiagnostic):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.start_time = None

        if self._mpi_rank == 0:
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
            if self.output_dir is not None:
                self.file = open(os.path.join(self.output_dir, "history.dat"), "w")
                line = ",".join(list(self.history))
                line = line[:-1] + "\n"
                self.file.write(line)

    def track(self, params_dict: dict) -> None:
        if self.start_time is None:
            self.start_time = time.time()

        bunch = params_dict["bunch"]
        beta = bunch.getSyncParticle().beta()
        gamma = bunch.getSyncParticle().gamma()
        size = bunch.getSizeGlobal()

        if self._mpi_rank == 0:
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
            if self._mpi_rank == 0:
                self.history[key] = value

        # Measure covariance matrix
        S = np.zeros((6, 6))
        for i in range(6):
            for j in range(i + 1):
                key = "cov_{}-{}".format(j, i)
                value = twiss_analysis.getCorrelation(j, i)
                if self._mpi_rank == 0:
                    self.history[key] = value
                S[j, i] = S[i, j] = value

        # Measure rms emittances
        (eps_x, eps_y) = projected_emittances(S[:4, :4])
        (eps_1, eps_2) = intrinsic_emittances(S[:4, :4])
        if self._mpi_rank == 0:
            self.history["eps_x"] = eps_x
            self.history["eps_y"] = eps_y
            self.history["eps_1"] = eps_1
            self.history["eps_2"] = eps_2

        # Compute rms sizes
        if self._mpi_rank == 0:
            x_rms = np.sqrt(self.history["cov_0-0"])
            y_rms = np.sqrt(self.history["cov_2-2"])
            z_rms = np.sqrt(self.history["cov_4-4"])
            self.history["x_rms"] = x_rms
            self.history["y_rms"] = y_rms
            self.history["z_rms"] = z_rms

        # Measure maximum phase space amplitudes
        extrema_calculator = BunchExtremaCalculator()
        (x_min, x_max, y_min, y_max, z_min, z_max) = extrema_calculator.extremaXYZ(bunch)
        if self._mpi_rank == 0:
            self.history["x_min"] = x_min
            self.history["x_max"] = x_max
            self.history["y_min"] = y_min
            self.history["y_max"] = y_max
            self.history["z_min"] = z_min
            self.history["z_max"] = z_max

        if self._mpi_rank == 0:
            runtime = time.time() - self.start_time
            self.history["runtime"] = runtime

        # Print update message
        if self.verbose and (self._mpi_rank == 0):
            message = f"turn={self.turn:05.0f} t={runtime:0.3f} size={size:05.0f}"
            message = "{} xrms={:0.2f}".format(message, x_rms * 1000.0)
            message = "{} yrms={:0.2f}".format(message, y_rms * 1000.0)
            message = "{} zrms={:0.2f}".format(message, z_rms * 1000.0)
            print(message)
            sys.stdout.flush()

        # Add line to history file
        if (self.file is not None) and (self._mpi_rank == 0):
            data = [self.history[key] for key in self.history]
            line = ""
            for i in range(len(data)):
                line += "{},".format(data[i])
            line = line[:-1] + "\n"
            self.file.write(line)