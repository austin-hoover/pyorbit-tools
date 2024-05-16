import os
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd

import orbit.core
import orbit.py_linac
import orbit.teapot
from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.core.orbit_utils import BunchExtremaCalculator
# from orbit.core.orbit_utils.bunch_utils_functions import copyCoordsToInitCoordsAttr
from orbit.bunch_utils import ParticleIdNumber
from orbit.lattice import AccActionsContainer
from orbit.py_linac.lattice import BaseLinacNode

from .bunch import get_z_to_phase_coeff


# _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
# _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
# _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)


class BunchWriter:
    """Writes bunch coordinates to file.

    Filename is {output_dir}/{name}_{index}.{ext}. Example: "./output/bunch_0001.dat".
    """
    def __init__(
        self, 
        output_dir: str = ".", 
        output_name: str = "bunch",
        output_ext: str = "dat",
        output_index_format: str = "04.0f",

        index: int = 0,
        position: float = 0.0,
        node_name: str = None,
        
        verbose: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.index = index
        self.node_name = node_name

        self.output_name = output_name
        self.output_index_format = output_index_format
        self.output_ext = output_ext

        self.verbose = verbose
        
    def __call__(self, bunch: Bunch, tag: str = None) -> None:
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        filename = self.output_name
        if self.index is not None:
            fstr = "{}_{:" + self.output_index_format + "}_{}"
            filename = fstr.format(filename, self.index)
        if tag is not None:
            filename = "{}_{}".format(filename, tag)
        filename = "{}.dat".format(filename)
        filename = os.path.join(self.outdir, filename)
        
        if _mpi_rank == 0 and self.verbose:
            print("Writing bunch to file {}".format(filename))
            
        bunch.dumpBunch(filename)
        
        if self.index is not None:
            self.index += 1
        if node_name is not None:
            self.node_name = node_name


class BunchWriterTEAPOT:
    def __init__(self):
        raise NotImplementedError
        
        
class BunchWriterLinacNode(BaseLinacNode):
    def __init__(
        self, 
        name: str = "bunch_writer_node", 
        node_name: str = None,
        active: bool = True,
        writer_kws: dict = None,
    ) -> None:
        BaseLinacNode.__init__(self, name)

        if writer_kws is None:
            writer_kws = dict()
            
        self.writer = BunchWriter(**writer_kws)
        self.node_name = node_name
        self.active = active

    def track(self, params_dict: dict) -> None:        
        if self.active and params_dict.has_key("bunch"):
            bunch = params_dict["bunch"]
            self.write(bunch, node_name=self.node_name)
            
    def trackDesign(self, params_dict: None) -> None:
        pass
            

class BunchMonitor:
    def __init__(
        self,
        plot: Callable,
        write: Callable,
        stride: dict,
        track_rms: bool = True,
        dispersion_flag: bool = False,
        emit_norm_flag: bool = False,
        position_offset: float = 0.0,
        verbose: bool = True,
        rf_frequency: float = 402.5e+06,
        history_filename: str = None,
    ) -> None:
        """Constructor.
        
        Parameters
        ----------
        plot : Callable
            Calls plotting functions.
        write : Callable
            Writes the bunch coordinates to a file.
        stride : dict
            Distance [m] between updates. Keys:
                "update": proceed with all updates (print, etc.)
                "write": call `self.write(bunch)`
                "plot": call `self.plot(bunch)`
        track_rms : bool
            Whether include RMS bunch parameters in history arrays.
        emit_norm_flag, dispersion_flag : bool
            Used by `BunchTwissAnalysis` class.
        position_offset : float
            The initial position in the lattice [m].
        verbose : bool
            Whether to print an update statement on each action.
        history_filename : str or None
            Scalar history output file.
        """
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        # Settings
        self.rf_frequency = rf_frequency
        self.dispersion_flag = int(dispersion_flag)
        self.emit_norm_flag = int(emit_norm_flag)
        self.track_rms = track_rms
        self.verbose = verbose

        self.stride = stride
        if self.stride is None:
            self.stride = dict()
        self.stride.setdefault("update", 0.1)
        self.stride.setdefault("write", np.inf)
        self.stride.setdefault("plot", np.inf)
        
        self.write = write
        self.plot = plot

        # State
        self.position = self.position_offset = position_offset
        self.last_plot_position = self.position
        self.last_write_position = self.position
        self.index = 0
        self.start_time = None

        # We store scalars only for the last update.
        if _mpi_rank == 0:
            keys = [
                "position",
                "node",
                "n_parts",
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

            self.history_filename = history_filename
            self.history_file = None
            if self.history_filename is not None:
                self.history_file = open(self.history_filename, "w")
                line = ",".join(list(self.history))
                line = line[:-1] + "\n"
                self.history_file.write(line)
            
    def __call__(self, params_dict, force_update=False):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        
        # Update position; decide whether to proceed.
        position = params_dict["path_length"] + self.position_offset
        if not force_update:
            if self.index > 0:
                if (position - self.position) < self.stride["update"]:
                    return
        self.position = position
        
        # Update clock.
        if self.start_time is None:
            self.start_time = time.clock()
        time_ellapsed = time.clock() - self.start_time
        
        # Get bunch and node.
        bunch = params_dict["bunch"]
        node = params_dict["node"]

        # Record scalar values (position, energy, etc.)
        beta = bunch.getSyncParticle().beta()
        gamma = bunch.getSyncParticle().gamma()
        bunch_size_global = bunch.getSizeGlobal()
        if _mpi_rank == 0:
            self.history["position"] = position
            self.history["node"] = node.getName()
            self.history["n_parts"] = bunch_size_global
            self.history["gamma"] = gamma
            self.history["beta"] = beta
            self.history["energy"] = bunch.getSyncParticle().kinEnergy()

        # Measure covariance matrix.
        if self.track_rms:
            twiss_analysis = BunchTwissAnalysis()
            order = 2
            twiss_analysis.computeBunchMoments(bunch, order, self.dispersion_flag, self.emit_norm_flag)
            for i in range(6):
                key = "mean_{}".format(i)
                value = twiss_analysis.getAverage(i)
                if _mpi_rank == 0:
                    self.history[key] = value
            for i in range(6):
                for j in range(i + 1):
                    key = "cov_{}-{}".format(j, i)
                    value = twiss_analysis.getCorrelation(j, i)
                    if _mpi_rank == 0:
                        self.history[key] = value
                                                   
        if _mpi_rank == 0 and self.track_rms:
            x_rms = np.sqrt(self.history["cov_0-0"])
            y_rms = np.sqrt(self.history["cov_2-2"])
            z_rms = np.sqrt(self.history["cov_4-4"])
            z_to_phase_coeff = get_z_to_phase_coeff(bunch, self.rf_frequency)
            z_rms_deg = -z_to_phase_coeff * z_rms
            self.history["x_rms"] = x_rms
            self.history["y_rms"] = y_rms
            self.history["z_rms"] = z_rms
            self.history["z_rms_deg"] = z_rms_deg
            self.history["z_to_phase_coeff"] = z_to_phase_coeff
            
        # Measure max phase space coordinates.
        extrema_calculator = BunchExtremaCalculator()
        (x_min, x_max, y_min, y_max, z_min, z_max) = extrema_calculator.extremaXYZ(bunch)
        if _mpi_rank == 0:
            self.history["x_min"] = x_min
            self.history["x_max"] = x_max
            self.history["y_min"] = y_min
            self.history["y_max"] = y_max
            self.history["z_min"] = z_min
            self.history["z_max"] = z_max
                                
        # Print update statement.
        if self.verbose and _mpi_rank == 0:
            if self.track_rms:
                fstr = "{:>5} | {:>10.2f} | {:>10.5f} | {:>8.4f} | {:>9.3f} | {:>9.3f} | {:>10.3f} | {:<9.0f} | {} "
                if self.index == 0:
                    print(
                        "{:<5} | {:<10} | {:<10} | {:<8} | {:<5} | {:<9} | {:<10} | {:<9} | {}"
                        .format("step", "time [s]", "s [m]", "T [MeV]", "xrms [mm]", "yrms [mm]", "zrms [deg]", "nparts", "node")
                    )
                    print(115 * "-")
                print(
                    fstr.format(
                        self.index,
                        time_ellapsed,  # [s]
                        position,  # [m]
                        1000.0 * bunch.getSyncParticle().kinEnergy(),
                        1000.0 * x_rms,
                        1000.0 * y_rms,
                        z_rms_deg,
                        bunch_size_global,
                        node.getName(),
                    )
                )
            else:
                fstr = "{:>5} | {:>10.2f} | {:>10.3f} | {:>8.4f} | {:<9.0f} | {} "
                if self.index == 0:
                    print(
                        "{:<5} | {:<10} | {:<10} | {:<10} | {:<9} | {}"
                        .format("step", "time [s]", "s [m]", "T [MeV]", "nparts", "node")
                    )
                    print(80 * "-")
                print(
                    fstr.format(
                        self.index,
                        time_ellapsed,  # [s]
                        position,  # [m]
                        1000.0 * bunch.getSyncParticle().kinEnergy(),
                        bunch_size_global,
                        node.getName(),
                    )
                )
        self.index += 1
                                                
        # Write phase space coordinates to file.
        if self.write is not None and self.stride["write"] is not None:
            if (position - last_write_position) >= self.stride["write"]:
                self.write(bunch, node_name=node.getName(), position=position)
                last_write_position = position

        # Call plotting routines.
        if self.plotter is not None and self.stride["plot"] is not None and _mpi_rank == 0:
            if (position - last_plot_position) >= self.stride["plot"]:
                info = dict()
                for key in self.history:
                    if self.history[key]:
                        info[key] = self.history[key]
                info["node"] = node.getName()
                info["step"] = self.index
                info["position"] = position
                info["gamma"] = gamma
                info["beta"] = beta
                self.plot(bunch, info=info, verbose=self.verbose)
                last_plot_position = position
                
        # Write new line to history file.
        if _mpi_rank == 0 and self.file is not None:
            data = [self.history[key] for key in self.history]
            line = ""
            for i in range(len(data)):
                line += "{},".format(data[i])
            line = line[:-1] + "\n"
            self.file.write(line)
    
    
# def get_node_info(node_name_or_position, lattice):
#     """Return node, node index, start and stop position for node name or center position.
    
#     Helper method for `track_bunch` and `track_bunch_reverse`.
    
#     Parameters
#     ----------
#     argument : node name or position.
#     lattice : LinacAccLattice
    
#     Returns
#     -------
#     dict
#         "node": the node instance
#         "index": the node index in the lattice
#         "s0": the node start position
#         "
#     """
#     if type(node_name_or_position) is str:
#         name = node_name_or_position
#         node = lattice.getNodeForName(name)
#         index = lattice.getNodeIndex(node)
#         s0 = node.getPosition() - 0.5 * node.getLength()
#         s1 = node.getPosition() + 0.5 * node.getLength()
#     else:
#         position = node_name_or_position
#         node, index, s0, s1 = lattice.getNodeForPosition(position)
#     return {
#         "node": node,
#         "index": index,
#         "s0": s0,
#         "s1": s1,
#     }
    
    
# def check_sync_part_time(bunch, lattice, start=0.0, set_design=False, verbose=True):
#     """Check if the synchronous particle time is set to the design value at start."""
#     _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
#     _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
#     _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

#     start = get_node_info(start, lattice)
#     sync_time = bunch.getSyncParticle().time()
#     sync_time_design = 0.0
#     if start["index"] > 0:
#         design_bunch = lattice.trackDesignBunch(bunch, index_start=0, index_stop=start["index"])
#         sync_time_design = design_bunch.getSyncParticle().time()
#     if _mpi_rank == 0 and verbose:
#         print("Start index = {}:".format(start["index"]))
#         print("    Synchronous particle time (actual) = {}".format(sync_time))
#         print("    Synchronous particle time (design) = {}".format(sync_time_design))
#     if set_design and abs(sync_time - sync_time_design) > 1.0e-30:
#         if _mpi_rank == 0 and verbose:
#             print("    Setting to design value.")
#         bunch.getSyncParticle().time(sync_time_design)
#         if _mpi_rank == 0 and verbose:
#             print("bunch.getSyncParticle().time() = {}".format(bunch.getSyncParticle().time()))
            

# def track(bunch, lattice, monitor=None, start=0.0, stop=None, verbose=True):
#     """Track bunch from start to stop."""
#     _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
#     _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

#     # Get start/stop node names, indices, and positions.
#     nodes = lattice.getNodes()
#     if stop is None:
#         stop = nodes[-1].getName()
#     start = get_node_info(start, lattice)
#     stop = get_node_info(stop, lattice)    
    
#     # Add actions.
#     action_container = AccActionsContainer("monitor")
#     if monitor is not None:
#         monitor.position_offset = start["s0"]
#         action_container.addAction(monitor.action, AccActionsContainer.ENTRANCE)
#         action_container.addAction(monitor.action, AccActionsContainer.EXIT)
        
#     # Create params dict and lost bunch.
#     params_dict = dict()
#     params_dict["lostbunch"] = Bunch()

#     if _mpi_rank == 0 and verbose:
#         print(
#             "Tracking from {} (s={}) to {} (s={}).".format(
#                 start["node"].getName(), 
#                 start["s0"],
#                 stop["node"].getName(),
#                 stop["s1"],
#             )
#         )

#     time_start = time.clock()
#     lattice.trackBunch(
#         bunch,
#         paramsDict=params_dict,
#         actionContainer=action_container,
#         index_start=start["index"],
#         index_stop=stop["index"],
#     )
#     monitor.action(params_dict, force_update=True)
    
#     if verbose and _mpi_rank == 0:
#         print("time = {:.3f} [sec]".format(time.clock() - time_start))
        
#     return params_dict


# def track_reverse(bunch, lattice, monitor=None, start=None, stop=0.0, verbose=0):
#     """Track bunch backward from stop to start."""
#     lattice.reverseOrder()
#     bunch = pyorbit_sim.bunch_utils.reverse(bunch)
#     params_dict = track(bunch, lattice, monitor=monitor, start=stop, stop=start, verbose=verbose)
#     lattice.reverseOrder()
#     bunch = pyorbit_sim.bunch_utils.reverse(bunch)
#     return params_dict