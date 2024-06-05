import os
import sys
import time
from typing import Callable
from typing import Union

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
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice import BaseLinacNode

from orbitsim.bunch import get_z_to_phase_coeff
from orbitsim.misc import get_lorentz_factors
from orbitsim.stats import apparent_emittances
from orbitsim.stats import intrinsic_emittances


def unnormalize_emittances(
    mass: float,
    kin_energy: float,
    eps_x: float,
    eps_y: float,
    eps_z: float,
    beta_z: float = None,
) -> float:
    (gamma, beta) = get_lorentz_factors(mass, kin_energy)
    eps_x /= (beta * gamma)  # [m * rad]
    eps_y /= (beta * gamma)  # [m * rad]
    eps_z /= (beta * gamma**3)  # [m * rad]    
    eps_z *= (gamma**3 * beta**2 * mass)  # [m * GeV]
    return (eps_x, eps_y, eps_z)
    

def unnormalize_beta_z(mass: float, kin_energy: float, beta_z: float) -> float:
    (gamma, beta) = get_lorentz_factors(mass, kin_energy)
    beta_z /= (gamma**3 * beta**2 * mass)
    return beta_z


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
        verbose: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.index = index

        self.output_name = output_name
        self.output_index_format = output_index_format
        self.output_ext = output_ext

        self.verbose = verbose
        
    def __call__(self, bunch: Bunch, tag: str = None) -> None:
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        filename = self.output_name
        if self.index is not None:
            fstr = "{}_{:" + self.output_index_format + "}"
            filename = fstr.format(filename, self.index)
        if tag is not None:
            filename = "{}_{}".format(filename, tag)
        filename = "{}.dat".format(filename)
        filename = os.path.join(self.output_dir, filename)
        
        if _mpi_rank == 0 and self.verbose:
            print("Writing bunch to file {}".format(filename))
            
        bunch.dumpBunch(filename)
        
        if self.index is not None:
            self.index += 1
            

class BunchMonitor:
    def __init__(
        self,
        plot: Callable,
        write: Callable,
        stride: float = 0.1,
        stride_plot: float = None,
        stride_write: float = None,
        dispersion_flag: bool = False,
        emit_norm_flag: bool = False,
        position_offset: float = 0.0,
        verbose: bool = True,
        rf_frequency: float = 402.5e+06,
        history_filename: str = None,
        last_node_name: str = None,
    ) -> None:
        """Constructor.
        
        Parameters
        ----------
        plot : Callable
            Calls plotting functions.
        write : Callable
            Writes the bunch coordinates to a file.
        stride : float
            Distance [m] between updates.
        stride_plot: float
            Distance [m] between calls to `self.plot(bunch)`.
        stride_write : float
            Distance [m] between calls to `self.write(bunch)`.
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
        self.verbose = verbose

        self.stride = stride
        self.stride_plot = stride_plot
        self.stride_write = stride_write

        if self.stride is None:
            self.stride = 0.100
        if self.stride_plot is None:
            self.stride_plot = np.inf
        if self.stride_write is None:
            self.stride_write = np.inf
        
        self.write = write
        self.plot = plot

        # State
        self.position = self.position_offset = position_offset
        self.last_plot_position = self.position
        self.last_write_position = self.position
        self.index = 0
        self.start_time = None

        # Store scalars only for the last update.
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
                "xp_rms",
                "yp_rms",
                "de_rms",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "z_min",
                "z_max",
                "eps_x",
                "eps_y",
                "eps_z",
                "eps_1",
                "eps_2",
                "eps_xy",
                "eps_xz",
                "eps_yz",
                "eps_xyz",
            ]
            for i in range(6):
                keys.append("mean_{}".format(i))
            for i in range(6):
                for j in range(i + 1):
                    keys.append("cov_{}-{}".format(j, i))
                    
            self.history = dict()
            for key in keys:
                self.history[key] = None

            self.history_filename = history_filename
            self.history_file = None
            if self.history_filename is not None:
                self.history_file = open(self.history_filename, "w")
                line = ",".join(list(self.history))
                line = line[:-1] + "\n"
                self.history_file.write(line)
            
    def __call__(self, params_dict: dict, force_update: bool = False) -> None:
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        if self.index == 0:
            force_update = True
        
        # Update position; decide whether to proceed.
        position = params_dict["path_length"] + self.position_offset
        if not force_update:
            if self.stride > (position - self.position):
                return
        self.position = position
        
        # Update clock.
        if self.start_time is None:
            self.start_time = time.time()
        time_ellapsed = time.time() - self.start_time
        
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

        # Measure mean and covariance.
        twiss_analysis = BunchTwissAnalysis()
        order = 2
        twiss_analysis.computeBunchMoments(bunch, order, self.dispersion_flag, self.emit_norm_flag)

        mean = np.zeros(6)
        for i in range(6):
            key = "mean_{}".format(i)
            value = twiss_analysis.getAverage(i)
            if _mpi_rank == 0:
                self.history[key] = value
            mean[i] = value

        cov = np.zeros((6, 6))
        for i in range(6):
            for j in range(i + 1):
                key = "cov_{}-{}".format(j, i)
                value = twiss_analysis.getCorrelation(j, i)
                if _mpi_rank == 0:
                    self.history[key] = value
                cov[i, j] = cov[j, i] = value
                                                   
        if _mpi_rank == 0:
            (x_rms, xp_rms, y_rms, yp_rms, z_rms, de_rms) = np.sqrt(np.diag(cov))
            z_to_phase_coeff = get_z_to_phase_coeff(bunch, self.rf_frequency)
            z_rms_deg = -z_to_phase_coeff * z_rms
            
            self.history["x_rms"] = x_rms
            self.history["y_rms"] = y_rms
            self.history["z_rms"] = z_rms
            self.history["z_rms_deg"] = z_rms_deg
            self.history["z_to_phase_coeff"] = z_to_phase_coeff
            self.history["xp_rms"] = xp_rms
            self.history["yp_rms"] = yp_rms
            self.history["de_rms"] = de_rms

        # Compute rms emittances for convenience.
        if _mpi_rank == 0:
            (eps_x, eps_y, eps_z) = apparent_emittances(cov[:6, :6])
            (eps_1, eps_2) = intrinsic_emittances(cov[:4, :4])
            eps_xy = np.sqrt(np.linalg.det(cov[:4, :4]))
            eps_xz = np.sqrt(np.linalg.det(cov[np.ix_([0, 1, 4, 5], [0, 1, 4, 5])]))
            eps_yz = np.sqrt(np.linalg.det(cov[np.ix_([2, 3, 4, 5], [2, 3, 4, 5])]))
            eps_xyz = np.sqrt(np.linalg.det(cov[:6, :6]))

            self.history["eps_x"] = eps_x
            self.history["eps_y"] = eps_y
            self.history["eps_z"] = eps_z
            self.history["eps_1"] = eps_1
            self.history["eps_2"] = eps_2
            self.history["eps_xy"] = eps_xy
            self.history["eps_xz"] = eps_xz
            self.history["eps_yz"] = eps_yz
            self.history["eps_xyz"] = eps_xyz

        # Measure maximum phase space coordinates.
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
            fstr = [
                "step={:05.0f}",
                "time={:0.2f}",
                "s={:0.3f}",
                "ekin={:0.3f}",
                "xrms={:0.3f}",
                "yrms={:0.3f}",
                "zrms={:0.3f}",
                "size={}",
                "node={}",
            ]
            fstr = " ".join(fstr)
            message = fstr.format(
                self.index,
                time_ellapsed,
                position,
                1000.0 * bunch.getSyncParticle().kinEnergy(),
                1000.0 * x_rms,
                1000.0 * y_rms,
                z_rms_deg,
                bunch_size_global,
                node.getName(),
            )
            print(message)
                                                
        # Write phase space coordinates to file.
        if self.write is not None:
            if force_update or (self.stride_write <= (position - self.last_write_position)):
                self.write(bunch, tag=node.getName())
                self.last_write_position = position

        # Call plotting routines.
        if self.plot is not None and _mpi_rank == 0:
            if force_update or (self.stride_plot <= (position - self.last_plot_position)):
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
                self.last_plot_position = position
                
        # Write new line to history file.
        if _mpi_rank == 0 and self.history_file is not None:
            data = [self.history[key] for key in self.history]
            line = ""
            for i in range(len(data)):
                line += "{},".format(data[i])
            line = line[:-1] + "\n"
            self.history_file.write(line)

        self.index += 1
    
    
def get_node_info(name: str = None, position: float = None, lattice: AccLattice = None) -> dict:
    """Return node, node index, start and stop position from node name or center position.

    Returns dict:
        "node" : str
            The AccNode instance.
        "index" : int
            The node index in the lattice.
        "pos_start", "pos_stop" : float
            The node's start/stop position.
    """
    if (name is None) and (position is None):
        raise ValueError("Must provide node name or center position")

    if position is not None:
        (node, index, s0, s1) = lattice.getNodeForPosition(position)
    else:
        node = lattice.getNodeForName(name)
        index = lattice.getNodeIndex(node)
        s0 = node.getPosition() - 0.5 * node.getLength()
        s1 = node.getPosition() + 0.5 * node.getLength()

    info = {
        "node": node,
        "name": node.getName(),
        "index": index,
        "start_position": s0,
        "stop_position" : s1,
    }
    return info


def get_node_info_from_name_or_position(name_or_position: Union[str, float], lattice: AccLattice) -> dict:
    info = {}
    if type(name_or_position) is str:
        name = name_or_position
        info = get_node_info(lattice=lattice, name=name)
    else:
        position = name_or_position
        info = get_node_info(lattice=lattice, position=position)
    return info
    
    
def check_sync_time(
    bunch: Bunch, 
    lattice: AccLattice, 
    start: float = 0.0, 
    set_design: bool = False, 
    verbose: bool = True
) -> None:
    """Check if the synchronous particle time is set to the design value at start.
    Optionally update the synchronous particle time if incorrect.
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

    start_node_info = get_node_info_from_name_or_position(start, lattice)

    sync_time = bunch.getSyncParticle().time()
    sync_time_design = 0.0
    
    if start_node_info["index"] > 0:
        design_bunch = lattice.trackDesignBunch(bunch, index_start=0, index_stop=start["index"])
        sync_time_design = design_bunch.getSyncParticle().time()
        
    if _mpi_rank == 0 and verbose:
        print("Start index = {}:".format(start_node_info["index"]))
        print("    Synchronous particle time (actual) = {}".format(sync_time))
        print("    Synchronous particle time (design) = {}".format(sync_time_design))
        
    if set_design and abs(sync_time - sync_time_design) > 1.00e-30:
        if _mpi_rank == 0 and verbose:
            print("    Setting to design value.")
        bunch.getSyncParticle().time(sync_time_design)
        if _mpi_rank == 0 and verbose:
            print("bunch.getSyncParticle().time() = {}".format(bunch.getSyncParticle().time()))
            

def track(
    bunch: Bunch, 
    lattice: AccLattice, 
    monitor: Callable = None, 
    start: Union[str, float] = 0.0, 
    stop: Union[str, float] = None,
    verbose: bool = True,
) -> dict:
    """Track bunch from start to stop node/position.
    
    monitor: callable
        Call signature: `monitor(params_dict: dict, force_update: bool)`.
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

    # Get start/stop node names, indices, and positions.
    nodes = lattice.getNodes()
    if stop is None:
        stop = nodes[-1].getName()
    start_node_info = get_node_info_from_name_or_position(start, lattice)
    stop_node_info = get_node_info_from_name_or_position(stop, lattice)
        
    # Add actions.
    action_container = AccActionsContainer("monitor")
    if monitor is not None:
        monitor.position_offset = start_node_info["start_position"]
        action_container.addAction(monitor, AccActionsContainer.ENTRANCE)
        action_container.addAction(monitor, AccActionsContainer.EXIT)
        
    # Create params dict and lost bunch.
    params_dict = dict()
    params_dict["lostbunch"] = Bunch()

    if _mpi_rank == 0 and verbose:
        print(
            "Tracking from {} (s={:0.5f}) to {} (s={:0.5f}).".format(
                start_node_info["name"],
                start_node_info["start_position"],
                stop_node_info["name"],
                stop_node_info["stop_position"],
            )
        )

    time_start = time.time()
    lattice.trackBunch(
        bunch,
        paramsDict=params_dict,
        actionContainer=action_container,
        index_start=start_node_info["index"],
        index_stop=stop_node_info["index"],
    )
    monitor(params_dict, force_update=True)
    
    if verbose and _mpi_rank == 0:
        print("time = {:.3f} [sec]".format(time.time() - time_start))
        
    return params_dict


def track_reverse(
    bunch: Bunch, 
    lattice: AccLattice, 
    monitor: Callable = None, 
    start: Union[str, float] = None,
    stop: Union[str, float] = 0.0,
    verbose: bool = True,
) -> dict:
    """Track bunch backward from stop to start node/position.

    See `track` method arguments.
    """
    lattice.reverseOrder()
    bunch = pyorbit_sim.bunch_utils.reverse(bunch)
    params_dict = track(bunch, lattice, monitor=monitor, start=stop, stop=start, verbose=verbose)
    lattice.reverseOrder()
    bunch = pyorbit_sim.bunch_utils.reverse(bunch)
    return params_dict
