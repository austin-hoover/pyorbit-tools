import os
import pathlib
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm import tqdm
from tqdm import trange

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_generators import WaterBagDist3D
from orbit.bunch_generators import TwissContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice import AxisFieldRF_Gap
from orbit.py_linac.lattice import AxisField_and_Quad_RF_Gap
from orbit.py_linac.lattice import BaseRF_Gap
from orbit.py_linac.lattice import Bend
from orbit.py_linac.lattice import Drift
from orbit.py_linac.lattice import LinacApertureNode
from orbit.py_linac.lattice import LinacEnergyApertureNode
from orbit.py_linac.lattice import LinacPhaseApertureNode
from orbit.py_linac.lattice import OverlappingQuadsNode 
from orbit.py_linac.lattice import Quad

import orbitsim.bunch
import orbitsim.linac
from orbitsim.linac import BunchWriter
from orbitsim.linac import BunchMonitor
from orbitsim.linac import unnormalize_emittances
from orbitsim.linac import unnormalize_beta_z
from orbitsim.misc import get_lorentz_factors
from orbitsim.models.sns import SNS_LINAC


@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(cfg : DictConfig) -> None:

    # Setup
    # ------------------------------------------------------------------------------------

    # MPI
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)
    
    # Print config parameters.
    print(OmegaConf.to_yaml(cfg))

    # Print output directory.
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print("output_dir:", output_dir)


    # Lattice
    # ------------------------------------------------------------------------------------
    
    model = SNS_LINAC(
        sequence_start=cfg.lattice.seq_start,
        sequence_stop=cfg.lattice.seq_stop,
        max_drift=cfg.lattice.max_drift,
        rf_freq=cfg.lattice.rf_freq,
    )

    lattice = model.lattice

    model.set_rf_gap_model(cfg.rf.gap)

    if cfg.lattice.overlap:
        model.set_overlapping_rf_and_quad_fields(
            sequences=model.sequences,
            z_step=0.002,
            xml_filename=None,
        )

    if cfg.lattice.sc:
        sc_nodes = model.add_sc_nodes(
            solver=cfg.sc.solver,
            gridx=cfg.sc.gridx,
            gridy=cfg.sc.gridy,
            gridz=cfg.sc.gridz,
            path_length_min=cfg.sc.path_length_min,
            n_ellipsoids=cfg.sc.n_ellipsoids,
        )
        
    if cfg.lattice.apertures.transverse:
        model.add_aperture_nodes(
            scrape_x=cfg.apertures.scrape.x,
            scrape_y=cfg.apertures.scrape.y,
        )

    if cfg.lattice.apertures.phase:
        model.add_phase_aperture_nodes(
            phase_min=cfg.apertures.phase.min,
            phase_max=cfg.apertures.phase.max,
            classes=[
                BaseRF_Gap, 
                AxisFieldRF_Gap, 
                AxisField_and_Quad_RF_Gap,
                Quad, 
                OverlappingQuadsNode,
            ],
            drifts=True,
            drift_start=0.0,
            drift_stop=4.0,
            drift_step=0.050,
        )
        
    if cfg.lattice.apertures.energy:
        model.add_energy_aperture_nodes(
            energy_min=cfg.apertures.energy.min,
            energy_max=cfg.apertures.energy.max,
            classes=[
                BaseRF_Gap, 
                AxisFieldRF_Gap, 
                AxisField_and_Quad_RF_Gap,
                Quad, 
                OverlappingQuadsNode,
            ],
            drifts=True,
            drift_step=0.1,
        )
        
    if cfg.tracker == "linac":
        lattice.setLinacTracker(True)
    else:
        lattice.setLinacTracker(False)


    aperture_nodes = model.aperture_nodes
    sc_nodes = model.sc_nodes

    
    # Bunch
    # ------------------------------------------------------------------------------------
    
    mass = cfg.bunch.mass
    charge = cfg.bunch.charge
    kin_energy = cfg.bunch.energy

    bunch = Bunch()
    bunch.mass(cfg.bunch.mass)
    bunch.charge(cfg.bunch.charge)
    bunch.getSyncParticle().kinEnergy(cfg.bunch.energy)

    if cfg.bunch.path:
        # Load bunch from file.
        pass
    else:
        # Sample coordinates from distribution function.
        dist_kws = dict(cfg.bunch.dist)
        dist_name = dist_kws.pop("name")
        alpha_x = dist_kws.pop("alpha_x")
        alpha_y = dist_kws.pop("alpha_y")
        alpha_z = dist_kws.pop("alpha_z")
        beta_x = dist_kws.pop("beta_x")
        beta_y = dist_kws.pop("beta_y")
        beta_z = dist_kws.pop("beta_z")
        eps_x = dist_kws.pop("eps_x")
        eps_y = dist_kws.pop("eps_y")
        eps_z = dist_kws.pop("eps_z")

        (eps_x, eps_y, eps_z) = unnormalize_emittances(mass, kin_energy, eps_x, eps_y, eps_z)
        beta_z = unnormalize_beta_z(mass, kin_energy, beta_z)

        dist_constructors = {
            "gaussian": GaussDist3D,
            "kv": KVDist3D,
            "waterbag": WaterBagDist3D,
        }
        dist_constructor = dist_constructors[dist_name]
        
        dist = dist_constructor(
            TwissContainer(alpha_x, beta_x, eps_x),
            TwissContainer(alpha_y, beta_y, eps_y),
            TwissContainer(alpha_z, beta_z, eps_z),
            **dist_kws
        )
        sample = dist.getCoordinates
        
        size = cfg.bunch.size
        if size is None:
            size = 100_000

        bunch = orbitsim.bunch.generate(sample=sample, size=size, bunch=bunch)

    # Set macroparticle size.
    bunch = orbitsim.bunch.set_current(bunch=bunch, current=cfg.bunch.current, frequency=cfg.lattice.rf_freq)


    # Diagnostics
    # ------------------------------------------------------------------------------------
    
    bunch_plotter = None
    
    bunch_writer = BunchWriter(
        output_dir=output_dir,
        output_name="bunch",
        output_ext="dat",
        output_index_format="04.0f",
        verbose=True,
    )

    bunch_monitor = BunchMonitor(
        plot=bunch_plotter,
        write=bunch_writer,
        stride=cfg.stride,
        stride_write=cfg.stride_write,
        stride_plot=cfg.stride_plot,
        dispersion_flag=False,
        emit_norm_flag=False,
        position_offset=0.0,
        verbose=True,
        rf_frequency=cfg.lattice.rf_freq,
        history_filename=os.path.join(output_dir, "history.dat"),
    )

    
    # Track
    # ------------------------------------------------------------------------------------

    if _mpi_rank == 0:
        print("Tracking design bunch")
        
    design_bunch = lattice.trackDesignBunch(bunch)

    orbitsim.linac.check_sync_time(
        bunch=bunch,
        lattice=lattice,
        start=cfg.start,
        set_design=False,
        verbose=True,
    )
       
    params_dict = orbitsim.linac.track(
        bunch=bunch,
        lattice=lattice,
        monitor=bunch_monitor,
        start=cfg.start,
        stop=cfg.stop,
        verbose=True,
    )

    print("output_dir:", output_dir)

    
if __name__ == "__main__":
    main()