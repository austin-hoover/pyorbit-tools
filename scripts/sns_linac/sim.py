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
from orbit.core.linac import BaseRfGap
from orbit.core.linac import MatrixRfGap
from orbit.core.linac import RfGapTTF
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_generators import WaterBagDist3D
from orbit.bunch_generators import TwissContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode

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
        rf_frequency=cfg.lattice.rf_frequency,
    )

    lattice = model.lattice
    
    for rf_gap in lattice.getRF_Gaps():
    	rf_gap.setCppGapModel(RfGapTTF())

    
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
        alpha_x = cfg.bunch.alpha_x
        alpha_y = cfg.bunch.alpha_y
        alpha_z = cfg.bunch.alpha_z
        beta_x = cfg.bunch.beta_x
        beta_y = cfg.bunch.beta_y
        beta_z = cfg.bunch.beta_z
        eps_x = cfg.bunch.eps_x
        eps_y = cfg.bunch.eps_y
        eps_z = cfg.bunch.eps_z
    
        (eps_x, eps_y, eps_z) = unnormalize_emittances(mass, kin_energy, eps_x, eps_y, eps_z)
        beta_z = unnormalize_beta_z(mass, kin_energy, beta_z)
        
        dist = WaterBagDist3D(
            TwissContainer(alpha_x, beta_x, eps_x),
            TwissContainer(alpha_y, beta_y, eps_y),
            TwissContainer(alpha_z, beta_z, eps_z),
        )
        
        size = cfg.bunch.size
        if size is None:
            size = 100_000

        bunch = orbitsim.bunch.generate(sample=dist.getCoordinates, size=size, bunch=bunch)

    # Set macro-particle size.
    bunch = orbitsim.bunch.set_current(bunch=bunch, current=cfg.bunch.current, frequency=cfg.lattice.rf_frequency)


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
        rf_frequency=cfg.lattice.rf_frequency,
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

    
if __name__ == "__main__":
    main()