"""Production painting."""
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
from orbit.kickernodes import SquareRootWaveform
from orbit.lattice import AccNode
from orbit.utils.consts import mass_proton

from orbitsim.models.sns.ring import SNS_RING
from orbitsim.ring import Monitor

import setup


@hydra.main(version_base=None, config_path="./config", config_name="paint_prod.yaml")
def main(cfg : DictConfig) -> None:

    # Setup
    # --------------------------------------------------------------------------------------
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if _mpi_rank == 0:
        print(OmegaConf.to_yaml(cfg))
        print("output_dir:", output_dir)
        print("mpi_size:", _mpi_size)

    
    # Initialize bunch
    # --------------------------------------------------------------------------------------
    bunch, lostbunch, params_dict = setup.make_bunch(cfg)
    macrosize = cfg.inj.intensity / float(cfg.macros_per_turn)
    bunch.macroSize(macrosize)
    
    
    # Initialize lattice
    # --------------------------------------------------------------------------------------    
    ring = setup.make_ring(cfg)
    ring.set_bunch(bunch, lostbunch, params_dict)
    
    
    # Set injection kicker waveforms
    # --------------------------------------------------------------------------------------    
    tih = -0.001  # [s]
    tiv = -0.002  # [s]
    tf = 0.001  # [s]
    si = 1.0  # initial amplitude
    sfh = 0.457  # final amplitude (x)
    sfv = 0.406  # final amplitude (y)
    
    strength_hkicker10 = 14.04e-03  # [units]
    strength_vkicker10 = 8.84e-03
    strength_hkicker11 = -4.28e-03
    strength_vkicker11 = -5.06e-03
    strength_hkicker12 = -4.36727974875e-03
    strength_vkicker12 = -5.32217284098e-03
    strength_hkicker13 = 14.092989681e-03
    strength_vkicker13 = 9.0098984536e-03
    
    sync_part = bunch.getSyncParticle()
    hkickerwave = SquareRootWaveform(sync_part, ring.getLength(), tih, tf, si, sfh)
    vkickerwave = SquareRootWaveform(sync_part, ring.getLength(), tiv, tf, si, sfv)
    
    ring.inj_kicker_nodes[0].setParam("kx", strength_hkicker10)
    ring.inj_kicker_nodes[1].setParam("ky", strength_vkicker10)
    ring.inj_kicker_nodes[2].setParam("kx", strength_hkicker11)
    ring.inj_kicker_nodes[3].setParam("ky", strength_vkicker11)
    ring.inj_kicker_nodes[4].setParam("ky", strength_vkicker12)
    ring.inj_kicker_nodes[5].setParam("kx", strength_hkicker12)
    ring.inj_kicker_nodes[6].setParam("ky", strength_vkicker13)
    ring.inj_kicker_nodes[7].setParam("kx", strength_hkicker13)
    
    ring.inj_kicker_nodes[0].setWaveform(hkickerwave)
    ring.inj_kicker_nodes[1].setWaveform(vkickerwave)
    ring.inj_kicker_nodes[2].setWaveform(hkickerwave)
    ring.inj_kicker_nodes[3].setWaveform(vkickerwave)
    ring.inj_kicker_nodes[4].setWaveform(vkickerwave)
    ring.inj_kicker_nodes[5].setWaveform(hkickerwave)
    ring.inj_kicker_nodes[6].setWaveform(vkickerwave)
    ring.inj_kicker_nodes[7].setWaveform(hkickerwave)
    
    
    # Set up lattice
    # --------------------------------------------------------------------------------------
    inj_dist_x = setup.make_dist_tran_joho(**cfg.inj.x)
    inj_dist_y = setup.make_dist_tran_joho(**cfg.inj.y)
    inj_dist_z = setup.make_dist_long_sns_espread(ring, bunch, **cfg.inj.z)

    ring.add_injection_node(
        n_parts=cfg.macros_per_turn,
        dist_x=inj_dist_x,
        dist_y=inj_dist_y,
        dist_z=inj_dist_z,
        n_parts_max=(cfg.turns_inj * cfg.macros_per_turn),
        parent_index=0,
    )
    
    ring = setup.setup_ring(cfg, ring)

    
    # Diagnostics
    # --------------------------------------------------------------------------------------
    monitor_nodes = []
    monitor_node = Monitor(output_dir=output_dir, verbose=True)
    monitor_nodes.append(monitor_node)

    
    # Tracking
    # --------------------------------------------------------------------------------------
    turns = range(cfg.turns_inj + cfg.turns_store)
    if cfg.progbar:
        turns = tqdm(turns)
    
    for turn in turns:
        ring.trackBunch(bunch, params_dict)
    
        if turn % cfg.write_bunch_freq == 0:
            filename = f"bunch_{turn:05.0f}.dat"
            filename = os.path.join(output_dir, filename)
            bunch.dumpBunch(filename)

        for monitor_node in monitor_nodes:
            monitor_node(params_dict)


if __name__ == "__main__":
    main()
