"""Production painting.

Parameters are hard-coded. Eventually I will put default values in a config file.
"""
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

from orbitsim.lattice import read_mad_file
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
    mass = cfg.bunch.mass
    kin_energy = cfg.bunch.energy
    
    minipulse_intensity = cfg.inj.intensity
    macros_per_turn = cfg.macros_per_turn
    macros_total = cfg.turns_inj * macros_per_turn
    macrosize = minipulse_intensity / float(macros_per_turn)
    
    bunch = Bunch()
    bunch.mass(mass)
    bunch.macroSize(macrosize)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    
    lostbunch = Bunch()
    lostbunch.addPartAttr("LostParticleAttributes")
    
    params_dict = {}
    params_dict["bunch"] = bunch
    params_dict["lostbunch"] = lostbunch
    
    
    # Initialize lattice
    # --------------------------------------------------------------------------------------    
    ring = SNS_RING(
        inj_x=cfg.inj.x.pos,
        inj_y=cfg.inj.y.pos,
        inj_xp=cfg.inj.x.mom,
        inj_yp=cfg.inj.y.mom,
    )
    ring = read_mad_file(ring, cfg.lattice.path, cfg.lattice.seq, kind="auto")
    ring.initialize()
    ring.set_bunch(bunch, lostbunch, params_dict)
    
    
    # Set injection kicker waveforms
    # --------------------------------------------------------------------------------------

    # [To do: set up from config?]
    
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
    
    
    # Add nodes
    # --------------------------------------------------------------------------------------
    inj_dist_x = setup.make_joho_dist(**cfg.inj.x)
    inj_dist_y = setup.make_joho_dist(**cfg.inj.y)
    inj_dist_z = setup.make_sns_espread_dist(ring, bunch, **cfg.inj.z)

    ring.add_injection_node(
        n_parts=macros_per_turn,
        dist_x=inj_dist_x,
        dist_y=inj_dist_y,
        dist_z=inj_dist_z,
        n_parts_max=macros_total,
        parent_index=0,
    )
    
    if cfg.lattice.foil:
        ring.add_foil_node(**cfg.foil)

    if cfg.lattice.apertures:
        ring.add_injection_chicane_apertures_and_displacements()
        ring.add_apertures()

    if cfg.lattice.rf:
        ring.add_rf_cavities(**cfg.rf)

    if cfg.lattice.impedance.z:
        ring.add_longitudinal_impedance_node(**cfg.impedance.z)

    if cfg.lattice.impedance.xy:
        ring.add_transverse_impedance_node(**cfg.impedance.xy)

    if cfg.lattice.spacecharge.z:
        ring.add_longitudinal_spacecharge_node(**cfg.spacecharge.z)

    if cfg.lattice.spacecharge.xy:
        ring.add_transverse_spacecharge_nodes(**cfg.spacecharge.xy)

    
    # Diagnostics
    # --------------------------------------------------------------------------------------

    monitor_nodes = []
    
    monitor_node = Monitor(output_dir=output_dir, verbose=True)
    monitor_nodes.append(monitor_node)

    # Plotting node
    # [...]

    
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
