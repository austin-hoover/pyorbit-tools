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
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.utils.consts import mass_proton
from orbit.utils.consts import speed_of_light

from orbitsim import coupling
from orbitsim.lattice import read_mad_file
from orbitsim.ring import get_transfer_matrix
from orbitsim.ring import Monitor
from orbitsim.models.sns.ring import SNS_RING
from orbitsim.models.sns.ring import RingInjectionController

import setup


@hydra.main(version_base=None, config_path="./config", config_name="paint_eig.yaml")
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
    
    bunch, lostbunch, params_dict = setup.make_bunch(cfg)
    sync_part = bunch.getSyncParticle()

    macrosize = cfg.inj.intensity / float(cfg.macros_per_turn)
    bunch.macroSize(macrosize)
    
    
    # Initialize lattice
    # --------------------------------------------------------------------------------------    
    ring = setup.make_ring(cfg)
    ring.set_bunch(bunch, lostbunch, params_dict)
    
    
    # Linear transfer matrix analysis
    # --------------------------------------------------------------------------------------
    ring.set_fringe(False)
    
    M = get_transfer_matrix(ring, mass, kin_energy, ndim=4)
    eigenvalues, eigenvectors = np.linalg.eig(M)
    eigentunes = coupling.eigentunes_from_eigenvalues(eigenvalues)
    eigenvectors = coupling.normalize_eigenvectors(eigenvectors)
    eigenvectors = eigenvectors[:, ::2]
    
    print(f"tune_1 = {eigentunes[0]}")
    print(f"tune_2 = {eigentunes[1]}")
    
    ring.set_fringe(cfg.lattice.fringe)
    
    
    # Injection kicker waveforms
    # --------------------------------------------------------------------------------------
    ric = RingInjectionController(
        ring,
        mass=mass,
        kin_energy=kin_energy,        
        inj_start="bpm_a09",
        inj_mid="injm1",
        inj_stop="bpm_b01",
    )

    v1 = eigenvectors[:, 0]
    v2 = eigenvectors[:, 1]
    psi1 = 0.0
    psi2 = 0.0
    J1 = 12.00e-06
    J2 = 0.00e-06
    
    vector = np.zeros(4)
    vector += np.real(np.sqrt(2.0 * J1) * v1 * np.exp(-1.0j * psi1))
    vector += np.real(np.sqrt(2.0 * J2) * v2 * np.exp(-1.0j * psi2))

    # Initial coordinates of closed orbit at injection point [x, x', y, y']
    inj_coords_ti = np.zeros(4)
    inj_coords_ti[0] = ring.inj_x
    inj_coords_ti[1] = ring.inj_xp
    inj_coords_ti[2] = ring.inj_y
    inj_coords_ti[3] = ring.inj_yp
    
    # Final coordinates of closed orbit at injection point  [x, x', y, y']
    inj_coords_tf = np.zeros(4)
    inj_coords_tf[0] = ring.inj_x  - vector[0]
    inj_coords_tf[1] = ring.inj_xp - vector[1]
    inj_coords_tf[2] = ring.inj_y  - vector[2]
    inj_coords_tf[3] = ring.inj_yp - vector[3]

    # Run optimizer.
    kicker_angles_ti = ric.set_inj_coords(*inj_coords_ti)
    kicker_angles_tf = ric.set_inj_coords(*inj_coords_tf)

    print(inj_coords_tf * 1000.0)

    # Create waveforms.
    time_per_turn = ring.getLength() / (sync_part.beta() * speed_of_light)
    ti = 0.0
    tf = cfg.turns_inj * time_per_turn
    strengths_ti = np.ones(8)
    strengths_tf = np.abs(kicker_angles_tf / kicker_angles_ti)
    for node, si, sf in zip(ric.kicker_nodes, strengths_ti, strengths_tf):
        waveform = SquareRootWaveform(sync_part, ring.getLength(), ti, tf, si, sf)
        node.setWaveform(waveform)

    # Set initial kicker settings.
    ric.set_kicker_angles(kicker_angles_ti)


    # Set up lattice
    # --------------------------------------------------------------------------------------

    inj_dist_x = setup.make_transverse_distribution_2d("joho", ring, bunch, **cfg.inj.x)
    inj_dist_y = setup.make_transverse_distribution_2d("joho", ring, bunch, **cfg.inj.y)
    inj_dist_z = setup.make_longitudinal_distribution("sns_espread", ring, bunch, **cfg.inj.z)
    
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
    monitor_node = Monitor(output_dir=output_dir, verbose=cfg.verbose)
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
