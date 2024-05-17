"""Track ring eigenvectors."""
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
from orbit.lattice import AccLattice
from orbit.lattice import AccNode

from orbitsim import coupling
from orbitsim.ring import get_transfer_matrix
from orbitsim.models.sns.ring import SNS_RING
from orbitsim.models.sns.ring import RingInjectionController

import setup


@hydra.main(version_base=None, config_path="./config", config_name="track_eig.yaml")
def main(cfg : DictConfig) -> None:

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    print(OmegaConf.to_yaml(cfg))
    print("output_dir:", output_dir)

    # Initialize ring.
    ring = setup.make_ring(cfg)
    ring = setup.setup_ring(cfg, ring)

    # Compute ring transfer matrix.
    ring.set_fringe(False)
    transfer_matrix = get_transfer_matrix(ring, cfg.bunch.mass, cfg.bunch.energy, ndim=4)
    ring.set_fringe(cfg.lattice.fringe)

    np.savetxt(os.path.join(output_dir, "transfer_matrix.dat"), transfer_matrix)

    # Place one particle on each eigenvector.
    eigenvalues, eigenvectors = np.linalg.eig(transfer_matrix)
    eigenvectors = coupling.normalize_eigenvectors(eigenvectors)
    eigenvectors = eigenvectors[:, ::2]

    v1 = eigenvectors[:, 0]
    v2 = eigenvectors[:, 1]
    psi1 = 0.0
    psi2 = 0.0
    J1 = 50.0e-6
    J2 = 50.0e-6
    
    x1 = np.real(np.sqrt(2.0 * J1) * v1 * np.exp(-1.0j * psi1))
    x2 = np.real(np.sqrt(2.0 * J2) * v2 * np.exp(-1.0j * psi2))
    print("x1 =", x1)
    print("x2 =", x2)

    # Create bunch.
    bunch = Bunch()
    bunch.mass(cfg.bunch.mass)
    bunch.getSyncParticle().kinEnergy(cfg.bunch.energy)
    for x in [x1, x2]:
        bunch.addParticle(x[0], x[1], x[2], x[3], 0.0, 0.0)

    # Track bunch.
    coords = np.zeros((cfg.turns + 1, bunch.getSize(), 6))
    coords[0, 0, :4] = x1
    coords[0, 1, :4] = x2
    for turn in range(1, cfg.turns + 1):
        ring.trackBunch(bunch)
        for i in range(coords.shape[1]):
            x = bunch.x(i)
            y = bunch.y(i)
            z = bunch.z(i)
            xp = bunch.xp(i)
            yp = bunch.yp(i)
            de = bunch.dE(i)
            coords[turn, i, :] = [x, xp, y, yp, z, de]
            print(
                "turn={:05.0f} i={} x={:+0.3f} xp={:+0.3f} y={:+0.3f} yp={:+0.3f}".format(
                    turn, 
                    i, 
                    1000.0 * x, 
                    1000.0 * xp, 
                    1000.0 * y, 
                    1000.0 * yp,
                )
            )
    np.save(os.path.join(output_dir, "coords.npy"), coords)

    print(output_dir)

if __name__ == "__main__":
    main()