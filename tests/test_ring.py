import os
import numpy as np
import pytest

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import DriftTEAPOT
from orbit.teapot import QuadTEAPOT
from orbit.teapot import TEAPOT_Lattice

import orbit_tools as ot


os.makedirs("outputs", exist_ok=True)


def make_bunch(mass: float = 0.938, energy: float = 1.000) -> Bunch:
    bunch = Bunch(mass=mass, energy=energy)
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(energy)
    return bunch


def make_lattice() -> AccLattice:
    # Settings
    length = 5.0  # lattice length [m]
    fill_fraction = 0.5  # quad fill fraction

    # Create nodes
    drift_nodes = [
        DriftTEAPOT("drift1"),
        DriftTEAPOT("drift2"),
        DriftTEAPOT("drift3"),
    ]
    quad_nodes = [
        QuadTEAPOT("qf"),
        QuadTEAPOT("qd"),
    ]

    # Set node lengths
    for node in [quad_nodes[0], quad_nodes[1], drift_nodes[1]]:
        node.setLength(length * fill_fraction / 2.0)
    for node in [drift_nodes[0], drift_nodes[2]]:
        node.setLength(length * fill_fraction / 4.0)

    # Set quad strengths
    quad_nodes[0].setParam("kq", +0.5)
    quad_nodes[1].setParam("kq", -0.5)

    # Build lattice
    lattice = TEAPOT_Lattice()
    lattice.addNode(drift_nodes[0])
    lattice.addNode(quad_nodes[0])
    lattice.addNode(drift_nodes[1])
    lattice.addNode(quad_nodes[1])
    lattice.addNode(drift_nodes[2])
    lattice.initialize()

    # Print nodes
    for node in lattice.getNodes():
        print(node.getName(), node)

    return lattice


def test_get_transfer_matrix():
    lattice = make_lattice()
    M = ot.ring.get_transfer_matrix(lattice, mass=0.938, kin_energy=1.000)
    assert M.shape == (6, 6)


def test_track_twiss():
    lattice = make_lattice()
    ot.ring.track_twiss(lattice, mass=0.938, kin_energy=1.000)


def test_track_dispersion():
    lattice = make_lattice()
    ot.ring.track_dispersion(lattice, mass=0.938, kin_energy=1.000)


def test_match_bunch():
    lattice = make_lattice()
    bunch = make_bunch()

    cov_matrix = np.eye(6)
    points = np.random.multivariate_normal(np.zeros(6), cov_matrix, size=10_000)
    bunch = ot.bunch.set_bunch_coords(bunch, points)

    transfer_matrix = ot.ring.get_transfer_matrix(lattice, mass=0.938, kin_energy=1.000)
    bunch = ot.ring.match_bunch(bunch=bunch, transfer_matrix=transfer_matrix, block_diag=True)


def test_ring_diag_writer():
    lattice = make_lattice()
    bunch = make_bunch()
    params_dict = {"bunch": bunch}

    diag = ot.ring.BunchWriter(output_dir="outputs", verbose=1, freq=1)
    diag(params_dict)


def test_ring_diag_monitor():
    bunch = make_bunch()
    params_dict = {"bunch": bunch}

    diag = ot.ring.BunchMonitor(output_dir="outputs", verbose=1, freq=1)
    diag(params_dict)
