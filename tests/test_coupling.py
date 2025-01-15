import numpy as np
import pytest

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import DriftTEAPOT
from orbit.teapot import QuadTEAPOT
from orbit.teapot import TEAPOT_Lattice

import orbit_tools as ot


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


def test_eigtunes():
    lattice = make_lattice()
    bunch = make_bunch()
    transfer_matrix = ot.ring.get_transfer_matrix(
        lattice, mass=bunch.mass(), kin_energy=bunch.getSyncParticle().kinEnergy()
    )
    eigtunes = ot.coupling.eigentunes(transfer_matrix)


def test_eigtunes_from_eigvals():
    lattice = make_lattice()
    bunch = make_bunch()
    transfer_matrix = ot.ring.get_transfer_matrix(
        lattice, mass=bunch.mass(), kin_energy=bunch.getSyncParticle().kinEnergy()
    )
    eigvals, eigvecs = np.linalg.eig(transfer_matrix)
    eigtunes = ot.coupling.eigtunes_from_eigvals(eigvals)
