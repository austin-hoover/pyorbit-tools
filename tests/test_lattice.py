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


def test_get_sublattice():
    lattice = make_lattice()
    sublattice = ot.lattice.get_sublattice(lattice, start=0, stop=5)
    assert sublattice.getNodes()[0] is lattice.getNodes()[0]
    assert sublattice.getNodes()[4] is lattice.getNodes()[4]


def test_split_node():
    lattice = make_lattice()
    node = lattice.getNodes()[0]
    node = ot.lattice.split_node(node, max_part_length=0.1)


def test_split_lattice():
    lattice = make_lattice()
    node = lattice.getNodes()[0]
    lattice = ot.lattice.split_lattice(lattice, 0.1)


def test_get_node_for_name_any_case():
    lattice = make_lattice()
    name = lattice.getNodes()[0].getName()
    node1 = ot.lattice.get_node_for_name_any_case(lattice, name=name.lower())
    node2 = ot.lattice.get_node_for_name_any_case(lattice, name=name.upper())
    assert node1 is node2
