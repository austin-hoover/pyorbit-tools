import os
import sys
import time
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from orbit.core.bunch import Bunch
from orbit.lattice import AccNode
from orbit.lattice import AccLattice


def get_sublattice(
    lattice: AccLattice,
    start: int | str = None,
    stop: int | str = None,
) -> AccLattice:
    if type(start) is str:
        start = lattice.getNodeIndex(lattice.getNodeForName(start))
    if type(stop) is str:
        stop = lattice.getNodeIndex(lattice.getNodeForName(stop))
    return lattice.getSubLattice(start, stop)


def split_node(node: AccNode, max_part_length: float = None) -> AccNode:
    if max_part_length is not None and max_part_length > 0.0:
        if node.getLength() > max_part_length:
            node.setnParts(1 + int(node.getLength() / max_part_length))
    return node


def split_lattice(lattice: AccLattice, max_part_length: float = None) -> AccLattice:
    for node in lattice.getNodes():
        split_node(node, max_part_length)
    return lattice


def get_node_for_name_any_case(lattice: AccLattice, name: str) -> AccNode:
    nodes = lattice.getNodes()
    node_names = [node.getName() for node in nodes]
    if name not in node_names:
        if name.lower() in node_names:
            name = name.lower()
        elif name.upper() in node_names:
            name = name.upper()
        else:
            raise ValueError(f"node {name} not found")
    return lattice.getNodeForName(name)


def get_node_name_and_index(
    lattice: AccLattice,
    node: AccNode = None,
    name: str = None,
    index: int = None,
) -> tuple[AccNode, str, int]:
    if node is None:
        if name is not None:
            node = lattice.getNodeForName(name)
        elif index is not None:
            node = lattice.getNodes()[index]
        else:
            raise ValueError("Must provide node, name or index.")
    name = node.getName()
    index = lattice.getNodeIndex(node)
    return (node, name, index)


def get_nodes_names_and_indices(
    lattice: AccLattice,
    nodes: list[AccNode] = None,
    names: list[str] = None,
    indices: list[int] = None,
) -> tuple[list[AccNode], list[str], list[int]]:
    if nodes is None:
        if names is not None:
            nodes = [lattice.getNodeForName(name) for name in names]
        elif indices is not None:
            nodes = [lattice.getNodes()[index] for index in indices]
        else:
            raise ValueError("Must provide nodes, names or indices.")

    names = [node.getName() for node in nodes]
    indices = [lattice.getNodeIndex(node) for node in nodes]
    return (nodes, names, indices)
