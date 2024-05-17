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
from orbit.matrix_lattice import MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice


def get_matrix_lattice(lattice: AccLattice, mass: float, kin_energy: float) -> MATRIX_Lattice:
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    return TEAPOT_MATRIX_Lattice(lattice, bunch)


def get_twiss(lattice: AccLattice) -> dict:
    (pos_nu_x, pos_alpha_x, pos_beta_x) = matrix_lattice.getRingTwissDataX()
    (pos_nu_y, pos_alpha_y, pos_beta_y) = matrix_lattice.getRingTwissDataY()
    data = {}
    data["s"] = np.array(pos_nu_x)[:, 0]
    data["nu_x"] = np.array(pos_nu_x)[:, 1]
    data["nu_y"] = np.array(pos_nu_y)[:, 1]
    data["alpha_x"] = np.array(pos_alpha_x)[:, 1]
    data["alpha_y"] = np.array(pos_alpha_y)[:, 1]
    data["beta_x"] = np.array(pos_beta_x)[:, 1]
    data["beta_y"] = np.array(pos_beta_y)[:, 1]
    return data
    

def get_dispersion(matrix_lattice):
    (pos_disp_x, pos_disp_p_x) = matrix_lattice.getRingDispersionDataX()
    (pos_disp_y, pos_disp_p_y) = matrix_lattice.getRingDispersionDataY()
    data = {}
    data["s"] = np.array(pos_disp_x)[:, 0]
    data["disp_x"] = np.array(pos_disp_x)[:, 1]
    data["disp_y"] = np.array(pos_disp_y)[:, 1]
    data["disp_xp"] = np.array(pos_disp_p_x)[:, 1]
    data["disp_yp"] = np.array(pos_disp_p_y)[:, 1]
    return data


def get_sublattice(
    lattice: AccLattice, 
    start: Union[int, str] = None, 
    stop: Union[int, str] = None, 
) -> AccLattice:
    def get_index(argument, default=0):
        index = default_index
        if type(argument) is str:
            name = argument
            index = lattice.getNodeIndex(lattice.getNodeForName(node_name))
        else:
            index = argument
        return index

    start_index = get_index(start, default=0)
    stop_index = get_index(stop, default=-1)
    return lattice.getSubLattice(start_index, stop_index)


def split_node(node: AccNode, max_part_length: float = None) -> AccNode:
    if max_part_length is not None and max_part_length > 0.0:
        if node.getLength() > max_part_length:
            node.setnParts(1 + int(node.getLength() / max_part_length))
    return node


def split(lattice: AccLattice, max_part_length: float = None) -> AccLattice:
    for node in lattice.getNodes():
        split_node(node, max_part_length)
    return lattice


def set_node_fringe(node: AccNode, setting: bool = False) -> AccNode:
    if hasattr(node, "setFringeFieldFunctionIN"):
        node.setUsageFringeFieldIN(setting)
    if hasattr(node, "setFringeFieldFunctionOUT"):
        node.setUsageFringeFieldIN(setting)
    return node


def set_fringe(lattice: AccLattice, setting: bool) -> AccLattice:
    for node in lattice.getNodes():
        set_node_fringe(node, setting)
    return lattice


def read_mad_file(lattice: AccLattice, path: str, sequence: str, kind: str = "auto") -> AccLattice:
    if not os.path.exists(path):
        raise FileNotFoundError 

    if kind == "auto":
        # MADX output is lowercase; MAD is upercase.
        kind = "madx"
        file = open(path, "r")
        for line in file:
            if line.isupper():
                kind = "mad"
                break
        file.close()
            
    if kind == "madx":
        lattice.readMADX(path, sequence)
    elif kind == "mad":
        lattice.readMAD(path, sequence)
    else:
        raise ValueError(f"Invalid kind {kind}")

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
