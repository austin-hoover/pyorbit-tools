import math
import os
import pathlib

import numpy as np
import scipy.optimize

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory


SEQUENCES = [
    "MEBT",
    "DTL1",
    "DTL2",
    "DTL3",
    "DTL4",
    "DTL5",
    "DTL6",
    "CCL1",
    "CCL2",
    "CCL3",
    "CCL4",
    "SCLMed",
    "SCLHigh",
    "HEBT1",
    "HEBT2",
]


class SNS_LINAC:
    def __init__(
        self, 
        xml_filename: str = None,
        sequence_start: str = "MEBT",
        sequence_stop: str = "HEBT2",
        max_drift: float = 0.010, 
        rf_frequency: float = 402.5e+06,
        verbose: bool = True,
    ) -> None:
        self.path = pathlib.Path(__file__)
        
        self.xml_filename = xml_filename
        if self.xml_filename is None:
            self.xml_filename = os.path.join(self.path.parent, "data/sns_linac.xml")

        
        self.rf_frequency = rf_frequency

        index_start = SEQUENCES.index(sequence_start)
        index_stop = SEQUENCES.index(sequence_stop)
        self.sequences = SEQUENCES[index_start : index_stop + 1]
        
        sns_linac_factory = SNS_LinacLatticeFactory()
        if max_drift:
            sns_linac_factory.setMaxDriftLength(max_drift)
    
        self.lattice = sns_linac_factory.getLinacAccLattice(self.sequences, self.xml_filename)

        self.aperture_nodes = []
        self.spacecharge_nodes = []

        if verbose:
            print("Initialized lattice.")
            print("XML filename = {}".format(xml_filename))
            print("lattice length = {:.3f} [m])".format(self.lattice.getLength()))
            
    def save_node_positions(self, filename: str = "lattice_nodes.txt") -> None:
        file = open(filename, "w")
        file.write("node position length\n")
        for node in self.lattice.getNodes():
            file.write("{} {} {}\n".format(node.getName(), node.getPosition(), node.getLength()))
        file.close()

    def save_lattice_structure(self, filename: str = "lattice_structure.txt"):
        file = open(filename, "w")
        file.write(self.lattice.structureToText())
        file.close()



        