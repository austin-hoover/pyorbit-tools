"""SNS linac model, containing boilerplate code to set up the lattice."""
import math
import os
import pathlib
from typing import Any
from typing import Callable

import numpy as np
import scipy.optimize

from orbit.core.bunch import Bunch
from orbit.core.linac import BaseRfGap
from orbit.core.linac import MatrixRfGap
from orbit.core.linac import RfGapTTF
from orbit.core.spacecharge import SpaceChargeCalc3D
from orbit.core.spacecharge import SpaceChargeCalcUnifEllipse
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice import AxisFieldRF_Gap
from orbit.py_linac.lattice import AxisField_and_Quad_RF_Gap
from orbit.py_linac.lattice import BaseRF_Gap
from orbit.py_linac.lattice import Bend
from orbit.py_linac.lattice import Drift
from orbit.py_linac.lattice import LinacApertureNode
from orbit.py_linac.lattice import LinacEnergyApertureNode
from orbit.py_linac.lattice import LinacPhaseApertureNode
from orbit.py_linac.lattice import OverlappingQuadsNode 
from orbit.py_linac.lattice import Quad
from orbit.py_linac.lattice_modifications import Add_drift_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_rfgap_apertures_to_lattice
from orbit.py_linac.lattice_modifications import AddMEBTChopperPlatesAperturesToSNS_Lattice
from orbit.py_linac.lattice_modifications import AddScrapersAperturesToLattice
from orbit.py_linac.lattice_modifications import GetLostDistributionArr
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_to_AxisField_Nodes
from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes
from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.overlapping_fields import SNS_EngeFunctionFactory
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.space_charge.sc3d import setUniformEllipsesSCAccNodes

from orbitsim.linac import add_aperture_nodes_to_classes
from orbitsim.linac import add_aperture_nodes_to_drifts
from orbitsim.linac import make_energy_aperture_node
from orbitsim.linac import make_phase_aperture_node


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
        rf_freq: float = 402.5e+06,
        verbose: bool = True,
    ) -> None:
        self.path = pathlib.Path(__file__)
        self.xml_filename = xml_filename
        if self.xml_filename is None:
            self.xml_filename = os.path.join(self.path.parent, "data/sns_linac.xml")

        index_start = SEQUENCES.index(sequence_start)
        index_stop = SEQUENCES.index(sequence_stop)
        self.sequences = SEQUENCES[index_start : index_stop + 1]
        
        sns_linac_factory = SNS_LinacLatticeFactory()
        if max_drift:
            sns_linac_factory.setMaxDriftLength(max_drift)
    
        self.lattice = sns_linac_factory.getLinacAccLattice(self.sequences, self.xml_filename)

        self.rf_freq = rf_freq
        self.verbose = verbose
        self.aperture_nodes = []
        self.sc_nodes = []

        if verbose:
            print("Initialized lattice")
            print("xml file = {}".format(self.xml_filename))
            print("lattice length = {:.3f} [m])".format(self.lattice.getLength()))

    def set_rf_gap_model(self, name: str = "ttf") -> None:            
        rf_gap_model_constructors = {
            "base": BaseRfGap,
            "matrix": MatrixRfGap,
            "ttf": RfGapTTF,
        }
        rf_gap_model_constructor = rf_gap_model_constructors[name]
        for rf_gap_node in self.lattice.getRF_Gaps():
            rf_gap_model = rf_gap_model_constructor()
            rf_gap_node.setCppGapModel(rf_gap_model)

    def add_aperture_nodes(
        self, 
        scrape_x: float = 0.042, 
        scrape_y: float = 0.042
    ) -> list[AccNode]:
        """Add aperture nodes to quads, rf gaps, mebt choppers, and scrapers."""
        aperture_nodes = Add_quad_apertures_to_lattice(self.lattice)
        aperture_nodes = Add_rfgap_apertures_to_lattice(self.lattice, aperture_nodes)
        aperture_nodes = AddMEBTChopperPlatesAperturesToSNS_Lattice(self.lattice, aperture_nodes)        
        aperture_nodes = AddScrapersAperturesToLattice(
            self.lattice, "MEBT_Diag:H_SCRP", scrape_x, scrape_y, aperture_nodes
        )    
        aperture_nodes = AddScrapersAperturesToLattice(
            self.lattice, "MEBT_Diag:V_SCRP", scrape_x, scrape_y, aperture_nodes
        )
        self.aperture_nodes.extend(aperture_nodes)
        return aperture_nodes

    def add_aperture_nodes_to_drifts(
        self, 
        start: float = 0.0, 
        stop: float = None, 
        step: float = 1.0, 
        radius: float = 0.042, 
    ) -> list[AccNode]:
        """Add circular apertures in drifts between start and stop position."""
        if stop is None:
            stop = self.lattice.getLength()
        aperture_nodes = Add_drift_apertures_to_lattice(self.lattice, start, stop, step, 2.0 * radius)
        self.aperture_nodes.extend(aperture_nodes)
        return aperture_nodes

    def add_phase_aperture_nodes(
        self,
        phase_min: float = -180.0,  # [deg]
        phase_max: float = +180.0,  # [deg]
        classes: list = None,
        drifts: bool = False,
        drift_start: float = 0.0,
        drift_stop: float = None,
        drift_step: float = 1.0,
        nametag: str = "phase_aprt",
    ) -> list[AccNode]:
        """Add longitudinal phase aperture nodes."""
        if classes is None:
            classes = [
                BaseRF_Gap, 
                AxisFieldRF_Gap, 
                AxisField_and_Quad_RF_Gap,
                Quad, 
                OverlappingQuadsNode,
            ]
            
        node_constructor = make_phase_aperture_node
        node_constructor_kws = {
            "phase_min": phase_min,
            "phase_max": phase_max,
            "rf_freq": self.rf_freq,
        }          
        if classes:
            aperture_nodes = add_aperture_nodes_to_classes(
                lattice=self.lattice,
                classes=classes,
                nametag=nametag,
                node_constructor=node_constructor,
                node_constructor_kws=node_constructor_kws,
            )
            self.aperture_nodes.extend(aperture_nodes)            
        if drifts:
            aperture_nodes = add_aperture_nodes_to_drifts(
                lattice=self.lattice,
                start=drift_start,
                stop=drift_stop,
                step=drift_step,
                nametag=nametag,
                node_constructor=node_constructor,
                node_constructor_kws=node_constructor_kws,
            )
            self.aperture_nodes.extend(aperture_nodes)                
        return aperture_nodes

    def add_energy_aperture_nodes(
        self,
        energy_min: float = -1.000,  # [GeV]
        energy_max: float = -1.000,  # [GeV]
        classes: list = None,
        drifts: bool = False,
        drift_start: float = 0.0,
        drift_stop: float = None,
        drift_step: float = 1.0,
        nametag: str = "phase_aprt",
    ) -> list[AccNode]:
        """Add longitudinal phase aperture nodes."""
        if classes is None:
            classes = [
                BaseRF_Gap, 
                AxisFieldRF_Gap, 
                AxisField_and_Quad_RF_Gap,
                Quad, 
                OverlappingQuadsNode,
            ]
            
        node_constructor = make_energy_aperture_node
        node_constructor_kws = {
            "energy_min": energy_min,
            "energy_max": energy_max,
        }          
        if classes:
            aperture_nodes = add_aperture_nodes_to_classes(
                lattice=self.lattice,
                classes=classes,
                nametag=nametag,
                node_constructor=node_constructor,
                node_constructor_kws=node_constructor_kws,
            )
            self.aperture_nodes.extend(aperture_nodes)            
        if drifts:
            aperture_nodes = add_aperture_nodes_to_drifts(
                lattice=self.lattice,
                start=drift_start,
                stop=drift_stop,
                step=drift_step,
                nametag=nametag,
                node_constructor=node_constructor,
                node_constructor_kws=node_constructor_kws,
            )
            self.aperture_nodes.extend(aperture_nodes)                
        return aperture_nodes

    def add_sc_nodes(
        self,
        solver: str = "fft",
        gridx: int = 64,
        gridy: int = 64,
        gridz: int = 64,
        path_length_min: float = 0.010,
        n_ellipsoids: int = 5,
    ) -> list[AccNode]:
        sc_nodes = []
        if solver == "fft":
            sc_calc = SpaceChargeCalc3D(gridx, gridy, gridz)
            sc_nodes = setSC3DAccNodes(self.lattice, path_length_min, sc_calc)
        elif solver == "ellipsoid":
            sc_calc = SpaceChargeCalcUnifEllipse(n_ellipsoids)
            sc_nodes = setUniformEllipsesSCAccNodes(self.lattice, path_length_min, sc_calc)
        else:
            raise ValueError(f"Invalid spacecharge solver {solver}")

        if self.verbose:
            lengths = [node.getLengthOfSC() for node in sc_nodes]
            min_length = min(min(lengths), self.lattice.getLength())
            max_length = max(max(lengths), 0.0)
            
            print(f"Added {len(sc_nodes)} space charge nodes (solver={solver})")
            print(f"min sc node length = {min_length}".format(min_length))
            print(f"max sc node length = {min_length}".format(max_length))
            
        self.sc_nodes = sc_nodes
        return sc_nodes

    def set_overlapping_rf_and_quad_fields(
        self, 
        sequences: list[str] = None, 
        z_step: float = 0.002,
        cav_names: list[str] = None,
        fields_dir: str = None,
        use_longitudinal_quad_field: bool = True,
    ) -> None:
        """Replace overlapping quad/rf nodes in specified sequences."""
        if fields_dir is None:
            fields_dir = os.path.join(self.path.parent, "data/sns_rf_fields/")
            
        if sequences is None:
            sequences = self.sequences
        sequences = sequences.copy()
        sequences = [seq for seq in sequences if seq not in ["HEBT1", "HEBT2"]]

        if cav_names is None:
            cav_names = []
            
        # Replace hard-edge quads with soft-edge quads; replace zero-length RF gap models
        # with field-on-axis RF gap models. Can be used for any sequences, no limitations.
        Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes(
            self.lattice, z_step, fields_dir, sequences, cav_names, SNS_EngeFunctionFactory
        )

        # Add tracking through the longitudinal field component of the quad. The
        # longitudinal component is nonzero only for the distributed magnetic field
        # of the quad. 
        for node in self.lattice.getNodes():
            if (isinstance(node, OverlappingQuadsNode) or isinstance(node, AxisField_and_Quad_RF_Gap)):
                node.setUseLongitudinalFieldOfQuad(use_longitudinal_quad_field)
