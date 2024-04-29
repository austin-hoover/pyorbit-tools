from typing import Any
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.optimize

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.foil import Foil
from orbit.core.impedances import LImpedance
from orbit.core.impedances import TImpedance
from orbit.core.spacecharge import LSpaceChargeCalc
from orbit.core.spacecharge import Boundary2D
from orbit.core.spacecharge import SpaceChargeCalc2p5D
from orbit.core.spacecharge import SpaceChargeCalcSliceBySlice2D
from orbit.aperture import CircleApertureNode
from orbit.aperture import EllipseApertureNode
from orbit.aperture import RectangleApertureNode
from orbit.aperture import TeapotApertureNode
from orbit.aperture.ApertureLatticeModifications import addTeapotApertureNode
from orbit.bumps import bumps
from orbit.bumps import BumpLatticeModifications
from orbit.bumps import TDTeapotSimpleBumpNode
from orbit.bumps import TeapotBumpNode
from orbit.bumps import TeapotSimpleBumpNode
from orbit.collimation import addTeapotCollimatorNode
from orbit.collimation import TeapotCollimatorNode
from orbit.diagnostics import addTeapotDiagnosticsNode
from orbit.diagnostics import TeapotMomentsNode
from orbit.diagnostics import TeapotStatLatsNode
from orbit.diagnostics import TeapotTuneAnalysisNode
from orbit.foils import addTeapotFoilNode
from orbit.foils import TeapotFoilNode
from orbit.impedances import addImpedanceNode
from orbit.impedances import BetFreqDep_LImpedance_Node
from orbit.impedances import BetFreqDep_TImpedance_Node
from orbit.impedances import FreqDep_LImpedance_Node
from orbit.impedances import FreqDep_TImpedance_Node
from orbit.impedances import LImpedance_Node
from orbit.impedances import TImpedance_Node
from orbit.injection import addTeapotInjectionNode
from orbit.injection import InjectParts
from orbit.injection import TeapotInjectionNode
from orbit.injection.joho import JohoTransverse
from orbit.injection.joho import JohoLongitudinal
from orbit.injection.distributions import SNSESpreadDist
from orbit.injection.distributions import UniformLongDist
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNode
from orbit.lattice import AccLattice
from orbit.rf_cavities import RFNode
from orbit.rf_cavities import RFLatticeModifications
from orbit.space_charge import sc2p5d
from orbit.space_charge import sc2dslicebyslice
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode
from orbit.space_charge.sc1d import SC1D_AccNode
from orbit.space_charge.sc2dslicebyslice.scLatticeModifications import setSC2DSliceBySliceAccNodes
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_Ring
from orbit.teapot import teapot
from orbit.time_dep.waveform import ConstantWaveform
from orbit.utils import consts
from orbit.utils.consts import mass_proton
from orbit.utils.consts import speed_of_light


# # Maximum injection kicker angles at 1.0 GeV [mrad]
# MIN_KICKER_ANGLES = 1.15 * np.array([0.0, 0.0, -7.13, -7.13, -7.13, -7.13, 0.0, 0.0])
# MAX_KICKER_ANGLES = 1.15 * np.array([12.84, 12.84, 0.0, 0.0, 0.0, 0.0, 12.84, 12.84])

# ## Maximum injection-region dipole corrector angle [mrad]
# MAX_VCORRECTOR_ANGLE = 1.5



class Ring(TEAPOT_Ring):
    def __init__(self) -> None:
        TEAPOT_Ring.__init__(self)
        
        self.bunch = None
        self.lostbunch = None
        self.params_dict = None
                       
        self.aperture_nodes = []
        self.collimator_nodes = []
        self.injection_node = None
        self.transverse_impedance_nodes = []
        self.transverse_spacecharge_nodes = []
        self.longitudinal_impedance_nodes = []
        self.longitudinal_spacecharge_nodes = []
        self.rf_nodes = []

    def read_lattice_file(self, path: str, sequence: str, kind: str = "madx") -> AccLattice:
        if not os.path.exists(path):
            raise FileNotFoundError 
    
        if kind == "madx":
            self.readMADX(path, sequence)
        elif kind == "mad":
            self.readMAD(path, sequence)
        else:
            raise ValueError(f"Invalid kind {kind}")
    
    def set_bunch(self, bunch: Bunch, lostbunch: Bunch = None, params_dict: dict = None) -> None:
        self.bunch = bunch
        
        self.lostbunch = lostbunch
        if self.lostbunch is None:
            self.lostbunch = Bunch()
            
        self.params_dict = params_dict
        if self.params_dict is None:
            self.params_dict = {"bunch": bunch, "lostbunch": lostbunch}
        
    def set_fringe_fields(self, setting: bool) -> None:
        self.fringe = setting
        for node in self.getNodes():
            if hasattr(node, "setUsageFringeFieldIN") and hasattr(node, "setUsageFringeFieldOUT"):
                node.setUsageFringeFieldIN(setting)
                node.setUsageFringeFieldOUT(setting)
            else:
                warnings.warnd(f"No fringe field for node {node.getName()}")
                
    def get_twiss(self, mass: float, kin_energy: float) -> dict[str, np.ndarray]:
        bunch = Bunch()
        bunch.mass(mass)
        bunch.getSyncParticle().kinEnergy(kin_energy)
        
        matrix_lattice = TEAPOT_MATRIX_Lattice(self, bunch)
        (pos_nu_x, pos_alpha_x, pos_beta_x) = matrix_lattice.getRingTwissDataX()
        (pos_nu_y, pos_alpha_y, pos_beta_y) = matrix_lattice.getRingTwissDataY()
        
        twiss = dict()
        twiss["pos"] = np.array(pos_nu_x)[:, 0]
        twiss["nu_x"] = np.array(pos_nu_x)[:, 1]
        twiss["nu_y"] = np.array(pos_nu_y)[:, 1]
        twiss["alpha_x"] = np.array(pos_alpha_x)[:, 1]
        twiss["alpha_y"] = np.array(pos_alpha_y)[:, 1]
        twiss["beta_x"] = np.array(pos_beta_x)[:, 1]
        twiss["beta_y"] = np.array(pos_beta_y)[:, 1]
        return twiss
    
    def get_dispersion(self, mass: float, kin_energy: float) -> dict[str, np.ndarray]:
        bunch = Bunch()
        bunch.mass(mass)
        bunch.getSyncParticle().kinEnergy(kin_energy)
        
        matrix_lattice = TEAPOT_MATRIX_Lattice(self, bunch)
        (pos_disp_x, pos_dispp_x) = matrix_lattice.getRingDispersionDataX()
        (pos_disp_y, pos_dispp_y) = matrix_lattice.getRingDispersionDataY()
        
        dispersion = dict()
        dispersion["s"] = np.array(pos_disp_x)[:, 0]
        dispersion["disp_x"] = np.array(pos_disp_x)[:, 1]
        dispersion["disp_y"] = np.array(pos_disp_y)[:, 1]
        dispersion["dispp_x"] = np.array(pos_dispp_x)[:, 1]
        dispersion["dispp_y"] = np.array(pos_dispp_y)[:, 1]
        return dispersion

    

class SNS_RING(Ring):
    def __init__(
        self, 
        x_inj: float = 0.0486,  
        y_inj: float = 0.0460,
        solenoid_names: list[str] = ["scbdsol_c13a", "scbdsol_c13b"]
    ) -> None:
        Ring.__init__(self)

        self.x_inj = x_inj
        self.y_inj = y_inj
        
        self.foil_node = None
        self.solenoid_nodes = []


    def find_solenoids(self) -> list[AccNode]:
        self.solenoid_nodes = []
        for node in self.getNodes():
            if node.getName() in solenoid_names:
                self.solenoid_nodes.append(node)
        return self.solenoid_nodes
        
    def set_solenoid_strengths(self, B: float) -> None:
        self.solenoid_nodes = []
        for node in self.solenoid_nodes:
            node.setParam("B", B)

    def add_inj_node(
        self,
        n_parts: int,
        dist_x: Any,
        dist_y: Any,
        dist_z: Any,
        limits: list[tuple[float]],
        index: int = 0,
        n_parts_max: int = None,
    ) -> AccNode:           
        
        (xmin, xmax), (ymin, ymax) = limits
        boundary = [xmin, xmax, ymin, ymax]

        if n_parts_max is None:
            n_parts_max = -1
        
        self.injection_node = TeapotInjectionNode(
            n_parts, 
            self.bunch, 
            self.lostbunch, 
            boundary, 
            dist_x, 
            dist_y, 
            dist_z, 
            n_parts_max,
        )
        
        node = self.getNodes()[index]
        node.addChildNode(self.injection_node, AccNode.ENTRANCE)

        return self.injection_node
        
    # def add_foil_node(
    #     self,
    #     xmin=-0.0085,
    #     xmax=+0.0085,
    #     ymin=-0.0080,
    #     ymax=+0.1000,
    #     thickness=390.0,
    #     scatter="full",  # {"full", "simple"}
    # ):
    #     xmin += self.x_inj
    #     xmax += self.x_inj
    #     ymin += self.y_inj
    #     ymax += self.y_inj
    #     scatter_options = {
    #         "full": 0, 
    #         "simple": 1
    #     }
    #     scatter = scatter_options[scatter]
    #     foil_node = TeapotFoilNode(xmin, xmax, ymin, ymax, thickness)
    #     foil_node.setScatterChoice(scatter)
    #     start_node = self.getNodes()[0]
    #     start_node.addChildNode(foil_node, AccNode.ENTRANCE)
    #     return foil_node

    # def add_longitudinal_impedance_node(
    #     self,
    #     n_macros_min=1000,
    #     n_bins=128,
    #     position=124.0,
    #     zl_ekicker_filename="",
    #     zl_rf_filename="",
    # ):
    #     """Sets impedance from file."""
    #     if not (zl_ekicker_filename or zl_rf_filename):
    #         raise ValueError()
        
    #     ZL = dict()
    #     for key, filename in zip(["ekicker", "rf"], [zl_ekicker_filename, zl_rf_filename]):
    #         file = open(filename, "r")
    #         ZL[key] = []
    #         for line in file:
    #             if line.startswith("#"):
    #                 continue
    #             real, imag = map(float, line.split(" "))
    #             ZL[key].append(complex(real, imag))
    #         file.close()
                
    #     Z = []
    #     for zk, zrf in zip(ZL["ekicker"], ZL["rf"]):
    #         zreal = (zk.real / 1.75) + zrf.real
    #         zimag = (zk.imag / 1.75) + zrf.imag
    #         Z.append(complex(zreal, zimag))
    #     impedance_node = LImpedance_Node(self.getLength(), n_macros_min, n_bins)
    #     impedance_node.assignImpedance(Z)
    #     addImpedanceNode(self, position, impedance_node)
    #     return impedance_node
        
    # def add_transverse_impedance_node(
    #     self,
    #     n_macros_min=1000,
    #     n_bins=64,
    #     use_x=0,
    #     use_y=1,
    #     position=124.0,
    #     alpha_x=0.0,
    #     alpha_y=-0.004,
    #     beta_x=10.191,
    #     beta_y=10.447,
    #     q_x=6.21991,
    #     q_y=6.20936,
    #     filename="",
    # ):
    #     if not filename:
    #         raise ValueError()
            
    #     INDEX, ZP, ZM = [], [], []
    #     file = open(filename, "r")
    #     for line in file:
    #         if line.startswith("#"):
    #             continue
    #         m, ZPR, ZPI, ZMR, ZMI = map(float, line.split())
    #         ZP.append(complex(ZPR, -ZPI))
    #         ZM.append(complex(ZMR, -ZMI))
    #         INDEX.append(int(m))
    #     file.close()

    #     impedance_node = TImpedance_Node(self.getLength(), n_macros_min, n_bins, use_x, use_y)
    #     impedance_node.assignLatFuncs(q_x, alpha_x, beta_x, q_y, alpha_y, beta_y)
    #     if use_x:
    #         impedance_node.assignImpedance("X", ZP, ZM)
    #     if use_y:
    #         impedance_node.assignImpedance("Y", ZP, ZM)
    #     addImpedanceNode(self, position, impedance_node)
    #     return impedance_node
        
    # def add_longitudinal_space_charge_node(
    #     self, 
    #     b_a=(10.0 / 3.0),
    #     n_macros_min=1000,
    #     use=1,
    #     n_bins=64,
    #     position=124.0,
    # ):
    #     sc_node = SC1D_AccNode(b_a, self.getLength(), n_macros_min, use, n_bins)
    #     Z = [complex(0.0, 0.0) for _ in range(n_bins // 2)]
    #     sc_node.assignImpedance(Z)
    #     addLongitudinalSpaceChargeNode(self, position, sc_node)
    #     return sc_node
        
    # def add_transverse_space_charge_nodes(
    #     self,
    #     n_macros_min=1000,
    #     size_x=128,
    #     size_y=128,
    #     size_z=64,
    #     path_length_min=1.0e-8,
    #     n_boundary_points=128,
    #     n_free_space_modes=32,
    #     radius=0.220,
    #     kind="slicebyslice",
    # ):
    #     boundary = Boundary2D(n_boundary_points, n_free_space_modes, "Circle", radius, radius)
    #     if kind == "2p5d":
    #         sc_nodes = setSC2p5DAccNodes(
    #             self, 
    #             path_length_min, 
    #             SpaceChargeCalc2p5D(size_x, size_y, size_z), 
    #             boundary,
    #         )
    #     elif kind == "slicebyslice":
    #         sc_nodes = setSC2DSliceBySliceAccNodes(
    #             self, 
    #             path_length_min, 
    #             SpaceChargeCalcSliceBySlice2D(size_x, size_y, size_z), 
    #             boundary,
    #         )
    #     else:
    #         raise ValueError("Invalid kind.")
    #     return sc_nodes
            
    # def add_tune_diagnostics_node(
    #     self,
    #     filename="tunes.dat",
    #     position=51.1921,
    #     alpha_x=-1.78574, 
    #     alpha_y=0.538244,
    #     beta_x=9.19025, 
    #     beta_y=8.66549, 
    #     eta_x=-0.000143012, 
    #     etap_x=-2.26233e-05, 
    # ):        
    #     tune_node = TeapotTuneAnalysisNode(filename)
    #     tune_node.assignTwiss(beta_x, alpha_x, eta_x, etap_x, beta_y, alpha_y)
    #     addTeapotDiagnosticsNode(self, position, tune_node)
    #     return tune_node
    
    # def add_moments_diagnostics_node(self, filename="moments.dat", order=4, position=0.0):
    #     moments_node = TeapotMomentsNode(filename, order)
    #     start_node = self.getNodes()[0]
    #     start_node.addChildNode(moments_node, AccNode.ENTRANCE)
    #     return moments_node

    # def add_inj_chicane_aperture_displacement_nodes(self):
    #     """Add apertures and displacements for injection chicane."""
    #     xcenter = 0.100
    #     xb10m = 0.0
    #     xb11m = 0.09677
    #     xb12m = 0.08899
    #     xb13m = 0.08484

    #     ycenter = 0.023

    #     bumpwave = ConstantWaveform(self.bunch.getSyncParticle(), self.getLength(), 1.0)
        
    #     xb10i = 0.0
    #     apb10xi = -xb10i
    #     apb10yi = ycenter
    #     rb10i = 0.1095375
    #     appb10i = CircleApertureNode(rb10i, 244.812, apb10xi, apb10yi, name="b10i")

    #     xb10f = 0.022683
    #     apb10xf = -xb10f
    #     apb10yf = ycenter
    #     rb10f = 0.1095375
    #     appb10f = CircleApertureNode(rb10f, 245.893, apb10xf, apb10yf, name="b10f")

    #     mag10x = (xb10i + xb10f) / 2.0 - xb10m
    #     cdb10i = TDTeapotSimpleBumpNode(self.bunch, +mag10x, 0.0, -ycenter, 0.0, bumpwave, "mag10bumpi")
    #     cdb10f = TDTeapotSimpleBumpNode(self.bunch, -mag10x, 0.0, +ycenter, 0.0, bumpwave, "mag10bumpf")

    #     xb11i = 0.074468
    #     apb11xi = xcenter - xb11i
    #     apb11yi = ycenter
    #     rb11i = 0.1095375
    #     appb11i = CircleApertureNode(rb11i, 247.265, apb11xi, apb11yi, name="b11i")

    #     xfoil = 0.099655
    #     apfoilx = xcenter - xfoil
    #     apfoily = ycenter
    #     rfoil = 0.1095375
    #     appfoil = CircleApertureNode(rfoil, 248.009, apfoilx, apfoily, name="bfoil")

    #     mag11ax = (xb11i + xfoil) / 2.0 - xb11m
    #     cdb11i = TDTeapotSimpleBumpNode(self.bunch, +mag11ax, 0.0, -ycenter, 0.0, bumpwave, "mag11bumpi")
    #     cdfoila = TDTeapotSimpleBumpNode(self.bunch, -mag11ax, 0.0, +ycenter, 0.0, bumpwave, "foilbumpa")

    #     xb11f = 0.098699
    #     apb11xf = xcenter - xb11f
    #     apb11yf = ycenter
    #     rb11f = 0.1095375
    #     appb11f = CircleApertureNode(rb11f, 0.195877, apb11xf, apb11yf, name="b11f")

    #     mag11bx = (xfoil + xb11f) / 2.0 - xb11m
    #     cdfoilb = TDTeapotSimpleBumpNode(self.bunch, +mag11bx, 0.0, -ycenter, 0.0, bumpwave, "foilbumpb")
    #     cdb11f = TDTeapotSimpleBumpNode(self.bunch, -mag11bx, 0.0, +ycenter, 0.0, bumpwave, "mag11bumpf")

    #     xb12i = 0.093551
    #     apb12xi = xcenter - xb12i
    #     apb12yi = ycenter
    #     rb12i = 0.1095375
    #     appb12i = CircleApertureNode(rb12i, 1.08593, apb12xi, apb12yi, name="b12i")

    #     xb12f = 0.05318
    #     apb12xf = xcenter - xb12f
    #     apb12yf = ycenter
    #     rb12f = 0.1174750
    #     appb12f = CircleApertureNode(rb12f, 1.99425, apb12xf, apb12yf, name="b12f")

    #     mag12x = (xb12i + xb12f) / 2.0 - xb12m
    #     cdb12i = TDTeapotSimpleBumpNode(self.bunch, +mag12x, 0.0, -ycenter, 0.0, bumpwave, "mag12bumpi")
    #     cdb12f = TDTeapotSimpleBumpNode(self.bunch, -mag12x, 0.0, +ycenter, 0.0, bumpwave, "mag12bumpf")

    #     xb13i = 0.020774
    #     apb13xi = xcenter - xb13i
    #     apb13yi = ycenter
    #     h13xi = 0.1913
    #     v13xi = 0.1016
    #     appb13i = RectangleApertureNode(h13xi, v13xi, 3.11512, apb13xi, apb13yi, name="b13i")

    #     xb13f = 0.0
    #     apb13xf = xcenter - xb13f
    #     apb13yf = ycenter
    #     h13xf = 0.1913
    #     v13xf = 0.1016
    #     appb13f = RectangleApertureNode(h13xf, v13xf, 4.02536, apb13xf, apb13yf, name="b13f")

    #     mag13x = (xb13i + xb13f) / 2.0 - xb13m
    #     cdb13i = TDTeapotSimpleBumpNode(self.bunch, +mag13x, 0.0, -ycenter, 0.0, bumpwave, "mag13bumpi")
    #     cdb13f = TDTeapotSimpleBumpNode(self.bunch, -mag13x, 0.0, +ycenter, 0.0, bumpwave, "mag13bumpf")

    #     node_names = ["dh_a10", "dh_a11", "dh_a12", "dh_a13"]
    #     nodes = {name: self.getNodeForName(name) for name in node_names}
    #     nodes["dh_a10"].addChildNode(appb10i, AccNode.ENTRANCE)
    #     nodes["dh_a10"].addChildNode(cdb10i, AccNode.ENTRANCE)
    #     nodes["dh_a10"].addChildNode(cdb10f, AccNode.EXIT)
    #     nodes["dh_a10"].addChildNode(appb10f, AccNode.EXIT)
    #     nodes["dh_a11"].addChildNode(appb11i, AccNode.ENTRANCE)
    #     nodes["dh_a11"].addChildNode(cdb11i, AccNode.ENTRANCE)
    #     nodes["dh_a11"].addChildNode(cdfoila, AccNode.EXIT)
    #     nodes["dh_a11"].addChildNode(appfoil, AccNode.EXIT)
    #     nodes["dh_a11"].addChildNode(cdfoilb, AccNode.ENTRANCE)
    #     nodes["dh_a11"].addChildNode(cdb11f, AccNode.EXIT)
    #     nodes["dh_a11"].addChildNode(appb11f, AccNode.EXIT)
    #     nodes["dh_a12"].addChildNode(appb12i, AccNode.ENTRANCE)
    #     nodes["dh_a12"].addChildNode(cdb12i, AccNode.ENTRANCE)
    #     nodes["dh_a12"].addChildNode(cdb12f, AccNode.EXIT)
    #     nodes["dh_a12"].addChildNode(appb12f, AccNode.EXIT)
    #     nodes["dh_a13"].addChildNode(appb13i, AccNode.ENTRANCE)
    #     nodes["dh_a13"].addChildNode(cdb13i, AccNode.ENTRANCE)
    #     nodes["dh_a13"].addChildNode(cdb13f, AccNode.EXIT)
    #     nodes["dh_a13"].addChildNode(appb13f, AccNode.EXIT)
                            
    # def add_aperture_node_circle(self, rx=0.0, position=0.5, x=0.0, y=0.0, name="aperture_node"):
    #     aperture_node = CircleApertureNode(rx, position, x, y, name)
    #     addTeapotApertureNode(self, position, aperture_node)        
    #     return aperture_node
    
    # def add_aperture_node_ellipse(self, rx=0.0, ry=0.0, position=0.5, x=0.0, y=0.0, name="aperture_node"):
    #     aperture_node = EllipseApertureNode(rx, ry, position, x, y, name)
    #     addTeapotApertureNode(self, position, aperture_node)        
    #     return aperture_node
    
    # def add_aperture_node_rectangle(self, rx=0.0, ry=0.0, position=0.5, x=0.0, y=0.0, name="aperture_node"):
    #     aperture_node = RectangleApertureNode(rx, ry, position, x, y, name)
    #     addTeapotApertureNode(self, position, aperture_node)        
    #     return aperture_node
                
    # def add_aperture_nodes(self):
    #     """Add apertures throughout the ring.

    #     This does not work with the standard MADX output file. The parent nodes are currently 
    #     selected by index; they need to be selected by name.
    #     """
    #     if True:
    #         warnings.warn("Cannot add aperture nodes.")
    #         return
            
    
    #     nodes = self.getNodes()

    #     a115p8 = 0.1158
    #     b078p7 = 0.0787
    #     a080p0 = 0.0800
    #     b048p0 = 0.0480

    #     r062 = 0.0625
    #     r100 = 0.1000
    #     r120 = 0.1200
    #     r125 = 0.1250
    #     r130 = 0.1300
    #     r140 = 0.1400

    #     app06200 = CircleApertureNode(r062, 58.21790, 0.0, 0.0, name="s1")
    #     app06201 = CircleApertureNode(r062, 65.87790, 0.0, 0.0, name="s2")

    #     app10000 = CircleApertureNode(r100, 10.54740, 0.0, 0.0, name="bp100")
    #     app10001 = CircleApertureNode(r100, 10.97540, 0.0, 0.0, name="bp100")
    #     app10002 = CircleApertureNode(r100, 13.57190, 0.0, 0.0, name="bp100")
    #     app10003 = CircleApertureNode(r100, 15.39140, 0.0, 0.0, name="21cmquad")
    #     app10004 = CircleApertureNode(r100, 19.39150, 0.0, 0.0, name="21cmquad")
    #     app10005 = CircleApertureNode(r100, 23.39170, 0.0, 0.0, name="21cmquad")
    #     app10006 = CircleApertureNode(r100, 31.39210, 0.0, 0.0, name="21cmquad")
    #     app10007 = CircleApertureNode(r100, 39.39250, 0.0, 0.0, name="21cmquad")
    #     app10008 = CircleApertureNode(r100, 43.39270, 0.0, 0.0, name="21cmquad")
    #     app10009 = CircleApertureNode(r100, 47.39290, 0.0, 0.0, name="21cmquad")
    #     app10010 = CircleApertureNode(r100, 48.86630, 0.0, 0.0, name="bp100")
    #     app10011 = CircleApertureNode(r100, 50.37710, 0.0, 0.0, name="P1a")
    #     app10012 = CircleApertureNode(r100, 51.19660, 0.0, 0.0, name="scr1c")
    #     app10013 = CircleApertureNode(r100, 51.24100, 0.0, 0.0, name="scr2c")
    #     app10014 = CircleApertureNode(r100, 51.39470, 0.0, 0.0, name="scr3c")
    #     app10015 = CircleApertureNode(r100, 51.43910, 0.0, 0.0, name="scr4c")
    #     app10016 = CircleApertureNode(r100, 51.62280, 0.0, 0.0, name="p2shield")
    #     app10017 = CircleApertureNode(r100, 53.74280, 0.0, 0.0, name="p2shield")
    #     app10018 = CircleApertureNode(r100, 54.75640, 0.0, 0.0, name="bp100")
    #     app10019 = CircleApertureNode(r100, 71.16320, 0.0, 0.0, name="bp100")
    #     app10020 = CircleApertureNode(r100, 73.35170, 0.0, 0.0, name="bp100")
    #     app10021 = CircleApertureNode(r100, 75.60170, 0.0, 0.0, name="bp100")
    #     app10022 = CircleApertureNode(r100, 76.79720, 0.0, 0.0, name="bp100")
    #     app10023 = CircleApertureNode(r100, 77.39290, 0.0, 0.0, name="21cmquad")
    #     app10024 = CircleApertureNode(r100, 81.39310, 0.0, 0.0, name="21cmquad")
    #     app10025 = CircleApertureNode(r100, 85.39330, 0.0, 0.0, name="21cmquad")
    #     app10026 = CircleApertureNode(r100, 93.39370, 0.0, 0.0, name="21cmquad")
    #     app10027 = CircleApertureNode(r100, 101.39400, 0.0, 0.0, name="21cmquad")
    #     app10028 = CircleApertureNode(r100, 105.39400, 0.0, 0.0, name="21cmquad")
    #     app10029 = CircleApertureNode(r100, 109.39400, 0.0, 0.0, name="21cmquad")
    #     app10030 = CircleApertureNode(r100, 110.49000, 0.0, 0.0, name="bp100")
    #     app10031 = CircleApertureNode(r100, 112.69100, 0.0, 0.0, name="bp100")
    #     app10032 = CircleApertureNode(r100, 114.82200, 0.0, 0.0, name="bp100")
    #     app10033 = CircleApertureNode(r100, 118.38600, 0.0, 0.0, name="bp100")
    #     app10034 = CircleApertureNode(r100, 120.37900, 0.0, 0.0, name="bp100")
    #     app10035 = CircleApertureNode(r100, 122.21700, 0.0, 0.0, name="bp100")
    #     app10036 = CircleApertureNode(r100, 124.64400, 0.0, 0.0, name="bp100")
    #     app10037 = CircleApertureNode(r100, 127.77400, 0.0, 0.0, name="bp100")
    #     app10038 = CircleApertureNode(r100, 132.53100, 0.0, 0.0, name="bp100")
    #     app10039 = CircleApertureNode(r100, 136.10400, 0.0, 0.0, name="bp100")
    #     app10040 = CircleApertureNode(r100, 138.79900, 0.0, 0.0, name="bp100")
    #     app10041 = CircleApertureNode(r100, 139.39400, 0.0, 0.0, name="21cmquad")
    #     app10042 = CircleApertureNode(r100, 143.39500, 0.0, 0.0, name="21cmquad")
    #     app10043 = CircleApertureNode(r100, 147.39500, 0.0, 0.0, name="21cmquad")
    #     app10044 = CircleApertureNode(r100, 155.39500, 0.0, 0.0, name="21cmquad")
    #     app10045 = CircleApertureNode(r100, 163.39600, 0.0, 0.0, name="21cmquad")
    #     app10046 = CircleApertureNode(r100, 167.39600, 0.0, 0.0, name="21cmquad")
    #     app10047 = CircleApertureNode(r100, 171.39600, 0.0, 0.0, name="21cmquad")
    #     app10048 = CircleApertureNode(r100, 173.70900, 0.0, 0.0, name="bp100")
    #     app10049 = CircleApertureNode(r100, 175.93900, 0.0, 0.0, name="bp100")
    #     app10050 = CircleApertureNode(r100, 180.38700, 0.0, 0.0, name="bp100")
    #     app10051 = CircleApertureNode(r100, 182.12700, 0.0, 0.0, name="bp100")
    #     app10052 = CircleApertureNode(r100, 184.27300, 0.0, 0.0, name="bp100")
    #     app10053 = CircleApertureNode(r100, 186.57100, 0.0, 0.0, name="bp100")
    #     app10054 = CircleApertureNode(r100, 188.86800, 0.0, 0.0, name="bp100")
    #     app10055 = CircleApertureNode(r100, 191.16500, 0.0, 0.0, name="bp100")
    #     app10056 = CircleApertureNode(r100, 194.53200, 0.0, 0.0, name="bp100")
    #     app10057 = CircleApertureNode(r100, 196.61400, 0.0, 0.0, name="bp100")
    #     app10058 = CircleApertureNode(r100, 199.47500, 0.0, 0.0, name="bp100")
    #     app10059 = CircleApertureNode(r100, 201.39600, 0.0, 0.0, name="21cmquad")
    #     app10060 = CircleApertureNode(r100, 205.39600, 0.0, 0.0, name="21cmquad")
    #     app10061 = CircleApertureNode(r100, 209.39600, 0.0, 0.0, name="21cmquad")
    #     app10062 = CircleApertureNode(r100, 217.39700, 0.0, 0.0, name="21cmquad")
    #     app10063 = CircleApertureNode(r100, 225.39700, 0.0, 0.0, name="21cmquad")
    #     app10064 = CircleApertureNode(r100, 229.39700, 0.0, 0.0, name="21cmquad")
    #     app10065 = CircleApertureNode(r100, 233.39700, 0.0, 0.0, name="21cmquad")
    #     app10066 = CircleApertureNode(r100, 234.87800, 0.0, 0.0, name="bp100")
    #     app10067 = CircleApertureNode(r100, 236.87700, 0.0, 0.0, name="bp100")
    #     app10068 = CircleApertureNode(r100, 238.74100, 0.0, 0.0, name="bp100")
    #     app10069 = CircleApertureNode(r100, 242.38900, 0.0, 0.0, name="bp100")

    #     app12000 = CircleApertureNode(r120, 6.89986, 0.0, 0.0, name="bp120")
    #     app12001 = CircleApertureNode(r120, 8.52786, 0.0, 0.0, name="bp120")
    #     app12002 = CircleApertureNode(r120, 57.20790, 0.0, 0.0, name="s1shield")
    #     app12003 = CircleApertureNode(r120, 59.40790, 0.0, 0.0, name="s1shield")
    #     app12004 = CircleApertureNode(r120, 64.86790, 0.0, 0.0, name="s2shield")
    #     app12005 = CircleApertureNode(r120, 67.06790, 0.0, 0.0, name="s2shield")
    #     app12006 = CircleApertureNode(r120, 116.75800, 0.0, 0.0, name="bp120")
    #     app12007 = CircleApertureNode(r120, 130.90300, 0.0, 0.0, name="bp120")
    #     app12008 = CircleApertureNode(r120, 178.75900, 0.0, 0.0, name="bp120")
    #     app12009 = CircleApertureNode(r120, 192.90400, 0.0, 0.0, name="bp120")
    #     app12010 = CircleApertureNode(r120, 240.76100, 0.0, 0.0, name="bp120")

    #     app12500 = CircleApertureNode(r125, 27.37140, 0.0, 0.0, name="26cmquad")
    #     app12501 = CircleApertureNode(r125, 35.37180, 0.0, 0.0, name="26cmquad")
    #     app12502 = CircleApertureNode(r125, 89.37300, 0.0, 0.0, name="26cmquad")
    #     app12503 = CircleApertureNode(r125, 97.37330, 0.0, 0.0, name="26cmquad")
    #     app12504 = CircleApertureNode(r125, 151.37400, 0.0, 0.0, name="26cmquad")
    #     app12505 = CircleApertureNode(r125, 159.37500, 0.0, 0.0, name="26cmquad")
    #     app12506 = CircleApertureNode(r125, 213.37600, 0.0, 0.0, name="26cmquad")
    #     app12507 = CircleApertureNode(r125, 221.37600, 0.0, 0.0, name="26cmquad")

    #     app13000 = CircleApertureNode(r130, 60.41790, 0.0, 0.0, name="bp130")
    #     app13001 = CircleApertureNode(r130, 64.42290, 0.0, 0.0, name="bp130")
    #     app13002 = CircleApertureNode(r130, 68.07790, 0.0, 0.0, name="bp130")
    #     app13003 = CircleApertureNode(r130, 68.90140, 0.0, 0.0, name="bp130")
    #     app13004 = CircleApertureNode(r130, 70.52940, 0.0, 0.0, name="bp130")

    #     app14000 = CircleApertureNode(r140, 7.43286, 0.0, 0.0, name="30cmquad")
    #     app14001 = CircleApertureNode(r140, 7.85486, 0.0, 0.0, name="30cmquad")
    #     app14002 = CircleApertureNode(r140, 55.42940, 0.0, 0.0, name="30cmquad")
    #     app14003 = CircleApertureNode(r140, 55.85140, 0.0, 0.0, name="30cmquad")
    #     app14004 = CircleApertureNode(r140, 56.38440, 0.0, 0.0, name="30cmquad")
    #     app14005 = CircleApertureNode(r140, 60.86290, 0.0, 0.0, name="bp140")
    #     app14006 = CircleApertureNode(r140, 62.64290, 0.0, 0.0, name="bp140")
    #     app14007 = CircleApertureNode(r140, 63.97790, 0.0, 0.0, name="bp140")
    #     app14008 = CircleApertureNode(r140, 69.43440, 0.0, 0.0, name="30cmquad")
    #     app14009 = CircleApertureNode(r140, 69.85640, 0.0, 0.0, name="30cmquad")
    #     app14010 = CircleApertureNode(r140, 117.43100, 0.0, 0.0, name="30cmquad")
    #     app14011 = CircleApertureNode(r140, 117.85300, 0.0, 0.0, name="30cmquad")
    #     app14012 = CircleApertureNode(r140, 131.43600, 0.0, 0.0, name="30cmquad")
    #     app14013 = CircleApertureNode(r140, 131.85800, 0.0, 0.0, name="30cmquad")
    #     app14014 = CircleApertureNode(r140, 179.43200, 0.0, 0.0, name="30cmquad")
    #     app14015 = CircleApertureNode(r140, 179.85400, 0.0, 0.0, name="30cmquad")
    #     app14016 = CircleApertureNode(r140, 193.43700, 0.0, 0.0, name="30cmquad")
    #     app14017 = CircleApertureNode(r140, 193.85900, 0.0, 0.0, name="30cmquad")
    #     app14018 = CircleApertureNode(r140, 241.43400, 0.0, 0.0, name="30cmquad")
    #     app14019 = CircleApertureNode(r140, 241.85600, 0.0, 0.0, name="30cmquad")

    #     appell00 = EllipseApertureNode(a115p8, b078p7, 16.9211, 0.0, 0.0, name="arcdipole")
    #     appell01 = EllipseApertureNode(a115p8, b078p7, 20.9213, 0.0, 0.0, name="arcdipole")
    #     appell02 = EllipseApertureNode(a115p8, b078p7, 24.9215, 0.0, 0.0, name="arcdipole")
    #     appell03 = EllipseApertureNode(a115p8, b078p7, 28.9217, 0.0, 0.0, name="arcdipole")
    #     appell04 = EllipseApertureNode(a115p8, b078p7, 32.9219, 0.0, 0.0, name="arcdipole")
    #     appell05 = EllipseApertureNode(a115p8, b078p7, 36.9221, 0.0, 0.0, name="arcdipole")
    #     appell06 = EllipseApertureNode(a115p8, b078p7, 40.9222, 0.0, 0.0, name="arcdipole")
    #     appell07 = EllipseApertureNode(a115p8, b078p7, 44.9224, 0.0, 0.0, name="arcdipole")
    #     appell08 = EllipseApertureNode(a080p0, b048p0, 51.9428, 0.0, 0.0, name="p2")
    #     appell09 = EllipseApertureNode(a115p8, b078p7, 78.9226, 0.0, 0.0, name="arcdipole")
    #     appell10 = EllipseApertureNode(a115p8, b078p7, 82.9228, 0.0, 0.0, name="arcdipole")
    #     appell11 = EllipseApertureNode(a115p8, b078p7, 86.9230, 0.0, 0.0, name="arcdipole")
    #     appell12 = EllipseApertureNode(a115p8, b078p7, 90.9232, 0.0, 0.0, name="arcdipole")
    #     appell13 = EllipseApertureNode(a115p8, b078p7, 94.9234, 0.0, 0.0, name="arcdipole")
    #     appell14 = EllipseApertureNode(a115p8, b078p7, 98.9236, 0.0, 0.0, name="arcdipole")
    #     appell15 = EllipseApertureNode(a115p8, b078p7, 102.9240, 0.0, 0.0, name="arcdipole")
    #     appell16 = EllipseApertureNode(a115p8, b078p7, 106.9240, 0.0, 0.0, name="arcdipole")
    #     appell17 = EllipseApertureNode(a115p8, b078p7, 140.9240, 0.0, 0.0, name="arcdipole")
    #     appell18 = EllipseApertureNode(a115p8, b078p7, 144.9240, 0.0, 0.0, name="arcdipole")
    #     appell19 = EllipseApertureNode(a115p8, b078p7, 148.9250, 0.0, 0.0, name="arcdipole")
    #     appell20 = EllipseApertureNode(a115p8, b078p7, 152.9250, 0.0, 0.0, name="arcdipole")
    #     appell21 = EllipseApertureNode(a115p8, b078p7, 156.9250, 0.0, 0.0, name="arcdipole")
    #     appell22 = EllipseApertureNode(a115p8, b078p7, 160.9250, 0.0, 0.0, name="arcdipole")
    #     appell23 = EllipseApertureNode(a115p8, b078p7, 164.9250, 0.0, 0.0, name="arcdipole")
    #     appell24 = EllipseApertureNode(a115p8, b078p7, 168.9260, 0.0, 0.0, name="arcdipole")
    #     appell25 = EllipseApertureNode(a115p8, b078p7, 202.9260, 0.0, 0.0, name="arcdipole")
    #     appell26 = EllipseApertureNode(a115p8, b078p7, 206.9260, 0.0, 0.0, name="arcdipole")
    #     appell27 = EllipseApertureNode(a115p8, b078p7, 210.9260, 0.0, 0.0, name="arcdipole")
    #     appell28 = EllipseApertureNode(a115p8, b078p7, 214.9260, 0.0, 0.0, name="arcdipole")
    #     appell29 = EllipseApertureNode(a115p8, b078p7, 218.9260, 0.0, 0.0, name="arcdipole")
    #     appell30 = EllipseApertureNode(a115p8, b078p7, 222.9270, 0.0, 0.0, name="arcdipole")
    #     appell31 = EllipseApertureNode(a115p8, b078p7, 226.9270, 0.0, 0.0, name="arcdipole")
    #     appell32 = EllipseApertureNode(a115p8, b078p7, 230.9270, 0.0, 0.0, name="arcdipole")

    #     ## These parent nodes should be defined by name, not index. The indexing throws errors
    #     ## in the standard MADX output lattice.
        
    #     ap06200 = nodes[173]
    #     ap06201 = nodes[186]

    #     ap10000 = nodes[15]
    #     ap10001 = nodes[16]
    #     ap10002 = nodes[21]
    #     ap10003 = nodes[31]
    #     ap10004 = nodes[45]
    #     ap10005 = nodes[58]
    #     ap10006 = nodes[78]
    #     ap10007 = nodes[103]
    #     ap10008 = nodes[116]
    #     ap10009 = nodes[130]
    #     ap10010 = nodes[140]
    #     ap10011 = nodes[144]
    #     ap10012 = nodes[147]
    #     ap10013 = nodes[150]
    #     ap10014 = nodes[153]
    #     ap10015 = nodes[156]
    #     ap10016 = nodes[158]
    #     ap10017 = nodes[160]
    #     ap10018 = nodes[168]
    #     ap10019 = nodes[198]
    #     ap10020 = nodes[199]
    #     ap10021 = nodes[202]
    #     ap10022 = nodes[203]
    #     ap10023 = nodes[211]
    #     ap10024 = nodes[225]
    #     ap10025 = nodes[238]
    #     ap10026 = nodes[258]
    #     ap10027 = nodes[283]
    #     ap10028 = nodes[296]
    #     ap10029 = nodes[310]
    #     ap10030 = nodes[319]
    #     ap10031 = nodes[322]
    #     ap10032 = nodes[330]
    #     ap10033 = nodes[344]
    #     ap10034 = nodes[351]
    #     ap10035 = nodes[358]
    #     ap10036 = nodes[359]
    #     ap10037 = nodes[360]
    #     ap10038 = nodes[364]
    #     ap10039 = nodes[371]
    #     ap10040 = nodes[372]
    #     ap10041 = nodes[380]
    #     ap10042 = nodes[394]
    #     ap10043 = nodes[407]
    #     ap10044 = nodes[427]
    #     ap10045 = nodes[452]
    #     ap10046 = nodes[465]
    #     ap10047 = nodes[479]
    #     ap10048 = nodes[489]
    #     ap10049 = nodes[491]
    #     ap10050 = nodes[502]
    #     ap10051 = nodes[503]
    #     ap10052 = nodes[504]
    #     ap10053 = nodes[506]
    #     ap10054 = nodes[508]
    #     ap10055 = nodes[510]
    #     ap10056 = nodes[514]
    #     ap10057 = nodes[521]
    #     ap10058 = nodes[524]
    #     ap10059 = nodes[535]
    #     ap10060 = nodes[549]
    #     ap10061 = nodes[562]
    #     ap10062 = nodes[582]
    #     ap10063 = nodes[607]
    #     ap10064 = nodes[620]
    #     ap10065 = nodes[634]
    #     ap10066 = nodes[644]
    #     ap10067 = nodes[647]
    #     ap10068 = nodes[651]
    #     ap10069 = nodes[661]

    #     ap12000 = nodes[5]
    #     ap12001 = nodes[8]
    #     ap12002 = nodes[172]
    #     ap12003 = nodes[174]
    #     ap12004 = nodes[185]
    #     ap12005 = nodes[187]
    #     ap12006 = nodes[341]
    #     ap12007 = nodes[361]
    #     ap12008 = nodes[498]
    #     ap12009 = nodes[511]
    #     ap12010 = nodes[658]

    #     ap12500 = nodes[70]
    #     ap12501 = nodes[91]
    #     ap12502 = nodes[250]
    #     ap12503 = nodes[271]
    #     ap12504 = nodes[419]
    #     ap12505 = nodes[440]
    #     ap12506 = nodes[574]
    #     ap12507 = nodes[595]

    #     ap13000 = nodes[175]
    #     ap13001 = nodes[184]
    #     ap13002 = nodes[188]
    #     ap13003 = nodes[189]
    #     ap13004 = nodes[192]

    #     ap14000 = nodes[6]
    #     ap14001 = nodes[7]
    #     ap14002 = nodes[169]
    #     ap14003 = nodes[170]
    #     ap14004 = nodes[171]
    #     ap14005 = nodes[176]
    #     ap14006 = nodes[180]
    #     ap14007 = nodes[183]
    #     ap14008 = nodes[190]
    #     ap14009 = nodes[191]
    #     ap14010 = nodes[342]
    #     ap14011 = nodes[343]
    #     ap14012 = nodes[362]
    #     ap14013 = nodes[363]
    #     ap14014 = nodes[500]
    #     ap14015 = nodes[501]
    #     ap14016 = nodes[512]
    #     ap14017 = nodes[513]
    #     ap14018 = nodes[659]
    #     ap14019 = nodes[660]

    #     apell00 = nodes[35]
    #     apell01 = nodes[49]
    #     apell02 = nodes[62]
    #     apell03 = nodes[74]
    #     apell04 = nodes[87]
    #     apell05 = nodes[99]
    #     apell06 = nodes[112]
    #     apell07 = nodes[126]
    #     apell08 = nodes[159]
    #     apell09 = nodes[215]
    #     apell10 = nodes[229]
    #     apell11 = nodes[242]
    #     apell12 = nodes[254]
    #     apell13 = nodes[267]
    #     apell14 = nodes[279]
    #     apell15 = nodes[292]
    #     apell16 = nodes[306]
    #     apell17 = nodes[384]
    #     apell18 = nodes[398]
    #     apell19 = nodes[411]
    #     apell20 = nodes[423]
    #     apell21 = nodes[436]
    #     apell22 = nodes[448]
    #     apell23 = nodes[461]
    #     apell24 = nodes[475]
    #     apell25 = nodes[539]
    #     apell26 = nodes[553]
    #     apell27 = nodes[566]
    #     apell28 = nodes[578]
    #     apell29 = nodes[591]
    #     apell30 = nodes[603]
    #     apell31 = nodes[616]
    #     apell32 = nodes[630]

    #     ap06200.addChildNode(app06200, AccNode.EXIT)
    #     ap06201.addChildNode(app06201, AccNode.EXIT)

    #     ap10000.addChildNode(app10000, AccNode.EXIT)
    #     ap10001.addChildNode(app10001, AccNode.EXIT)
    #     ap10002.addChildNode(app10002, AccNode.EXIT)
    #     ap10003.addChildNode(app10003, AccNode.EXIT)
    #     ap10004.addChildNode(app10004, AccNode.EXIT)
    #     ap10005.addChildNode(app10005, AccNode.EXIT)
    #     ap10006.addChildNode(app10006, AccNode.EXIT)
    #     ap10007.addChildNode(app10007, AccNode.EXIT)
    #     ap10008.addChildNode(app10008, AccNode.EXIT)
    #     ap10009.addChildNode(app10009, AccNode.EXIT)
    #     ap10010.addChildNode(app10010, AccNode.EXIT)
    #     ap10011.addChildNode(app10011, AccNode.EXIT)
    #     ap10012.addChildNode(app10012, AccNode.EXIT)
    #     ap10013.addChildNode(app10013, AccNode.EXIT)
    #     ap10014.addChildNode(app10014, AccNode.EXIT)
    #     ap10015.addChildNode(app10015, AccNode.EXIT)
    #     ap10016.addChildNode(app10016, AccNode.EXIT)
    #     ap10017.addChildNode(app10017, AccNode.EXIT)
    #     ap10018.addChildNode(app10018, AccNode.EXIT)
    #     ap10019.addChildNode(app10019, AccNode.EXIT)
    #     ap10020.addChildNode(app10020, AccNode.EXIT)
    #     ap10021.addChildNode(app10021, AccNode.EXIT)
    #     ap10022.addChildNode(app10022, AccNode.EXIT)
    #     ap10023.addChildNode(app10023, AccNode.EXIT)
    #     ap10024.addChildNode(app10024, AccNode.EXIT)
    #     ap10025.addChildNode(app10025, AccNode.EXIT)
    #     ap10026.addChildNode(app10026, AccNode.EXIT)
    #     ap10027.addChildNode(app10027, AccNode.EXIT)
    #     ap10028.addChildNode(app10028, AccNode.EXIT)
    #     ap10029.addChildNode(app10029, AccNode.EXIT)
    #     ap10030.addChildNode(app10030, AccNode.EXIT)
    #     ap10031.addChildNode(app10031, AccNode.EXIT)
    #     ap10032.addChildNode(app10032, AccNode.EXIT)
    #     ap10033.addChildNode(app10033, AccNode.EXIT)
    #     ap10034.addChildNode(app10034, AccNode.EXIT)
    #     ap10035.addChildNode(app10035, AccNode.EXIT)
    #     ap10036.addChildNode(app10036, AccNode.EXIT)
    #     ap10037.addChildNode(app10037, AccNode.EXIT)
    #     ap10038.addChildNode(app10038, AccNode.EXIT)
    #     ap10039.addChildNode(app10039, AccNode.EXIT)
    #     ap10040.addChildNode(app10040, AccNode.EXIT)
    #     ap10041.addChildNode(app10041, AccNode.EXIT)
    #     ap10042.addChildNode(app10042, AccNode.EXIT)
    #     ap10043.addChildNode(app10043, AccNode.EXIT)
    #     ap10044.addChildNode(app10044, AccNode.EXIT)
    #     ap10045.addChildNode(app10045, AccNode.EXIT)
    #     ap10046.addChildNode(app10046, AccNode.EXIT)
    #     ap10047.addChildNode(app10047, AccNode.EXIT)
    #     ap10048.addChildNode(app10048, AccNode.EXIT)
    #     ap10049.addChildNode(app10049, AccNode.EXIT)
    #     ap10050.addChildNode(app10050, AccNode.EXIT)
    #     ap10051.addChildNode(app10051, AccNode.EXIT)
    #     ap10052.addChildNode(app10052, AccNode.EXIT)
    #     ap10053.addChildNode(app10053, AccNode.EXIT)
    #     ap10054.addChildNode(app10054, AccNode.EXIT)
    #     ap10055.addChildNode(app10055, AccNode.EXIT)
    #     ap10056.addChildNode(app10056, AccNode.EXIT)
    #     ap10057.addChildNode(app10057, AccNode.EXIT)
    #     ap10058.addChildNode(app10058, AccNode.EXIT)
    #     ap10059.addChildNode(app10059, AccNode.EXIT)
    #     ap10060.addChildNode(app10060, AccNode.EXIT)
    #     ap10061.addChildNode(app10061, AccNode.EXIT)
    #     ap10062.addChildNode(app10062, AccNode.EXIT)
    #     ap10063.addChildNode(app10063, AccNode.EXIT)
    #     ap10064.addChildNode(app10064, AccNode.EXIT)
    #     ap10065.addChildNode(app10065, AccNode.EXIT)
    #     ap10066.addChildNode(app10066, AccNode.EXIT)
    #     ap10067.addChildNode(app10067, AccNode.EXIT)
    #     ap10068.addChildNode(app10068, AccNode.EXIT)
    #     ap10069.addChildNode(app10069, AccNode.EXIT)

    #     ap12000.addChildNode(app12000, AccNode.EXIT)
    #     ap12001.addChildNode(app12001, AccNode.EXIT)
    #     ap12002.addChildNode(app12002, AccNode.EXIT)
    #     ap12003.addChildNode(app12003, AccNode.EXIT)
    #     ap12004.addChildNode(app12004, AccNode.EXIT)
    #     ap12005.addChildNode(app12005, AccNode.EXIT)
    #     ap12006.addChildNode(app12006, AccNode.EXIT)
    #     ap12007.addChildNode(app12007, AccNode.EXIT)
    #     ap12008.addChildNode(app12008, AccNode.EXIT)
    #     ap12009.addChildNode(app12009, AccNode.EXIT)
    #     ap12010.addChildNode(app12010, AccNode.EXIT)

    #     ap12500.addChildNode(app12500, AccNode.EXIT)
    #     ap12501.addChildNode(app12501, AccNode.EXIT)
    #     ap12502.addChildNode(app12502, AccNode.EXIT)
    #     ap12503.addChildNode(app12503, AccNode.EXIT)
    #     ap12504.addChildNode(app12504, AccNode.EXIT)
    #     ap12505.addChildNode(app12505, AccNode.EXIT)
    #     ap12506.addChildNode(app12506, AccNode.EXIT)
    #     ap12507.addChildNode(app12507, AccNode.EXIT)

    #     ap13000.addChildNode(app13000, AccNode.EXIT)
    #     ap13001.addChildNode(app13001, AccNode.EXIT)
    #     ap13002.addChildNode(app13002, AccNode.EXIT)
    #     ap13003.addChildNode(app13003, AccNode.EXIT)
    #     ap13004.addChildNode(app13004, AccNode.EXIT)

    #     ap14000.addChildNode(app14000, AccNode.EXIT)
    #     ap14001.addChildNode(app14001, AccNode.EXIT)
    #     ap14002.addChildNode(app14002, AccNode.EXIT)
    #     ap14003.addChildNode(app14003, AccNode.EXIT)
    #     ap14004.addChildNode(app14004, AccNode.EXIT)
    #     ap14005.addChildNode(app14005, AccNode.EXIT)
    #     ap14006.addChildNode(app14006, AccNode.EXIT)
    #     ap14007.addChildNode(app14007, AccNode.EXIT)
    #     ap14008.addChildNode(app14008, AccNode.EXIT)
    #     ap14009.addChildNode(app14009, AccNode.EXIT)
    #     ap14010.addChildNode(app14010, AccNode.EXIT)
    #     ap14011.addChildNode(app14011, AccNode.EXIT)
    #     ap14012.addChildNode(app14012, AccNode.EXIT)
    #     ap14013.addChildNode(app14013, AccNode.EXIT)
    #     ap14014.addChildNode(app14014, AccNode.EXIT)
    #     ap14015.addChildNode(app14015, AccNode.EXIT)
    #     ap14016.addChildNode(app14016, AccNode.EXIT)
    #     ap14017.addChildNode(app14017, AccNode.EXIT)
    #     ap14018.addChildNode(app14018, AccNode.EXIT)
    #     ap14019.addChildNode(app14019, AccNode.EXIT)

    #     apell00.addChildNode(appell00, AccNode.EXIT)
    #     apell01.addChildNode(appell01, AccNode.EXIT)
    #     apell02.addChildNode(appell02, AccNode.EXIT)
    #     apell03.addChildNode(appell03, AccNode.EXIT)
    #     apell04.addChildNode(appell04, AccNode.EXIT)
    #     apell05.addChildNode(appell05, AccNode.EXIT)
    #     apell06.addChildNode(appell06, AccNode.EXIT)
    #     apell07.addChildNode(appell07, AccNode.EXIT)
    #     apell08.addChildNode(appell08, AccNode.EXIT)
    #     apell09.addChildNode(appell09, AccNode.EXIT)
    #     apell10.addChildNode(appell10, AccNode.EXIT)
    #     apell11.addChildNode(appell11, AccNode.EXIT)
    #     apell12.addChildNode(appell12, AccNode.EXIT)
    #     apell13.addChildNode(appell13, AccNode.EXIT)
    #     apell14.addChildNode(appell14, AccNode.EXIT)
    #     apell15.addChildNode(appell15, AccNode.EXIT)
    #     apell16.addChildNode(appell16, AccNode.EXIT)
    #     apell17.addChildNode(appell17, AccNode.EXIT)
    #     apell18.addChildNode(appell18, AccNode.EXIT)
    #     apell19.addChildNode(appell19, AccNode.EXIT)
    #     apell20.addChildNode(appell20, AccNode.EXIT)
    #     apell21.addChildNode(appell21, AccNode.EXIT)
    #     apell22.addChildNode(appell22, AccNode.EXIT)
    #     apell23.addChildNode(appell23, AccNode.EXIT)
    #     apell24.addChildNode(appell24, AccNode.EXIT)
    #     apell25.addChildNode(appell25, AccNode.EXIT)
    #     apell26.addChildNode(appell26, AccNode.EXIT)
    #     apell27.addChildNode(appell27, AccNode.EXIT)
    #     apell28.addChildNode(appell28, AccNode.EXIT)
    #     apell29.addChildNode(appell29, AccNode.EXIT)
    #     apell30.addChildNode(appell30, AccNode.EXIT)
    #     apell31.addChildNode(appell31, AccNode.EXIT)
    #     apell32.addChildNode(appell32, AccNode.EXIT)

    # def add_collimator_node(
    #     self,
    #     length=0.00001,
    #     ma=9,
    #     density_fac=1.0,
    #     shape=1,
    #     a=0.110,
    #     b=0.0,
    #     c=0.0,
    #     d=0.0,
    #     angle=0.0,
    #     position=0.5,
    #     name="collimator_node",
    # ):            
    #     collimator_node = TeapotCollimatorNode(length, ma, density_fac, shape, a, b, c, d, angle, position, name)
    #     addTeapotCollimatorNode(self, position, collimator_node)        
    #     return collimator_node

    # def add_collimator_nodes(self):
    #     self.add_collimator_node(0.60000, 3, 1.00, 2, 0.100, 0.100, 0.0, 0.0, 0.0, 50.3771, "p1")
    #     self.add_collimator_node(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0, 0.0, 51.1921, "scr1t")
    #     self.add_collimator_node(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0, 0.0, 51.1966, "scr1c")
    #     self.add_collimator_node(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0, 90.0, 51.2365, "scr2t")
    #     self.add_collimator_node(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0, 90.0, 51.2410, "scr2c")
    #     self.add_collimator_node(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0, 45.0, 51.3902, "scr3t")
    #     self.add_collimator_node(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0, 45.0, 51.3947, "scr3c")
    #     self.add_collimator_node(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0, -45.0, 51.4346, "scr4t")
    #     self.add_collimator_node(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0, -45.0, 51.4391, "scr4c")
    #     self.add_collimator_node(0.32000, 3, 1.00, 2, 0.100, 0.100, 0.0, 0.0, 0.0, 51.6228, "p2sh1")
    #     self.add_collimator_node(1.80000, 3, 0.65, 2, 0.080, 0.048, 0.0, 0.0, 0.0, 51.9428, "p2")
    #     self.add_collimator_node(0.32000, 3, 1.00, 2, 0.100, 0.100, 0.0, 0.0, 0.0, 53.7428, "p2sh2")
    #     self.add_collimator_node(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0, 0.0, 57.2079, "s1sh1")
    #     self.add_collimator_node(1.19000, 3, 0.65, 2, 0.0625, 0.0625, 0.0, 0.0, 0.0, 58.2179, "s1")
    #     self.add_collimator_node(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0, 0.0, 59.4079, "s1sh2")
    #     self.add_collimator_node(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0, 0.0, 64.8679, "s2sh1")
    #     self.add_collimator_node(1.19000, 3, 0.65, 2, 0.0625, 0.0625, 0.0, 0.0, 0.0, 65.8779, "s2")
    #     self.add_collimator_node(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0, 0.0, 67.0679, "s2sh2")
        
    # def add_harmonic_rf_node(self, rf1=None, rf2=None):
    #     z_to_phi = 2.0 * math.pi / self.getLength()
    #     dE_sync = 0.0
    #     length = 0.0
    #     if rf1 is None:
    #         rf1 = dict()
    #     if rf2 is None:
    #         rf2 = dict()
    #     rf1.setdefault("phase", 0.0)
    #     rf1.setdefault("hnum", 1.0)
    #     rf1.setdefault("voltage", +2.00e-06)
    #     rf2.setdefault("phase", 0.0)
    #     rf2.setdefault("hnum", 2.0)
    #     rf2.setdefault("voltage", -4.00e-06)
        
    #     rf_nodes = {
    #         "rf1a": RFNode.Harmonic_RFNode(
    #             z_to_phi, 
    #             dE_sync, 
    #             rf1["hnum"], 
    #             rf1["voltage"], 
    #             rf1["phase"],
    #             length,
    #             name="rf1a",
    #         ),
    #         "rf1b": RFNode.Harmonic_RFNode(
    #             z_to_phi,
    #             dE_sync, 
    #             rf1["hnum"], 
    #             rf1["voltage"], 
    #             rf1["phase"], 
    #             length, 
    #             name="rf1b",
    #         ),
    #         "rf1c": RFNode.Harmonic_RFNode(
    #             z_to_phi, 
    #             dE_sync,
    #             rf1["hnum"], 
    #             rf1["voltage"], 
    #             rf1["phase"], 
    #             length, 
    #             name="rf1c",
    #         ),
    #         "rf2": RFNode.Harmonic_RFNode(
    #             z_to_phi, 
    #             dE_sync, 
    #             rf2["hnum"], 
    #             rf2["voltage"], 
    #             rf2["phase"], 
    #             length, 
    #             name="rf2",
    #         )
    #     }          
    #     # Positions are hard-coded.
    #     rf_nodes = [rf_nodes[key] for key in ["rf1a", "rf1b", "rf1c", "rf2"]]
    #     RFLatticeModifications.addRFNode(self, 184.273, rf_nodes[0])
    #     RFLatticeModifications.addRFNode(self, 186.571, rf_nodes[1])
    #     RFLatticeModifications.addRFNode(self, 188.868, rf_nodes[2])
    #     RFLatticeModifications.addRFNode(self, 191.165, rf_nodes[3])
    #     return rf_nodes

    # def get_injection_controller(
    #     self, 
    #     mass=None,
    #     kin_energy=None,
    #     inj_mid="injm1",
    #     inj_start="bpm_a09",
    #     inj_end="bpm_b01",
    #     **kws
    # ):
    #     return InjectionController(
    #         ring=self,
    #         mass=mass,
    #         kin_energy=kin_energy,
    #         inj_mid=inj_mid,
    #         inj_start=inj_start,
    #         inj_end=inj_end,
    #     )
    
    
    
    
    
    
    
    
# class SingleParticleMonitorNode(teapot.DriftTEAPOT):
# 	def __init__(self, position=None):
# 		DriftTEAPOT.__init__(self)
#         self.position = position
#         self.coords = np.zeros(4)

# 	def track(self, params_dict):
# 		bunch = params_dict["bunch"]
#         self.coords[0] = bunch.x(0)
#         self.coords[2] = bunch.y(0)
#         self.coords[1] = bunch.xp(0)
#         self.coords[3] = bunch.yp(0)
        
    
class InjectionController:    
    def __init__(
        self,
        ring=None,
        mass=None,
        kin_energy=None,
        inj_mid="injm1",
        inj_start="bpm_a09",
        inj_end="bpm_b01",
    ):
        self.ring = ring
        self.mass = mass
        self.kin_energy = kin_energy
        self.kicker_names = [
            "ikickh_a10",
            "ikickv_a10",
            "ikickh_a11",
            "ikickv_a11",
            "ikickv_a12",
            "ikickh_a12",
            "ikickv_a13",
            "ikickh_a13",
        ]
        self.kicker_nodes = [ring.getNodeForName(name) for name in self.kicker_names]

        # Maximum injection kicker angles at 1.0 GeV [mrad].
        self.min_kicker_angles = MIN_KICKER_ANGLES
        self.max_kicker_angles = MAX_KICKER_ANGLES

        # Convert the maximum kicker angles from [mrad] to [rad].
        self.min_kicker_angles = 1.00e-03 * self.min_kicker_angles
        self.max_kicker_angles = 1.00e-03 * self.max_kicker_angles

        # Scale the maximum kicker angles based on kinetic energy. (They are defined for
        # the nominal SNS kinetic energy of 1.0 GeV.)
        self.kin_energy_scale_factor = get_pc(mass, kin_energy=1.0) / get_pc(mass, kin_energy)
        self.min_kicker_angles *= self.kin_energy_scale_factor
        self.max_kicker_angles *= self.kin_energy_scale_factor

        # Identify the horizontal and vertical kicker nodes. (PyORBIT does not
        # distinguish between horizontal/vertical kickers.)
        self.kicker_idx_x = [0, 2, 5, 7]
        self.kicker_idx_y = [1, 3, 4, 6]
        self.kicker_nodes_x = [self.kicker_nodes[i] for i in self.kicker_idx_x]
        self.kicker_nodes_y = [self.kicker_nodes[i] for i in self.kicker_idx_y]
        self.min_kicker_angles_x = self.min_kicker_angles[self.kicker_idx_x]
        self.max_kicker_angles_x = self.max_kicker_angles[self.kicker_idx_x]
        self.min_kicker_angles_y = self.min_kicker_angles[self.kicker_idx_y]
        self.max_kicker_angles_y = self.max_kicker_angles[self.kicker_idx_y]

        # Identify dipole correctors. These will be used to make a closed bump.
        self.vcorrector_names = ["dmcv_a09", "dchv_a10", "dchv_a13", "dmcv_b01"]
        self.vcorrector_nodes = [self.ring.getNodeForName(name) for name in self.vcorrector_names]
        self.max_vcorrector_angle = MAX_VCORRECTOR_ANGLE  # [mrad]
        self.max_vcorrector_angle *= 1.0e-3  # [mrad] --> [rad]
        self.max_vcorrector_angle *= self.kin_energy_scale_factor
        self.min_vcorrector_angle = -self.max_vcorrector_angle

        # Create one sublattice for the first half of the injection region (before the foil)
        # and one sublattice for the second half of the injection region (afterthe foil).
        self.sublattice1 = self.ring.getSubLattice(
            self.ring.getNodeIndex(self.ring.getNodeForName(inj_start)),
            -1,
        )
        self.sublattice2 = self.ring.getSubLattice(
            self.ring.getNodeIndex(self.ring.getNodeForName(inj_mid)),
            self.ring.getNodeIndex(self.ring.getNodeForName(inj_end)),
        )        
        self.sublattices = [self.sublattice1, self.sublattice2]
                
        # Initialize bunch for single-particle tracking.
        self.bunch = Bunch()
        self.bunch.mass(mass)
        self.bunch.getSyncParticle().kinEnergy(kin_energy)
        self.params_dict = {"lostbunch": Bunch()}  # for apertures
        self.bunch.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    def scale_kicker_limits(self, factor=1.0):
        """Scale all kicker limits by factor."""
        self.min_kicker_angles = factor * self.min_kicker_angles
        self.max_kicker_angles = factor * self.max_kicker_angles
        self.min_kicker_angles_x = self.min_kicker_angles[self.kicker_idx_x]
        self.max_kicker_angles_x = self.max_kicker_angles[self.kicker_idx_x]
        self.min_kicker_angles_y = self.min_kicker_angles[self.kicker_idx_y]
        self.max_kicker_angles_y = self.max_kicker_angles[self.kicker_idx_y]
        
    def get_nodes(self):
        """Return list of nodes inj injection region (in order)."""
        return self.sublattice1.getNodes() + self.sublattice2.getNodes()

    def get_kicker_angles_x(self):
        """Return horizontal kicker angles"""
        return np.array([node.getParam("kx") for node in self.kicker_nodes_x])

    def get_kicker_angles_y(self):
        """Return vertical kicker angles"""
        return np.array([node.getParam("ky") for node in self.kicker_nodes_y])

    def get_kicker_angles(self):
        """Return all kicker angles."""
        angles = []
        for node in self.kicker_nodes:
            if node in self.kicker_nodes_x:
                angles.append(node.getParam("kx"))
            elif node in self.kicker_nodes_y:
                angles.append(node.getParam("ky"))
        return np.array(angles)

    def set_kicker_angles_x(self, angles=None):
        """Set horizontal kicker angles."""
        if angles is not None:
            for angle, node in zip(angles, self.kicker_nodes_x):
                node.setParam("kx", angle)

    def set_kicker_angles_y(self, angles=None):
        """Set vertical kicker angles."""
        if angles is not None:
            for angle, node in zip(angles, self.kicker_nodes_y):
                node.setParam("ky", angle)
                
    def set_kicker_angles(self, angles):
        """Set all kicker angles."""
        angles_x = [angles[i] for i in self.kicker_idx_x]
        angles_y = [angles[i] for i in self.kicker_idx_y]
        self.set_kicker_angles_x(angles_x)
        self.set_kicker_angles_y(angles_y)
        
    def get_vcorrector_angles(self):
        """Return vertical orbit corrector angles."""
        return np.array([node.getParam("ky") for node in self.vcorrector_nodes])

    def set_vcorrector_angles(self, angles):
        """Set vertical orbit corrector angles."""
        for angle, node in zip(angles, self.vcorrector_nodes):
            node.setParam("ky", angle)
        
    def init_part(self, x=0.0, xp=0.0, y=0.0, yp=0.0):
        """Initialize a single particle at the origin."""
        self.bunch.deleteParticle(0)
        self.bunch.addParticle(x, xp, y, yp, 0.0, 0.0)
        
    def track_part(self, sublattice=0):
        """Track particle through one half of injection"""
        self.sublattices[sublattice].trackBunch(self.bunch, self.params_dict)
        return np.array([self.bunch.x(0), self.bunch.xp(0), self.bunch.y(0), self.bunch.yp(0)])
                            
    def set_inj_coords(self, coords, guess=None, weight_momentum=10.0, **kws):
        
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        
        kws.setdefault("xtol", 1.00e-12)
        kws.setdefault("ftol", 1.00e-12)
        kws.setdefault("gtol", 1.00e-12)
                
        coords = np.array(coords)
        if guess is None:
            guess = np.zeros(8)
                    
        def norm(_coords):
            for i in (1, 3):
                _coords[i] *= weight_momentum
            return np.mean(np.square(_coords))
                            
        def cost_func(angles):
            self.set_kicker_angles(angles)
            self.init_part(0.0, 0.0, 0.0, 0.0)
            coords_mid = self.track_part(sublattice=0)
            coords_end = self.track_part(sublattice=1)
            cost = norm(coords_mid - coords) + norm(coords_end)
            cost = cost * 1.00e+06
            return cost, coords_mid, coords_end
        
        
        class Optimizer:
            def __init__(self):
                self.calls = 0
                
            def objective(self, x):
                cost, coords_mid, coords_end = cost_func(x)
                self.calls += 1
                message = "call={:05.0f} ".format(self.calls)
                message += "x={:>+5.3f} xp={:>+5.3f} y={:>+5.3f} yp={:>+5.3f}".format(
                    1000.0 * coords_mid[0],
                    1000.0 * coords_mid[1],
                    1000.0 * coords_mid[2],
                    1000.0 * coords_mid[3],
                )
                message += " | "
                message += "x={:>+5.3f} xp={:>+5.3f} y={:>+5.3f} yp={:>+5.3f}".format(
                    1000.0 * coords_end[0],
                    1000.0 * coords_end[1],
                    1000.0 * coords_end[2],
                    1000.0 * coords_end[3],
                )
                if _mpi_rank == 0:
                    print(message)
                return cost
            
        optimizer = Optimizer()
        opt.minimize(
            optimizer.objective, 
            guess, 
            bounds=opt.Bounds(self.min_kicker_angles, self.max_kicker_angles), 
            method="trust-constr", 
            options=dict(xtol=1.00e-10, gtol=1.00e-10),
        )
        return self.get_kicker_angles()
    
    def get_trajectory(self):
        node_pos_dict = self.ring.getNodePositionsDict()
        
        class Monitor:
            def __init__(self):
                self.coords = []
                self.nodes = []
                
            def action(self, params_dict):
                bunch = params_dict["bunch"]
                node = params_dict["node"]
                if node in node_pos_dict:
                    self.nodes.append(node)
                    self.coords.append([bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)])

        monitors = []
        self.init_part(0.0, 0.0, 0.0, 0.0)
        for i, sublattice in enumerate(self.sublattices):
            monitor = Monitor()
            actions_container = AccActionsContainer()
            actions_container.addAction(monitor.action, AccActionsContainer.ENTRANCE)
            sublattice.trackBunch(self.bunch, self.params_dict, actions_container)
            monitors.append(monitor)
            
        positions = []
        offset = 0.0
        for i, monitor in enumerate(monitors):
            if i == 1:
                offset = positions[-1]
            positions.append(offset)
            for j in range(1, len(monitor.nodes)):
                start, _ = node_pos_dict[monitor.nodes[j - 1]]
                stop, _ = node_pos_dict[monitor.nodes[j]]
                step = np.abs(stop - start)
                positions.append(positions[-1] + step)
            
        nodes = monitors[0].nodes + monitors[1].nodes
        nodes = [node.getName() for node in nodes]
        
        coords = monitors[0].coords + monitors[1].coords
            
        data = pd.DataFrame(coords, columns=["x", "xp", "y", "yp"])
        data["s"] = positions
        data["node"] = nodes
        return data
    
    def print_inj_coords(self):
        coords_start = np.zeros(4)
        self.init_part(*coords_start)
        coords_mid = self.track_part(sublattice=0)
        coords_end = self.track_part(sublattice=1)
        for _coords, tag in zip([coords_start, coords_mid, coords_end], ["start", "mid", "end"]):
            _coords = _coords * 1000.0
            print("Coordinates at inj_{}:".format(tag))
            print("  x  = {:0.3f} [mm]".format(_coords[0]))
            print("  y  = {:0.3f} [mm]".format(_coords[2]))
            print("  xp = {:0.3f} [mrad]".format(_coords[1]))
            print("  yp = {:0.3f} [mrad]".format(_coords[3]))