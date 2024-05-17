import numpy as np
import scipy.optimize as opt

from orbit.core.bunch import Bunch
from orbit.core import orbit_mpi
from orbit.lattice import AccLattice
from orbit.lattice import AccNode

from orbitsim.misc import get_magnetic_rigidity
from orbitsim.lattice import get_node_for_name_any_case


def get_inj_kicker_angle_limits(mass: float, kin_energy: float) -> tuple[np.ndarray]:
    min_angles = 1.15 * np.array([0.0, 0.0, -7.13, -7.13, -7.13, -7.13, 0.0, 0.0])  # [mrad]
    max_angles = 1.15 * np.array([12.84, 12.84, 0.0, 0.0, 0.0, 0.0, 12.84, 12.84])  # [mrad]
    min_angles *= 0.001
    max_angles *= 0.001
    scale_factor = get_magnetic_rigidity(mass=mass, kin_energy=1.0) / get_magnetic_rigidity(mass=mass, kin_energy=kin_energy)
    min_angles *= scale_factor
    max_angles *= scale_factor
    return (min_angles, max_angles)


def get_inj_corrector_angle_limits(mass: float, kin_energy: float) -> tuple[float]:
    max_angle = 1.5  # [mrad]
    min_angle = -max_angle
    min_angle *= 0.001
    max_angle *= 0.001
    scale_factor = get_magnetic_rigidity(mass=mass, kin_energy=1.0) / get_magnetic_rigidity(mass=mass, kin_energy=kin_energy)
    min_angle *= scale_factor
    max_angle *= scale_factor
    return (min_angle, max_angle)


class RingInjectionController:    
    def __init__(
        self,
        lattice: AccLattice,
        mass: float,
        kin_energy: float,
        inj_start: str = "bpm_a09",
        inj_mid: str = "injm1",
        inj_stop: str = "bpm_b01",
    ) -> None:
        self.lattice = lattice
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

        self.kicker_nodes = [get_node_for_name_any_case(lattice, name) for name in self.kicker_names]
        
        # Get kicker limits.
        self.min_kicker_angles, self.max_kicker_angles = get_inj_kicker_angle_limits(mass, kin_energy)

        # Identify the horizontal and vertical kicker nodes.
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
        self.vcorrector_nodes = [get_node_for_name_any_case(lattice, name) for name in self.vcorrector_names]
        self.min_vcorrector_angle, self.max_vcorrector_angle = get_inj_corrector_angle_limits(mass, kin_energy)

        # Create one sublattice for the first half of the injection region (before the foil)
        # and one sublattice for the second half of the injection region (after the foil).
        lo = lattice.getNodeIndex(get_node_for_name_any_case(lattice, inj_start))
        hi = -1
        self.sublattice1 = self.lattice.getSubLattice(lo, hi)

        lo = lattice.getNodeIndex(get_node_for_name_any_case(lattice, inj_mid))
        hi = lattice.getNodeIndex(get_node_for_name_any_case(lattice, inj_stop))
        self.sublattice2 = self.lattice.getSubLattice(lo, hi)

        self.sublattices = [self.sublattice1, self.sublattice2]
        
        # Add inactive monitor nodes
        # [...]
                
        # Initialize bunch for single-particle tracking.
        self.bunch = Bunch()
        self.bunch.mass(mass)
        self.bunch.getSyncParticle().kinEnergy(kin_energy)
        self.params_dict = {"lostbunch": Bunch()}
        self.bunch.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def get_nodes(self) -> list[AccNode]:
        return self.sublattice1.getNodes() + self.sublattice2.getNodes()

    def get_kicker_angles_x(self) -> np.ndarray:
        return np.array([node.getParam("kx") for node in self.kicker_nodes_x])

    def get_kicker_angles_y(self) -> np.ndarray:
        return np.array([node.getParam("ky") for node in self.kicker_nodes_y])

    def get_kicker_angles(self) -> np.ndarray:
        angles = []
        for node in self.kicker_nodes:
            if node in self.kicker_nodes_x:
                angles.append(node.getParam("kx"))
            elif node in self.kicker_nodes_y:
                angles.append(node.getParam("ky"))
        return np.array(angles)

    def set_kicker_angles_x(self, angles_x: np.ndarray) -> None:
        if angles_x is not None:
            for angle, node in zip(angles_x, self.kicker_nodes_x):
                node.setParam("kx", angle)

    def set_kicker_angles_y(self, angles_y: np.ndarray) -> None:
        if angles_y is not None:
            for angle, node in zip(angles_y, self.kicker_nodes_y):
                node.setParam("ky", angle)

    def set_kicker_angles(self, angles: np.ndarray) -> None:
        angles_x = [angles[i] for i in self.kicker_idx_x]
        angles_y = [angles[i] for i in self.kicker_idx_y]
        self.set_kicker_angles_x(angles_x)
        self.set_kicker_angles_y(angles_y)

    def get_vcorrector_angles(self) -> np.ndarray:
        return np.array([node.getParam("ky") for node in self.vcorrector_nodes])

    def set_vcorrector_angles(self, angles: np.ndarray):
        for angle, node in zip(angles, self.vcorrector_nodes):
            node.setParam("ky", angle)

    def init_part(self, x: float = 0.0, xp: float = 0.0, y: float = 0.0, yp: float = 0.0):
        self.bunch.deleteParticle(0)
        self.bunch.addParticle(x, xp, y, yp, 0.0, 0.0)

    def track_part(self, sublattice: int = 0):
        self.sublattices[sublattice].trackBunch(self.bunch, self.params_dict)
        x = self.bunch.x(0)
        y = self.bunch.y(0)
        xp = self.bunch.xp(0)
        yp = self.bunch.yp(0)
        return np.array([x, xp, y, yp])

    def set_inj_coords(self, x: float, xp: float, y: float, yp: float, method: str = "bfgs") -> None:
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        coords = np.array([x, xp, y, yp])
        
        def magnitude(coords: np.ndarray, scale: float = 1.00e+04):
            return scale * np.sum(np.square(coords))
                            
        def cost_func(angles):
            self.set_kicker_angles(angles)
            self.init_part(0.0, 0.0, 0.0, 0.0)
            coords_mid = self.track_part(sublattice=0)
            coords_end = self.track_part(sublattice=1)

            if _mpi_rank == 0:
                (x_beg, xp_beg, y_beg, yp_beg) = np.zeros(4)
                (x_mid, xp_mid, y_mid, yp_mid) = coords_mid * 1000.0
                (x_end, xp_end, y_end, yp_end) = coords_end * 1000.0
                print(f"x_beg={x_beg:+0.3f} xp_beg={xp_beg:+0.3f} y_beg={y_beg:+0.3f} yp_beg={yp_beg:+0.3f}")
                print(f"x_mid={x_mid:+0.3f} xp_mid={xp_mid:+0.3f} y_mid={y_mid:+0.3f} yp_mid={yp_mid:+0.3f}")
                print(f"x_end={x_end:+0.3f} xp_end={xp_end:+0.3f} y_end={y_end:+0.3f} yp_end={yp_end:+0.3f}")
                print()
            
            cost = magnitude(coords_mid - coords) + magnitude(coords_end)
            return cost
            
        lb = self.min_kicker_angles
        ub = self.max_kicker_angles
        guess = np.zeros(8)

        result = None
        if method == "bfgs":
            result = opt.minimize(
                cost_func, 
                guess, 
                bounds=opt.Bounds(lb, ub), 
                method="L-BFGS-B", 
                options=dict(disp=1, ftol=1.00e-12, gtol=1.00e-12)
            )
        elif method == "least_squares":
            result = opt.least_squares(
                cost_func, 
                guess, 
                bounds=(lb, ub), 
                # xtol=1.00e-08,
                # gtol=1.00e-08,
                # ftol=1.00e-08,
                max_nfev=10000,
                verbose=2, 
            )
        else:
            raise ValueError(f"Invalid method {method}")
        return result.x
        
    def set_inj_coords_vcorrectors(self, x: float, xp: float, y: float, yp: float, method: str = "bfgs"):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        coords = np.array([x, xp, y, yp])

        def magnitude(coords: np.ndarray, scale: float = 1.00e+04):
            return scale * np.sum(np.square(coords))
                            
        def cost_func(angles):
            self.set_vcorrector_angles(angles)
            self.init_part(0.0, 0.0, 0.0, 0.0)
            coords_mid = self.track_part(sublattice=0)
            coords_end = self.track_part(sublattice=1)

            if _mpi_rank == 0:
                (x_beg, xp_beg, y_beg, yp_beg) = np.zeros(4)
                (x_mid, xp_mid, y_mid, yp_mid) = coords_mid * 1000.0
                (x_end, xp_end, y_end, yp_end) = coords_end * 1000.0
                print(f"x_beg={x_beg:+0.3f} xp_beg={xp_beg:+0.3f} y_beg={y_beg:+0.3f} yp_beg={yp_beg:+0.3f}")
                print(f"x_mid={x_mid:+0.3f} xp_mid={xp_mid:+0.3f} y_mid={y_mid:+0.3f} yp_mid={yp_mid:+0.3f}")
                print(f"x_end={x_end:+0.3f} xp_end={xp_end:+0.3f} y_end={y_end:+0.3f} yp_end={yp_end:+0.3f}")
                print()
            
            cost = magnitude(coords_mid - coords) + magnitude(coords_end)
            return cost

        lb = [self.min_vcorrector_angle] * 4
        ub = [self.max_vcorrector_angle] * 4
        guess = np.zeros(4)

        result = None
        if method == "bfgs":
            result = opt.minimize(
                cost_func, 
                guess, 
                bounds=opt.Bounds(lb, ub), 
                method="L-BFGS-B", 
                options=dict(disp=1, ftol=1.00e-12, gtol=1.00e-12)
            )
        elif method == "least_squares":
            result = opt.least_squares(
                cost_func, 
                guess, 
                bounds=(lb, ub), 
                verbose=1, 
                xtol=1.00e-12,
                gtol=1.00e-12, 
                ftol=1.00e-12,
                max_nfev=10000,
            )
        else:
            raise ValueError(f"Invalid method {method}")
        return result.x
            
        