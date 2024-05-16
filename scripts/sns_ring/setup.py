from typing import Callable
from omegaconf import DictConfig

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.injection import JohoTransverse
from orbit.injection import JohoLongitudinal
from orbit.injection import SNSESpreadDist
from orbit.injection import UniformLongDist
from orbit.utils.consts import speed_of_light

from orbitsim.models.sns.ring import SNS_RING


def make_dist_tran_joho(
    order: int,
    alpha: float,
    beta: float,
    eps: float,
    pos: float,
    mom: float,
) -> JohoTransverse:
    """Make 2D joho distribution (transverse)."""
    eps_lim = (2.0 * (1.0 + order) * eps)
    return JohoTransverse(order, alpha, beta, eps_lim, pos, mom)


def make_dist_long_uniform(
    lattice: AccLattice,
    bunch: Bunch,
    fill_fraction: float,
    energy_offset: float,
    frac_energy_spread: float,
) -> UniformLongDist:
    """Make 2D uniform distribution (longitudional)."""
    zmax = 0.5 * fill_fraction * lattice.getLength()
    zmin = -zmax
    sync_part = bunch.getSyncParticle()
    return UniformLongDist(zmin, zmax, sync_part, energy_offset, frac_energy_spread)


def make_dist_long_sns_espread(
    lattice: AccLattice,
    bunch: Bunch, 
    fill_fraction: float,
    tail_fraction: float,
    energy: dict,
) -> SNSESpreadDist:
    """Make 2D SNS energy spread distribution (longitudinal)."""
    energy = DictConfig(energy)

    sync_part = bunch.getSyncParticle()

    zmax = 0.5 * fill_fraction * lattice.getLength()
    zmin = -zmax

    emean = sync_part.kinEnergy()
    esigma = energy.sigma
    etrunc = energy.trunc
    emin = sync_part.kinEnergy() + energy.min
    emax = sync_part.kinEnergy() + energy.max

    # Check what this is doing. Do we need to change if tracking fewer turns?
    time_per_turn = lattice.getLength() / (sync_part.beta() * speed_of_light)
    n_turns = 1026
    drift_time = 1000.0 * n_turns * time_per_turn
    
    ec_params = (
        energy.centroid.mean, 
        energy.centroid.sigma,
        energy.centroid.trunc, 
        energy.centroid.min, 
        energy.centroid.max, 
        energy.centroid.drifti, 
        energy.centroid.driftf, 
        drift_time,
    )
    
    es_params = (
        energy.spread.nu,
        energy.spread.phase,
        energy.spread.max,
        energy.spread.nulltime,
    )
    
    dist = SNSESpreadDist(
        lattice.getLength(), 
        zmin, 
        zmax, 
        tail_fraction, 
        sync_part, 
        emean, 
        esigma, 
        etrunc, 
        emin, 
        emax,
        ec_params, 
        es_params,
    )
    return dist


def make_lostbunch() -> Bunch:
    lostbunch = Bunch()
    lostbunch.addPartAttr("LostParticleAttributes")
    return lostbunch


def make_params_dict(bunch, lostbunch):
    params_dict = {}
    params_dict["bunch"] = bunch
    params_dict["lostbunch"] = lostbunch
    return params_dict


def make_empty_bunch(cfg: DictConfig) -> Bunch:
    bunch = Bunch()
    bunch.mass(cfg.bunch.mass)
    bunch.getSyncParticle().kinEnergy(cfg.bunch.energy)
    return bunch


def make_bunch_from_dist(cfg: DictConfig) -> Bunch:
    """Make bunch from particle generator."""
    bunch = make_empty_bunch(cfg)
    # name = cfg.bunch.dist.name
    # [...]
    return bunch


def setup_bunch(cfg: DictConfig, make_bunch: Callable) -> tuple[Bunch, Bunch, dict]:
    """Set up bunch from config and `make_bunch` function."""
    bunch = make_bunch(cfg)
    lostbunch = make_lostbunch()
    params_dict = make_params_dict(bunch, lostbunch)
    return (bunch, lostbunch, params_dict)
    

def setup_ring(cfg: DictConfig, ring: SNS_RING) -> SNS_RING:
    """Set up ring (add nodes) from config.

    - foil scattering node
    - aperture nodes
    - displacement nodes
    - rf cavity nodes
    - longitudinal impedance node
    - transverse impedance nodes
    - longitudional space charge node
    - transverse space charge nodes
    """
    if cfg.lattice.foil:
        ring.add_foil_node(**cfg.foil)

    if cfg.lattice.apertures:
        ring.add_injection_chicane_apertures_and_displacements()
        ring.add_apertures()

    if cfg.lattice.rf:
        ring.add_rf_cavities(**cfg.rf)

    if cfg.lattice.impedance.z:
        ring.add_longitudinal_impedance_node(**cfg.impedance.z)

    if cfg.lattice.impedance.xy:
        ring.add_transverse_impedance_node(**cfg.impedance.xy)

    if cfg.lattice.spacecharge.z:
        ring.add_longitudinal_spacecharge_node(**cfg.spacecharge.z)

    if cfg.lattice.spacecharge.xy:
        ring.add_transverse_spacecharge_nodes(**cfg.spacecharge.xy)

    return ring