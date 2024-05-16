from omegaconf import DictConfig

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.injection import JohoTransverse
from orbit.injection import JohoLongitudinal
from orbit.injection import SNSESpreadDist
from orbit.injection import UniformLongDist
from orbit.utils.consts import speed_of_light


def make_joho_dist(
    order: int,
    alpha: float,
    beta: float,
    eps: float,
    pos: float,
    mom: float,
) -> JohoTransverse:
    """Make 2D transverse Joho distribution."""
    eps_lim = (2.0 * (1.0 + order) * eps)
    return JohoTransverse(order, alpha, beta, eps_lim, pos, mom)


def make_uniform_dist(
    lattice: AccLattice,
    bunch: Bunch,
    fill_fraction: float,
    energy_offset: float,
    frac_energy_spread: float,
) -> UniformLongDist:
    """Make 2D longitudinal Joho distribution."""
    zmax = 0.5 * fill_fraction * lattice.getLength()
    zmin = -zmax
    sync_part = bunch.getSyncParticle()
    return UniformLongDist(zmin, zmax, sync_part, energy_offset, frac_energy_spread)


def make_sns_espread_dist(
    lattice: AccLattice,
    bunch: Bunch, 
    fill_fraction: float,
    tail_fraction: float,
    energy: dict,
) -> SNSESpreadDist:
    """Make 2D longitudinal SNS energy spread distribution."""
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


def make_minipulse_generators_production(cfg: DictConfig, lattice: AccLattice, bunch: Bunch) -> tuple:
    dist_x = make_joho_dist(**cfg.inj.x)
    dist_y = make_joho_dist(**cfg.inj.y)
    dist_z = make_sns_espread_dist(lattice, bunch, **cfg.inj.z)
    return (dist_x, dist_y, dist_z)

