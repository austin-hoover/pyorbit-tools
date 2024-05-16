"""Setup functions that may be useful to multiple simulation models."""
from typing import Any
from typing import Callable
from omegaconf import DictConfig

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.injection import JohoTransverse
from orbit.injection import JohoLongitudinal
from orbit.injection import SNSESpreadDist
from orbit.injection import UniformLongDist
from orbit.utils.consts import speed_of_light


def make_joho_distribution(
    lattice: AccLattice,
    bunch: Bunch,
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


def make_uniform_longitudinal_distribution(
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


def make_sns_espread_distribution(
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


def make_distribution(funcs: dict, name: str, lattice: AccLattice, bunch: Bunch, **kws) -> Any:
    return funcs[name](lattice, bunch, **kws)


def make_distribution_6d(name: str, lattice: AccLattice, bunch: Bunch, **kws) -> Any:
    funcs = {
        "guassian": None,
        "waterbag": None,
        "kv": None,
    }
    return make_distribution(funcs, name, lattice, bunch, **kws)
    

def make_transverse_distribution_4d(name: str, lattice: AccLattice, bunch: Bunch, **kws) -> Any:
    funcs = {}
    return make_distribution(funcs, name, lattice, bunch, **kws)


def make_transverse_distribution_2d(name: str, lattice: AccLattice, bunch: Bunch, **kws) -> Any:
    funcs = {
        "joho": make_joho_distribution,
        "guassian": None,
        "waterbag": None,
        "kv": None,
    }
    return make_distribution(funcs, name, lattice, bunch, **kws)


def make_longitudinal_distribution(name: str, lattice: AccLattice, bunch: Bunch, **kws) -> Any:
    funcs = {
        "uniform": make_uniform_longitudinal_distribution,
        "sns_espread": make_sns_espread_distribution,
    }
    return make_distribution(funcs, name, lattice, bunch, **kws)


def make_lostbunch() -> Bunch:
    lostbunch = Bunch()
    lostbunch.addPartAttr("LostParticleAttributes")
    return lostbunch


def make_params_dict(bunch, lostbunch):
    params_dict = {}
    params_dict["bunch"] = bunch
    params_dict["lostbunch"] = lostbunch
    return params_dict


def make_empty_bunch(cfg: DictConfig) -> tuple[Bunch, Bunch, dict]:
    bunch = Bunch()
    bunch.mass(cfg.bunch.mass)
    bunch.getSyncParticle().kinEnergy(cfg.bunch.energy)
    
    lostbunch = make_lostbunch()
    params_dict = make_params_dict(bunch, lostbunch)
    
    return (bunch, lostbunch, params_dict)
