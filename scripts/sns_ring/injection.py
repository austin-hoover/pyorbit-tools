import orbit.injection


def get_joho_eps_lim(eps_rms: float, order: int) -> float:
    return eps_rms * 2.0 * (order + 1.0)


def make_transverse_dist_joho(
    order: int, 
    alpha: float, 
    beta: float, 
    eps_rms: float, 
    center: tuple[float],
    tailfraction: float = 0.0,
    tailfactor: float = 1.0,
):
    return None