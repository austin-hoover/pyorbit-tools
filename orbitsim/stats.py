from typing import Union

import numpy as np


def emittance(cov: np.ndarray) -> float:
    return np.sqrt(np.linalg.det(cov))


def apparent_emittances(cov: np.ndarray) -> Union[float, tuple[float]]:
    apparent_emittances = []
    for i in range(0, cov.shape[0], 2):
        apparent_emittances.append(emittance(cov[i : i + 2, i : i + 2]))
    if len(apparent_emittances) == 1:
        apparent_emittances = apparent_emittances[0]
    return apparent_emittances


def intrinsic_emittances(cov: np.ndarray) -> tuple[float]:
    # To do: compute eigenvalues to extend to 6 x 6, rather than 
    # using analytic eigenvalue solution specific to 4 x 4.
    Sigma = cov[:4, :4]
    U = np.array([
        [+0.0, +1.0, +0.0, +0.0], 
        [-1.0, +0.0, +0.0, +0.0], 
        [+0.0, +0.0, +0.0, +1.0], 
        [+0.0, +0.0, -1.0, +0.0],
    ])
    tr_SU2 = np.trace(np.linalg.matrix_power(np.matmul(Sigma, U), 2))
    det_S = np.linalg.det(Sigma)
    eps_1 = 0.5 * np.sqrt(-tr_SU2 + np.sqrt(tr_SU2**2 - 16.0 * det_S))
    eps_2 = 0.5 * np.sqrt(-tr_SU2 - np.sqrt(tr_SU2**2 - 16.0 * det_S))
    return (eps_1, eps_2)


def twiss_2d(cov: np.ndarray) -> tuple[float]:
    eps = emittance_2d(cov)
    beta = cov[0, 0] / eps
    alpha = -cov[0, 1] / eps
    return (alpha, beta)


def twiss(cov: np.ndarray) -> Union[float, tuple[float]]:
    params = []
    for i in range(0, cov.shape[0], 2):
        params.extend(twiss_2d(cov[i : i + 2, i : i + 2]))
    return params
