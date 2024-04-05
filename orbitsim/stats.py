from typing import Tuple
from typing import Union

import numpy as np


def emittance_2d(cov: np.ndarray) -> float:
    return np.sqrt(np.linalg.det(cov))


def twiss_2d(cov: np.ndarray) -> Tuple[float]:
    eps = emittance_2d(cov)
    beta = cov[0, 0] / eps
    alpha = -cov[0, 1] / eps
    return (alpha, beta)


def emittance(cov: np.ndarray) -> Union[float, Tuple[float]]:
    emittances = []
    for i in range(0, cov.shape[0], 2):
        emittances.append(emittance_2d(cov[i : i + 2, i : i + 2]))
    if len(emittances) == 1:
        emittances = emittances[0]
    return emittances


def twiss(cov: np.ndarray) -> Union[float, Tuple[float]]:
    params = []
    for i in range(0, cov.shape[0], 2):
        params.extend(twiss_2d(cov[i : i + 2, i : i + 2]))
    return params
