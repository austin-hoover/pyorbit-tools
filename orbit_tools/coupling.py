import numpy as np


def eigentunes(M: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(M)
    return eigtunes_from_eigvals(eigvals)
    

def eigtunes_from_eigvals(eigvals: np.ndarray) -> np.ndarray:
    return np.arccos(eigvals[::2].real) / (2.0 * np.pi)


