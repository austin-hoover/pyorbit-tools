import numpy as np


def eigentunes(M: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(M)
    return eigentunes_from_eigenvalues(eigenvalues)
    

def eigentunes_from_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    return np.arccos(eigenvalues[::2].real) / (2.0 * np.pi)


