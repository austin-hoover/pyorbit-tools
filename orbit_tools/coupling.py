import numpy as np


def eigentunes(M: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(M)
    return eigentunes_from_eigenvalues(eigenvalues)
    

def eigentunes_from_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    return np.arccos(eigenvalues[::2].real) / (2.0 * np.pi)


def unit_symplectic_matrix(ndim: int = 4) -> np.ndarray:
    U = np.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        U[i : i + 2, i : i + 2] = [[0.0, 1.0], [-1.0, 0.0]]
    return U


def normalize_eigenvectors(eigenvectors: np.ndarray) -> np.ndarray:
    ndim = eigenvectors.shape[0]
    U = unit_symplectic_matrix(ndim)
    for i in range(0, ndim, 2):
        v = eigenvectors[:, i]
        val = np.linalg.multi_dot([np.conj(v), U, v]).imag
        if val > 0.0:
            (eigenvectors[:, i], eigenvectors[:, i + 1]) = (eigenvectors[:, i + 1], eigenvectors[:, i])
        eigenvectors[:, i : i + 2] *= np.sqrt(2.0 / np.abs(val))
    return eigenvectors


def normalization_matrix_from_eigenvectors(eigenvectors: np.ndarray) -> np.ndarray:
    V = np.zeros(eigenvectors.shape)
    for i in range(0, V.shape[1], 2):
        V[:, i] = eigenvectors[:, i].real
        V[:, i + 1] = (1.0j * eigenvectors[:, i]).real
    Vinv = np.linalg.inv(V)
    return Vinv


def normalization_matrix_from_distribution(X: np.ndarray) -> np.ndarray:
    ndim = 4
    Sigma = np.cov(X[:, :ndim].T)
    U = unit_symplectic_matrix(ndim)
    SU = np.matmul(Sigma, U)
    eigenvalues, eigenvectors = np.linalg.eig(SU)
    eigenvectors = normalize_eigenvectors(eigenvectors)
    Winv = normalization_matrix_from_eigenvectors(eigenvectors)
    return Winv


