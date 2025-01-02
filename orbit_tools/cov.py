import numpy as np


def rms_emittance(S: np.ndarray) -> float:
    return np.sqrt(np.linalg.det(S))


def projected_emittances(S: np.ndarray) -> tuple[float, ...]:
    emittances = []
    for i in range(0, S.shape[0], 2):
        emittance = rms_emittance(S[i : i + 2, i : i + 2])
        emittances.append(emittance)
    return emittances


def intrinsic_emittances(S: np.ndarray) -> tuple[float, ...]:
    S = S[:4, :4].copy()
    U = unit_symplectic_matrix(ndim=4)
    tr_SU2 = np.trace(np.linalg.matrix_power(np.matmul(S, U), 2))
    det_S = np.linalg.det(S)
    eps1 = 0.5 * np.sqrt(-tr_SU2 + np.sqrt(tr_SU2**2 - 16.0 * det_S))
    eps2 = 0.5 * np.sqrt(-tr_SU2 - np.sqrt(tr_SU2**2 - 16.0 * det_S))
    return (eps1, eps2)


def twiss_2d(S: np.ndarray) -> tuple[float]:
    eps = rms_emittance(S)
    beta = S[0, 0] / eps
    alpha = -S[0, 1] / eps
    return (alpha, beta, eps)


def twiss(S: np.ndarray) -> tuple[float] | list[tuple[float]]:
    params = []
    for i in range(0, S.shape[0], 2):
        params.extend(twiss_2d(S[i : i + 2, i : i + 2]))
    if len(params) == 1:
        params = params[0]
    return params


def twiss_dict(S: np.ndarray) -> dict[str, float] | dict[str, dict[str, float]]:
    ndim = S.shape[0]

    params = twiss(S)
    if ndim == 2:
        params = [params]

    twiss = {}
    for p, dim in zip(params, ["x", "y", "z"]):
        twiss[dim] = {"alpha": p[0], "beta": p[1], "emittance": p[2]}

    if ndim == 1:
        twiss = twiss[twiss.keys()[0]]

    return twiss


def unit_symplectic_matrix(ndim: int) -> np.ndarray:
    """Return matrix U such that, if M is a symplectic matrix, UMU^T = M."""
    U = np.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        U[i : i + 2, i : i + 2] = [[0.0, +1.0], [-1.0, 0.0]]
    return U


def normalize_eigvecs(eigvecs: np.ndarray) -> np.ndarray:
    """Normalize eigenvectors according to Lebedev-Bogacz convention."""
    ndim = eigvecs.shape[0]
    U = unit_symplectic_matrix(ndim)
    for i in range(0, ndim, 2):
        v = eigvecs[:, i]
        value = np.linalg.multi_dot([np.conj(v), U, v]).imag
        if value > 0.0:
            (eigvecs[:, i], eigvecs[:, i + 1]) = (eigvecs[:, i + 1], eigvecs[:, i])
        eigvecs[:, i : i + 2] *= np.sqrt(2.0 / np.abs(value))
    return eigvecs


def norm_matrix_from_eigvecs(eigvecs: np.ndarray) -> np.ndarray:
    """Return normalization matrix V^-1 from eigenvectors."""
    V = np.zeros(eigvecs.shape)
    for i in range(0, V.shape[1], 2):
        V[:, i] = eigvecs[:, i].real
        V[:, i + 1] = (1.0j * eigvecs[:, i]).real
    return np.linalg.inv(V)


def norm_matrix_from_twiss_2d(alpha: float, beta: float, emittance: float = None) -> np.ndarray:
    V = np.array([[beta, 0.0], [-alpha, 1.0]]) * np.sqrt(1.0 / beta)
    A = np.eye(2)
    if eps is not None:
        A = np.sqrt(np.diag([eps, eps]))
    V = np.matmul(V, A)
    return np.linalg.inv(V)


def norm_matrix(S: np.ndarray, scale: bool = False, block_diag: bool = False) -> np.ndarray:
    """Return normalization matrix V^{-1} from covariance matrix S.

    Parameters
    ----------
    S : np.ndarray
        An N x N covariance matrix.
    scale : bool
        If True, normalize to unit rms emittance.
    block_diag : bool
        If true, normalize only 2x2 block-diagonal elements (x-x', y-y', etc.).
    """
    ndim = S.shape[0]
    V_inv = np.eye(ndim)
    if block_diag:
        V_inv = _norm_matrix(S, scale=scale)
    else:
        V_inv = _norm_matrix_block_diag(S, scale=scale)
    return V_inv


def _norm_matrix(S: np.ndarray, scale: bool = False) -> np.ndarray:
    ndim = S.shape[0]
    assert ndim % 2 == 0

    U = unit_symplectic_matrix(ndim)
    SU = np.matmul(S, U)
    eigvals, eigvecs = np.linalg.eig(SU)
    eigvecs = normalize_eigvecs(eigvecs)
    V_inv = norm_matrix_from_eigvecs(eigvecs)
    if scale:
        V = np.linalg.inv(V_inv)
        A = np.eye(ndim)
        for i in range(0, ndim, 2):
            emittance = rms_emittance(S[i : i + 2, i : i + 2])
            A[i : i + 2, i : i + 2] *= np.sqrt(emittance)
        V = np.matmul(V, A)
        V_inv = np.linalg.inv(V)
    return V_inv


def _norm_matrix_block_diag(S: np.ndarray, scale: bool = False) -> np.ndarray:
    ndim = S.shape[0]
    V_inv = np.eye(ndim)
    for i in range(0, ndim, 2):
        V_inv[i : i + 2, i : i + 2] = _norm_matrix(S[i : i + 2, i : i + 2], scale=scale)
    return V_inv


class CovarianceMatrix:
    def __init__(self, S: np.ndarray) -> None:
        self.S = S
        self.emittance = rms_emittance(S)
        self.projected_emittances = projected_emittances(S)
        self.intrinsic_emittances = intrinsic_emittances(S)
        self.twiss = twiss(S)

    def norm_matrix(self, **kwargs) -> np.ndarray:
        return norm_matrix(self.S, **kwargs)
