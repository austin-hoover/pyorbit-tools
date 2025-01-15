"""Covariance matrix analysis."""

import numpy as np


def rms_ellipsoid_volume(S: np.ndarray) -> float:
    return np.sqrt(np.linalg.det(S))


def projected_emittances(S: np.ndarray) -> tuple[float, ...]:
    emittances = []
    for i in range(0, S.shape[0], 2):
        emittance = rms_ellipsoid_volume(S[i : i + 2, i : i + 2])
        emittances.append(emittance)
    return emittances


def intrinsic_emittances(S: np.ndarray) -> tuple[float, ...]:
    S = S[:4, :4].copy()  # [to do]: expand to NxN
    U = np.array(
        [
            [+0.0, +1.0, +0.0, +0.0],
            [-1.0, +0.0, +0.0, +0.0],
            [+0.0, +0.0, +0.0, +1.0],
            [+0.0, +0.0, -1.0, +0.0],
        ]
    )
    tr_SU2 = np.trace(np.linalg.matrix_power(np.matmul(S, U), 2))
    det_S = np.linalg.det(S)
    eps_1 = 0.5 * np.sqrt(-tr_SU2 + np.sqrt(tr_SU2**2 - 16.0 * det_S))
    eps_2 = 0.5 * np.sqrt(-tr_SU2 - np.sqrt(tr_SU2**2 - 16.0 * det_S))
    return (eps_1, eps_2)


def twiss_2d(S: np.ndarray) -> tuple[float, float, float]:
    """Compute twiss parameters from 2 x 2 covariance matrix.

    alpha = -<xx'> / sqrt(<xx><x'x'> - <xx'><xx'>)
    beta  =  <xx>  / sqrt(<xx><x'x'> - <xx'><xx'>)
    """
    emittance = rms_ellipsoid_volume(S)
    beta = S[0, 0] / emittance
    alpha = -S[0, 1] / emittance
    return (alpha, beta, emittance)


def twiss(
    S: np.ndarray,
) -> tuple[float, float, float] | list[tuple[float, float, float]]:
    parameters = []
    for i in range(0, S.shape[0], 2):
        parameters.append(twiss_2d(S[i : i + 2, i : i + 2]))

    if len(parameters) == 1:
        parameters = parameters[0]

    return parameters


def unit_symplectic_matrix(ndim: int) -> np.ndarray:
    """Return matrix U such that, if M is a symplectic matrix, UMU^T = M."""
    U = np.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        U[i : i + 2, i : i + 2] = [[0.0, 1.0], [-1.0, 0.0]]
    return U


def normalize_eigvecs(eigvecs: np.ndarray) -> np.ndarray:
    """Normalize eigenvectors according to Lebedev-Bogacz convention."""
    ndim = eigvecs.shape[0]
    U = unit_symplectic_matrix(ndim)
    for i in range(0, ndim, 2):
        v = eigvecs[:, i]
        val = np.linalg.multi_dot([np.conj(v), U, v]).imag
        if val > 0.0:
            (eigvecs[:, i], eigvecs[:, i + 1]) = (eigvecs[:, i + 1], eigvecs[:, i])
        eigvecs[:, i : i + 2] *= np.sqrt(2.0 / np.abs(val))
    return eigvecs


def normalization_matrix_from_eigvecs(eigvecs: np.ndarray) -> np.ndarray:
    """Return normalization matrix V^-1 from eigenvectors."""
    V = np.zeros(eigvecs.shape)
    for i in range(0, V.shape[1], 2):
        V[:, i] = eigvecs[:, i].real
        V[:, i + 1] = (1.0j * eigvecs[:, i]).real
    return np.linalg.inv(V)


def normalization_matrix_from_twiss_2d(
    alpha: float, beta: float, emittance: float = None
) -> np.ndarray:
    """Return 2 x 2 normalization matrix V^-1 from Twiss parameters."""
    V = np.array([[beta, 0.0], [-alpha, 1.0]]) * np.sqrt(1.0 / beta)
    A = np.eye(2)
    if emittance is not None:
        A = np.sqrt(np.diag([emittance, emittance]))
    V = np.matmul(V, A)
    return np.linalg.inv(V)


def normalization_matrix_from_twiss(
    twiss_params: list[tuple[float, float, float]]
) -> np.ndarray:
    """2N x 2N block-diagonal normalization matrix from Twiss parameters.

    Parameters
    ----------
    twiss_params : list[tuple[float, float, float]]
        Twiss parameters (alpha, beta, emittance) in each dimension.

    Returns
    -------
    V : ndarray, shape (2N, 2N)
        Block-diagonal normalization matrix.
    """
    ndim = len(twiss_params) // 2
    V = np.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        V[i : i + 2, i : i + 2] = normalization_matrix_from_twiss_2d(
            *twiss_params[i : i + 2]
        )
    return np.linalg.inv(V)


def normalization_matrix(
    S: np.ndarray, scale: bool = False, block_diag: bool = False
) -> np.ndarray:
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

    def _normalization_matrix(S: np.ndarray, scale: bool = False) -> np.ndarray:
        U = unit_symplectic_matrix(ndim)
        SU = np.matmul(S, U)
        eigvals, eigvecs = np.linalg.eig(SU)
        eigvecs = normalize_eigvecs(eigvecs)
        V_inv = normalization_matrix_from_eigvecs(eigvecs)

        if scale:
            V = np.linalg.inv(V_inv)
            A = np.eye(ndim)
            for i in range(0, ndim, 2):
                emittance = rms_ellipsoid_volume(S[i : i + 2, i : i + 2])
                A[i : i + 2, i : i + 2] *= np.sqrt(emittance)
            V = np.matmul(V, A)
            V_inv = np.linalg.inv(V)

        return V_inv

    ndim = S.shape[0]
    V_inv = np.eye(ndim)
    if block_diag:
        V_inv = _normalization_matrix(S, scale=scale)
    else:
        for i in range(0, ndim, 2):
            V_inv[i : i + 2, i : i + 2] = _normalization_matrix(
                S[i : i + 2, i : i + 2], scale=scale
            )
    return V_inv


def cov_to_corr(S: np.ndarray) -> np.ndarray:
    """Compute correlation matrix from covariance matrix."""
    D = np.sqrt(np.diag(S.diagonal()))
    Dinv = np.linalg.inv(D)
    return np.linalg.multi_dot([Dinv, S, Dinv])


def rms_ellipse_params(
    S: np.ndarray, axis: tuple[int, ...] = None
) -> tuple[float, ...]:
    """Return projected rms ellipse dimensions and orientation.

    Parameters
    ----------
    S : ndarray, shape (2N, 2N)
        The phase space covariance matrix.
    axis : tuple[int]
        Projection axis. Example: if the axes are {x, xp, y, yp}, and axis=(0, 2),
        the four-dimensional ellipsoid is projected onto the x-y plane.

    Returns
    -------
    c1, c2 : float
        The ellipse semi-axis widths.
    angle : float
        The tilt angle below the x axis [radians].
    """
    if S.shape[0] == 2:
        axis = (0, 1)
    (i, j) = axis
    sii = S[i, i]
    sjj = S[j, j]
    sij = S[i, j]
    angle = -0.5 * np.arctan2(2 * sij, sii - sjj)
    _sin = np.sin(angle)
    _cos = np.cos(angle)
    _sin2 = _sin**2
    _cos2 = _cos**2
    c1 = np.sqrt(abs(sii * _cos2 + sjj * _sin2 - 2 * sij * _sin * _cos))
    c2 = np.sqrt(abs(sii * _sin2 + sjj * _cos2 + 2 * sij * _sin * _cos))
    return (c1, c2, angle)
