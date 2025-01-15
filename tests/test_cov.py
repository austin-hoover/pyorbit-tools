import pytest
import numpy as np

import orbit_tools as ot


def test_rms_ellipsoid_volume_identity():
    S = np.eye(2)
    assert np.isclose(ot.cov.rms_ellipsoid_volume(S), 1.0)


def test_rms_ellipsoid_volume_diagonal():
    S = np.diag([2.0, 3.0])
    assert np.isclose(ot.cov.rms_ellipsoid_volume(S), np.sqrt(6.0))


def test_projected_emittances_identity():
    S = np.eye(4)
    assert np.allclose(ot.cov.projected_emittances(S), [1.0, 1.0])


def test_intrinsic_emittances_identity():
    S = np.eye(4)
    assert np.allclose(ot.cov.intrinsic_emittances(S), [1.0, 1.0])


def test_twiss_identity():
    S = np.eye(4)
    twiss_params = ot.cov.twiss(S)
    assert np.allclose(twiss_params, [(0.0, 1.0, 1.0), (0.0, 1.0, 1.0)])


def test_unit_symplectic_matrix():
    U = ot.cov.unit_symplectic_matrix(4)
    expected = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    assert np.allclose(U, expected)


def test_normalization_matrix_from_twiss_2d():
    alpha = 0.0
    beta = 1.0
    emittance = 1.0
    V_inv = ot.cov.normalization_matrix_from_twiss_2d(alpha, beta, emittance)
    expected = np.eye(2)
    assert np.allclose(V_inv, expected)


def test_cov_to_corr_identity():
    S = np.eye(2)
    assert np.allclose(ot.cov.cov_to_corr(S), S)


def test_rms_ellipse_params():
    S = np.eye(2)
    cx, cy, theta = ot.cov.rms_ellipse_params(S)
    assert np.isclose(cx, 1.0)
    assert np.isclose(cy, 1.0)
    assert np.isclose(theta, 0.0)
