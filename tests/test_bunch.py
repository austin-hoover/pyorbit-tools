import numpy as np

import orbit.core
import orbit.lattice
from orbit.core.bunch import Bunch

import orbitsim as pos


def make_gaussian_bunch(size: int = 128) -> Bunch:
    bunch = Bunch()
    for _ in range(size):
        x, xp, y, yp, z, de = np.random.normal(size=6)
        bunch.addParticle(x, xp, y, yp, z, de)
    return bunch


def test_set_current():
    bunch = make_gaussian_bunch()
    pos.bunch.set_current(bunch, current=0.025, frequency=402.5e+06)


def test_get_coords():
    bunch = make_gaussian_bunch()
    coords = pos.bunch.get_coords(bunch)
    for i in range(coords.shape[0]):
        assert bunch.x(i) == coords[i, 0]
        assert bunch.y(i) == coords[i, 2]
        assert bunch.z(i) == coords[i, 4]
        assert bunch.xp(i) == coords[i, 1]
        assert bunch.yp(i) == coords[i, 3]
        assert bunch.dE(i) == coords[i, 5]


def test_set_coords():
    coords = np.random.normal(size=(128, 6))
    bunch = Bunch()
    bunch = pos.bunch.set_coords(bunch, coords)
    assert np.all(coords == pos.bunch.get_coords(bunch))


def test_decorrelate_x_y_z():
    indices = [(0, 2)]

    cov = np.eye(6)
    for (i, j) in indices:
        cov[i, j] = cov[j, i] = 0.5

    coords = np.random.multivariate_normal(mean=np.zeros(6), cov=cov, size=1000)
    cov = np.cov(coords.T)

    bunch = Bunch()
    bunch = pos.bunch.set_coords(bunch, coords)
    bunch = pos.bunch.decorrelate_x_y_z(bunch)

    coords_out = pos.bunch.get_coords(bunch)
    cov_out = np.cov(coords_out.T)
    for (i, j) in indices:
        assert np.abs(cov_out[i, j]) < np.abs(cov[i, j])


def test_decorrelate_xy_z():
    indices = [(0, 4), (1, 5)]

    cov = np.eye(6)
    for (i, j) in indices:
        cov[i, j] = cov[j, i] = 0.5

    coords = np.random.multivariate_normal(mean=np.zeros(6), cov=cov, size=1000)
    cov = np.cov(coords.T)

    bunch = Bunch()
    bunch = pos.bunch.set_coords(bunch, coords)
    bunch = pos.bunch.decorrelate_xy_z(bunch)

    coords_out = pos.bunch.get_coords(bunch)
    cov_out = np.cov(coords_out.T)
    for (i, j) in indices:
        assert np.abs(cov_out[i, j]) < np.abs(cov[i, j])


def test_downsample():
    size = 1000
    new_size = 100
    bunch = make_gaussian_bunch()
    bunch = pos.bunch.downsample(bunch, new_size, conserve_intensity=True)
    new_bunch_size_global = bunch.getSizeGlobal()
    assert new_bunch_size_global == new_size


def test_revese_bunch():
    bunch = make_gaussian_bunch()
    bunch = pos.bunch.reverse(bunch)


def test_get_centroid():
    bunch = make_gaussian_bunch()
    centroid = pos.bunch.get_centroid(bunch)


def test_shift_centroid():
    bunch = make_gaussian_bunch()
    bunch = pos.bunch.shift_centroid(bunch, np.ones(6))


def test_set_centroid():
    bunch = make_gaussian_bunch()
    bunch = pos.bunch.set_centroid(bunch, np.ones(6))


def test_get_mean_and_covariance():
    bunch = make_gaussian_bunch()
    mean, cov = pos.bunch.get_mean_and_covariance(bunch)


def test_load_bunch():
    filename = None
    pos.bunch.load(filename=None)


def test_generate():
    def func():
        return np.zeros(6)

    bunch = Bunch()
    bunch = pos.bunch.generate(func, size=128, bunch=bunch)
    return bunch


def test_get_twiss_containers():
    bunch = make_gaussian_bunch()
    twiss_x, twiss_y, twiss_z = pos.bunch.get_twiss_containers(bunch)