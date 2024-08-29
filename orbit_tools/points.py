from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import numpy as np


def transform(points: np.ndarray, function: Callable, **kws) -> np.ndarray:
    return np.apply_along_axis(lambda x: function(x, **kws), 1, points)


def transform_linear(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.matmul(points, matrix.T)


def get_radii(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points, axis=1)


def get_ellipsoid_radii(points: np.ndarray) -> np.ndarray:
    cov = np.cov(points.T)
    cov_inv = np.linalg.inv(cov)
    function = lambda x: np.sqrt(np.linalg.multi_dot([x.T, cov_inv, x]))
    return transform(points, function)


def get_enclosing_sphere_radius(points: np.ndarray, fraction: float = 1.0) -> float:
    radii = get_radii(points)
    radii = np.sort(radii)
    index = int(np.round(points.shape[0] * fraction)) - 1
    radius = radii[index]
    return radius


def get_enclosing_ellipsoid_radius(points: np.ndarray, fraction: float = 1.0) -> float:
    radii = get_ellipsoid_radii(points)
    radii = np.sort(radii)
    index = int(np.round(points.shape[0] * fraction)) - 1
    radius = radii[index]
    return radius


def norm_2d(points: np.ndarray, scale_emittance: bool = False) -> np.ndarray:
    cov = np.cov(point.T)
    eps = np.sqrt(np.linalg.det(cov))
    alpha = -cov[0, 1] / eps
    beta = cov[0, 0] / eps

    points_n = np.zeros(points.shape)
    points_n[:, 0] = points[:, 0] / np.sqrt(beta)
    points_n[:, 1] = np.sqrt(beta) * X[:, 1] + (alpha / np.sqrt(beta)) * X[:, 0]
    if scale_emittance:
        points_n = points_n / np.sqrt(eps)
    return points_n


def slice_idx(
    points: np.ndarray, 
    axis: Tuple[int] = None, 
    center: np.ndarray = None,
    width: np.ndarray = None,
    limits: List[Tuple[float]] = None,
) -> np.ndarray:
    if axis is None:
        axis = np.arange(points.shape[1])
    if limits is None:
        limits = list(zip(center - 0.5 * width, center + 0.5 * width))  
    limits = np.array(limits)
    if limits.ndim == 0:
        limits = limits[None, :]
    conditions = []
    for j, (xmin, xmax) in zip(axis, limits):
        conditions.append(X[:, j] > xmin)
        conditions.append(X[:, j] < xmax)
    idx = np.logical_and.reduce(conditions)
    

def slice(
    points: np.ndarray, 
    axis: Tuple[int] = None, 
    center: np.ndarray = None,
    width: np.ndarray = None,
    limits: List[Tuple[float]] = None,
) -> np.ndarray:
    idx = slice_idx(points, axis, center=center, width=width, limits=limits)
    return points[idx, :]


def slice_idx_sphere(
    points: np.ndarray, 
    axis: Tuple[int] = None, 
    rmin: float = 0.0, 
    rmax: float = None,
) -> np.ndarray:
    if axis is None:
        axis = np.arange(points.shape[1])
    if rmax is None:
        rmax = np.inf
    radii = get_radii(points[:, axis])
    idx = np.logical_and(radii > rmin, radii < rmax)
    return idx


def slice_sphere(
    points: np.ndarray, 
    axis: Tuple[int] = None, 
    rmin: float = 0.0, 
    rmax: float = None
) -> np.ndarray:
    idx = slice_idx_sphere(points, axis, rmin=rmin, rmax=rmax)
    return points[idx, :]


def slice_idx_ellipsoid(
    points: np.ndarray, 
    axis: Tuple[int] = None, 
    rmin: float = 0.0, 
    rmax: float = None,
) -> np.ndarray:
    if axis is None:
        axis = np.arange(points.shape[1])
    if rmax is None:
        rmax = np.inf
    radii = get_ellipsoid_radii(points[:, axis])
    idx = np.logical_and(radii > rmin, radii < rmax)
    return idx


def slice_ellipsoid(
    points: np.ndarray, 
    axis: Tuple[int] = None, 
    rmin: float = 0.0, 
    rmax: float = None
) -> np.ndarray:
    idx = slice_idx_ellipsoid(points, axis, rmin=rmin, rmax=rmax)
    return points[idx, :]
    

def slice_idx_contour(
    points: np.ndarray, 
    axis: Tuple[int] = None, 
    lmin: float = 0.0, 
    lmax: float = 1.0, 
    interp: bool = True, 
    hist_kws: dict = None,
) -> np.ndarray:
    if axis is None:
        axis = np.arange(points.shape[1])

    _points = points[:, axis]

    if hist_kws is None:
        hist_kws = dict()
        
    hist, edges = np.histogramdd(_points, **hist_kws)
    hist = hist / np.max(hist)
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]
    
    if interp:
        interpolator = scipy.interpolate.RegularGridInterpolator(
            centers,
            hist,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        values = interpolator(_points)
        idx = np.logical_and(values >= lmin, values <= lmax)
        return ids
    else:
        valid_indices = np.stack(
            np.where(np.logical_and(hist >= lmin, hist <= lmax)), 
            axis=-1
        )
        indices = [np.digitize(_points[:, j], edges[j]) for j in range(len(axis))]
        indices = np.stack(indices, axis=-1)
        idx = []
        for i in range(len(indices)):
            if indices[i].tolist() in valid_indices.tolist():
                idx.append(i)
        return idx

