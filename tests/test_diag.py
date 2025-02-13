import os
import numpy as np
import pytest

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import DriftTEAPOT
from orbit.teapot import QuadTEAPOT
from orbit.teapot import TEAPOT_Lattice

from orbit_tools.bunch import set_bunch_coords
from orbit_tools.diag import BunchHistogram


def test_hist():
    nbins = 100
    seed = 123
    
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(10_000, 6))

    bunch = Bunch()
    bunch.mass(0.938)
    bunch.getSyncParticle().kinEnergy(1.000)
    bunch.macroSize(1.0)
    bunch = set_bunch_coords(bunch, x)

    axis_list = []
    for i in range(6):
        axis_list.append((i,))
        
    for i in range(6):
        for j in range(i):
            axis_list.append((i, j))

    for axis in axis_list:
        ndim = len(axis)        
        shape = tuple(ndim * [nbins])
        limits = ndim * [(-5.0, 5.0)]        

        # Compute histogram using BunchHistogram
        hist = BunchHistogram(axis=axis, shape=shape, limits=limits)        
        values = hist.compute_histogram(bunch)
        values = values / np.max(values)
        
        # Compute histogram using NumPy
        values_np, _ = np.histogramdd(x[:, axis], bins=hist.edges)
        values_np = values_np / np.max(values_np)

        # Compare the histograms. There will be differences because Grid
        # classes use weighting.
        print(np.max(np.abs(values - values_np)))

    
    