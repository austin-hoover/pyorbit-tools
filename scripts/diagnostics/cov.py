"""Compute bunch covariance matrix."""
import os
import sys

import numpy as np

from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis



# Generate a normal distribution. Note: since ORBIT accounts for dispersion, x-dE 
# (0-5) or y-dE (2-5) correlations will change the computed Twiss parameters.
rng = np.random.default_rng(15520)

cov = np.identity(6)
for (i, j) in [(0, 1), (2, 3), (4, 5), (0, 2)]: 
    cov[i, j] = cov[j, i] = rng.uniform(-0.8, 0.8)
cov *= 100.0

coords = rng.multivariate_normal(np.zeros(6), cov, size=100_000)

bunch = Bunch()
for (x, xp, y, yp, z, de) in coords:
    bunch.addParticle(x, xp, y, yp, z, de)

# Compute covariance matrix moments in ORBIT
bunch_twiss_analysis = BunchTwissAnalysis()
bunch_twiss_analysis.analyzeBunch(bunch)
order = 2
dispersion_flag = 0
emit_norm_flag = 0
bunch_twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)
cov = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        cov[i, j] = bunch_twiss_analysis.getCorrelation(i, j)

# Compute covariance matrix in NumPy.
cov_np = np.cov(coords.T)


print("Second-order moments")
print("--------------------")
dims = ["x", "x'", "y", "y'", "z", "z'"]
for i in range(6):
    for j in range(6):
        print("<{}{}>:".format(dims[i], dims[j]))
        print("  numpy = {}".format(cov_np[i, j]))
        print("  orbit = {}".format(cov[i, j]))
print()
print("Twiss parameters")
print("----------------")
for i, dim in enumerate(["x", "y", "z"]):
    j = 2 * i
    np_eps = np.sqrt(cov_np[j, j] * cov_np[j + 1, j + 1] - cov_np[j, j + 1] ** 2)
    np_beta = cov_np[j, j] / np_eps
    np_alpha = -cov_np[j, j + 1] / np_eps
    np_gamma = cov_np[j + 1, j + 1] / np_eps
    alpha, beta, gamma, eps = bunch_twiss_analysis.getTwiss(i)
    print("alpha_{}:".format(dim))
    print("  numpy = {}".format(np_alpha))
    print("  orbit = {}".format(alpha))
    print("beta_{}:".format(dim))
    print("  numpy = {}".format(np_beta))
    print("  orbit = {}".format(beta))
    print("gamma_{}:".format(dim))
    print("  numpy = {}".format(np_gamma))
    print("  orbit = {}".format(gamma))
    print("eps_{}:".format(dim))
    print("  numpy = {}".format(np_eps))
    print("  orbit = {}".format(eps))