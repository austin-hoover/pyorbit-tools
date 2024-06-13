import numpy as np

from orbit.core.bunch import Bunch
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode

import orbitsim


acc_model = orbitsim.models.sns.SNS_LINAC()
lattice = acc_model.lattice

diag_node_names = [
    "HEBT_Diag:WS01",
    "HEBT_Diag:WS02",
    "HEBT_Diag:WS03",
    "HEBT_Diag:WS04",
    "HEBT_Diag:WS09",
]

bunch = Bunch()
bunch.mass(0.939294)
bunch.charge(-1.0)
bunch.getSyncParticle().kinEnergy(1.0012134710494787)
axis = (0, 1)

orbit_transforms = []
for diag_node_name in diag_node_names:
    orbit_transform = orbitsim.linac.LinacTransform(
        lattice=lattice, 
        bunch=bunch,
        index_start=lattice.getNodeIndex(lattice.getNodeForName(diag_node_names[0])),
        index_stop =lattice.getNodeIndex(lattice.getNodeForName(diag_node_name)),
        axis=axis,
        linear=True,
    )
    orbit_transforms.append(orbit_transform)

rng = np.random.default_rng(1234)
ub = np.array([+0.020, +0.020, +0.020, +0.020, +0.020, 0.0])
lb = -ub
X_true = rng.uniform(lb, ub, size=(100, 6))
X_true = X_true[:, axis]

for orbit_transform, diag_node_name in zip(orbit_transforms, diag_node_names):
    X = orbit_transform(X_true, linear=True)
    Y = orbit_transform(X_true, linear=False)
    print(diag_node_name)
    for i in range(3):
        delta = X[i, :] - Y[i, :]
        delta = delta * 1000.0
        print("(part {}) delta (mm, mrad) = [{:0.2e} {:0.2e}]".format(i, *delta))
