import numpy as np

import orbit.lattice
import orbit.teapot
from orbit.core.bunch import Bunch

from model import SNS_RING


lattice = SNS_RING()
lattice.read_lattice_file(
    path="./inputs/lattice/sns_ring_nux6.175_nuy6.175_sol.lattice", 
    sequence="rnginjsol",
    kind="madx",
)
