from orbit.core.bunch import Bunch

import orbit.lattice
import orbit.teapot



lattice_filename = "sns_ring_nux6.175_nuy6.175_sol.lattice"
lattice_sequence = "rnginjsol"

lattice = orbit.teapot.TEAPOT_Ring()
lattice.readMADX(lattice_filename, lattice_sequence)

