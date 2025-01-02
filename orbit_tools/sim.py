from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice

from .bunch import reverse_bunch


def track_bunch(
    bunch: Bunch,
    lattice: AccLattice,
    index_start: int = None,
    index_stop: int = None,
    copy: bool = False,
    **kwargs
) -> Bunch:

    if index_start is None:
        index_start = 0

    if index_stop is None:
        index_stop = len(lattice.getNodes()) - 1

    reverse = index_start > index_stop
    node_start = lattice.getNodes()[index_start]
    node_stop = lattice.getNodes()[index_stop]

    bunch_out = None
    if copy:
        bunch_out = Bunch()
        bunch.copyBunchTo(bunch_out)
    else:
        bunch_out = bunch

    if reverse:
        bunch_out = reverse_bunch(bunch_out)
        lattice.reverseOrder()

    lattice.trackBunch(
        bunch_out,
        index_start=lattice.getNodeIndex(node_start),
        index_stop=lattice.getNodeIndex(node_stop),
        **kwargs
    )

    if reverse:
        bunch_out = reverse_bunch(bunch_out)
        lattice.reverseOrder()

    return bunch_out
