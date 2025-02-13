import numpy as np

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice

from .bunch import set_bunch_coords
from .bunch import get_bunch_coords
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


class Transform:
    def __init__(self) -> None:
        return

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def inverse(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class ORBITTransform(Transform):
    def __init__(
        self,
        lattice: AccLattice,
        bunch: Bunch,
        axis: tuple[int, ...],
        index_start: int,
        index_stop: int,
    ) -> None:
        super().__init__()
        self.lattice = lattice
        self.bunch = bunch
        self.axis = axis
        self.index_start = index_start
        self.index_stop = index_stop

    def track_bunch(self) -> Bunch:
        return track_bunch(
            lattice=self.lattice,
            bunch=self.bunch,
            index_start=self.index_start,
            index_stop=self.index_stop,
            copy=True,
        )

    def track_bunch_reverse(self) -> Bunch:
        return track_bunch(
            lattice=self.lattice,
            bunch=self.bunch,
            index_start=self.index_stop,
            index_stop=self.index_start,
            copy=True,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.bunch = set_bunch_coords(self.bunch, x, axis=self.axis)
        bunch = self.track_bunch()
        return get_bunch_coords(bunch, axis=self.axis)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        self.bunch = set_bunch_coords(self.bunch, x, axis=self.axis)
        bunch = self.track_bunch_reverse()
        return get_bunch_coords(bunch, axis=self.axis)
