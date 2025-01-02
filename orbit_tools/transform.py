import numpy as np

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice

from .bunch import set_bunch_coords
from .bunch import get_bunch_coords
from .sim import track_bunch


class Transform:
    def __init__(self) -> None:
        return

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def inverse(self, U: np.ndarray) -> np.ndarray:
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

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.bunch = set_bunch_coords(self.bunch, X, axis=self.axis)
        bunch = self.track_bunch()
        return get_bunch_coords(bunch, axis=self.axis)

    def inverse(self, X: np.ndarray) -> np.ndarray:
        self.bunch = set_bunch_coords(self.bunch, X, axis=self.axis)
        bunch = self.track_bunch_reverse()
        return get_bunch_coords(bunch, axis=self.axis)
