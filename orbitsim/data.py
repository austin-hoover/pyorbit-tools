from typing import Union

import numpy as np
import pandas as pd


def read_bunch(path: str, dataframe: bool = False) -> Union[np.ndarray, pd.DataFrame]:
    """Read `bunch.dumpBunch(path)` as Numpy array or DataFrame."""
    dims = ["x", "xp", "y", "yp", "z", "de"]
    data = pd.read_table(
        path, 
        sep=" ", 
        comment="%", 
        usecols=range(len(dims)), 
        names=dims
    )
    if not dataframe:
        return data.values
    return data