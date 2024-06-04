import os
import pathlib
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm import tqdm
from tqdm import trange

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode

from orbitsim.models.sns import SNS_LINAC


@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(cfg : DictConfig) -> None:

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    print(OmegaConf.to_yaml(cfg))
    print("output_dir:", output_dir)


    print(output_dir)

if __name__ == "__main__":
    main()