import os
import sys
from typing import Union


def get_input_dir(timestamp: Union[int, str], script_name: str, outputs_dir: str = "outputs") -> str:
    input_dir = None
    if timestamp is None:
        # Get latest run
        input_dirs = os.listdir(os.path.join(outputs_dir, script_name))
        input_dirs = [input_dir for input_dir in input_dirs if not input_dir.startswith(".")]
        input_dirs = sorted(input_dirs)
        input_dir = input_dirs[-1]
        input_dir = os.path.join(outputs_dir, script_name, input_dir)
    else:
        timestamp = str(timestamp)
        input_dir = os.path.join(outputs_dir, script_name, timestamp)
    return input_dir
