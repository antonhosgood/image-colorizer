from os import PathLike
from typing import AnyStr

import yaml


def load_config(path: PathLike[AnyStr] | AnyStr) -> dict:
    """Loads a YAML configuration file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
