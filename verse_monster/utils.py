from typing import *

import yaml


def save_yaml(obj: Any, filename: str):
    with open(filename, 'w') as f:
        yaml.dump(obj, f)