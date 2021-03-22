import itertools
import time
from typing import *

import cloudpickle
from sklearn.model_selection import train_test_split
import yaml


def save_yaml(obj: Any, filename: str):
    with open(filename, 'w') as f:
        yaml.dump(obj, f)


def load_yaml(filename: str):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def save_cloudpickle(obj: Any, filename: str):
    with open(filename, 'wb') as f:
        cloudpickle.dump(obj, f)


def load_cloudpickle(filename: str):
    with open(filename, 'rb') as f:
        return cloudpickle.load(f)


def train_valid_test_split(list_of_datapoints, seed=1234, p_valid=0.1, p_test=0.1):
    num_data_total = len(list_of_datapoints)

    num_valid = int(num_data_total * p_valid)
    num_test = int(num_data_total * p_test)

    train, valid = train_test_split(list_of_datapoints, test_size=num_valid, random_state=seed)
    train, test = train_test_split(train, test_size=num_test, random_state=seed + 1)

    return train, valid, test


def flatten_list_of_lists(lol: List[List[Any]]):
    return list(itertools.chain(*lol))


class Timer:
    TIME_FORMAT = '%H:%M:%S'

    def __init__(self, name: str, do_print_outputs=True):
        self.name = name
        self.name_str = '' if not self.name else f' "{self.name}"'
        self.do_print_outputs = do_print_outputs

    def __enter__(self):
        self.t = time.time()
        time_str = time.strftime(self.TIME_FORMAT, time.localtime(self.t))
        if self.do_print_outputs:
            print(f'Starting timer{self.name_str} at time {time_str}.', end=" ")

    def __exit__(self, *args):
        if self.do_print_outputs:
            print(f'Timer{self.name_str} took {time.time() - self.t:2.3g} secs.')
