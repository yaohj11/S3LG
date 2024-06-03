import yaml
import os
import argparse

__all__ = [
    'load_config', 'config_model', 'config_dataset',
]

def load_config(path: str, with_prefix=False) -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    if with_prefix:
        path = os.path.join(os.path.dirname(__file__), '..', path)
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def _select_dir(dir_list):
    for base_dir in dir_list:
        if isinstance(base_dir, str) and os.path.isdir(base_dir) and os.path.exists(base_dir):
            return base_dir
    return None


def _dict2namespace(x: dict) -> argparse.Namespace:
    return argparse.Namespace(**x)

