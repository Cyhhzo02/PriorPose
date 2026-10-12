"""Small YAML configuration loader with attribute access."""

from pathlib import Path

import yaml


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    __setattr__ = dict.__setitem__


def _convert(value):
    if isinstance(value, dict):
        return Config({key: _convert(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_convert(item) for item in value]
    return value


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    config = _convert(payload)
    config.config_path = str(path)
    return config
