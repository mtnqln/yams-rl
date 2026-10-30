import tomllib
from functools import lru_cache
from typing import Any


@lru_cache
def get_config() -> dict[str, Any]:
    with open("config.toml", "rb") as f:
        return tomllib.load(f)
