"""Unified configuration access for the early sepsis prediction project."""

from dataclasses import dataclass
from types import ModuleType

from src.config import constants, paths, settings


@dataclass(frozen=True)
class Configuration:
    """Container exposing project constants, paths, and settings modules."""

    constants: ModuleType = constants
    paths: ModuleType = paths
    settings: ModuleType = settings


config = Configuration()
