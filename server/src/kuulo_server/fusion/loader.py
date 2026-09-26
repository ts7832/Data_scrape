"""Load a plugin from a "module:Class" string, so a private engine or locator can be dropped in."""

from __future__ import annotations

from importlib import import_module

from .base import FusionEngine


def load_plugin(spec: str, *args):
    """Instantiate "module:Class" with args, so a private engine or locator can be dropped in."""
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"plugin must look like 'module:Class', got {spec!r}")
    return getattr(import_module(module_name), class_name)(*args)


def load_engine(spec: str) -> FusionEngine:
    return load_plugin(spec)
