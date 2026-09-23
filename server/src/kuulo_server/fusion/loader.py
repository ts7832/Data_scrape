"""Load a fusion engine from a "module:Class" string, so a private engine can be dropped in."""

from __future__ import annotations

from importlib import import_module

from .base import FusionEngine


def load_engine(spec: str) -> FusionEngine:
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"fusion engine must look like 'module:Class', got {spec!r}")
    return getattr(import_module(module_name), class_name)()
