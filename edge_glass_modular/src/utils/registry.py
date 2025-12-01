"""Simple registry to decouple experiment configs from implementations."""

from __future__ import annotations

from typing import Any, Callable, Dict


class Registry:
    def __init__(self):
        self._fns: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def decorator(fn: Callable[..., Any]):
            self._fns[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._fns:
            raise KeyError(f"{name} not found in registry")
        return self._fns[name]


ENCODER_REGISTRY = Registry()
DECODER_REGISTRY = Registry()
