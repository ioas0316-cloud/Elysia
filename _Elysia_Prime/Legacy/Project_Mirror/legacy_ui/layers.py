# [Genesis: 2025-12-02] Purified by Elysia
from __future__ import annotations

from collections import OrderedDict


LAYERS = OrderedDict({
    "agents": True,
    "structures": True,
    "flora": True,
    "fauna": True,
    "will_field": True,
})


def set_visible(name: str, on: bool) -> None:
    if name in LAYERS:
        LAYERS[name] = bool(on)


def is_visible(name: str) -> bool:
    return bool(LAYERS.get(name, False))


def toggle(name: str) -> None:
    if name in LAYERS:
        LAYERS[name] = not LAYERS[name]
