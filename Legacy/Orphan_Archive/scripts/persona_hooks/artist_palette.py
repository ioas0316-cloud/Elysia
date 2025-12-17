"""Persona hook utilities for visualization-friendly payloads."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class PersonaFrame:
    """Minimal state that Godot / UI clients can consume."""

    mood_color: str
    energy_level: float
    will_rhythm: float
    focus_target: str
    caption: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def value_mass_to_color(value_mass: float) -> str:
    """Map normalized value mass to a hex color."""
    v = _clamp(value_mass)
    if v < 0.33:
        return "#4466ff"
    if v < 0.66:
        return "#88bbff"
    return "#ffd26a"


def will_field_to_rhythm(will_tension: float) -> float:
    """Map will tension to an animation tempo."""
    return _clamp(will_tension)


def build_persona_frame(world_state: Dict[str, Any]) -> PersonaFrame:
    """Convert a raw world_state dictionary to a PersonaFrame."""
    vm = float(world_state.get("value_mass_avg", 0.5))
    wt = float(world_state.get("will_tension_avg", 0.5))
    focus = str(world_state.get("focus_node", "unknown"))
    mode = str(world_state.get("mode", "idle"))

    return PersonaFrame(
        mood_color=value_mass_to_color(vm),
        energy_level=_clamp(vm),
        will_rhythm=will_field_to_rhythm(wt),
        focus_target=focus,
        caption=f"{mode} @ {focus}",
    )
