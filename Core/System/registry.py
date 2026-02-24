from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PersonaProfile:
    """Represents a persona/    that shares Elysia's inner engine."""

    key: str
    title: str
    archetype: str
    description: str
    focus_fields: List[str] = field(default_factory=list)
    expression_channels: List[str] = field(default_factory=list)
    default_scripts: List[str] = field(default_factory=list)

    def as_payload(self) -> Dict[str, object]:
        """Return a serializable payload for UI or external engines."""
        return {
            "key": self.key,
            "title": self.title,
            "archetype": self.archetype,
            "description": self.description,
            "focus_fields": list(self.focus_fields),
            "expression_channels": list(self.expression_channels),
            "default_scripts": list(self.default_scripts),
        }


PERSONA_REGISTRY: Dict[str, PersonaProfile] = {
    "elysia.artist": PersonaProfile(
        key="elysia.artist",
        title="           ",
        archetype="Artist / Painter",
        description=(
            "CellWorld                              "
            "        . Value-Mass                   ."
        ),
        focus_fields=["value_mass_field", "intention_field"],
        expression_channels=["digital_canvas", "animation_cues"],
        default_scripts=["scripts/persona_hooks/artist_palette.py"],
    ),
    "elysia.dancer": PersonaProfile(
        key="elysia.dancer",
        title="               ",
        archetype="Dancer / Performer",
        description=(
            "Will Field  rhythm telemetry        /         "
            "    . VTuber rig                     ."
        ),
        focus_fields=["will_field", "value_mass_field"],
        expression_channels=["motion_capture", "vtuber_avatar"],
        default_scripts=["scripts/persona_hooks/dancer_flow.py"],
    ),
    "elysia.engineer": PersonaProfile(
        key="elysia.engineer",
        title="         ",
        archetype="Engineer / Architect",
        description=(
            "Concept OS, nano_core, curriculum                "
            "                . caretakers          "
            "               ."
        ),
        focus_fields=["concept_kernel", "curriculum_engine", "logs"],
        expression_channels=["notebook", "shell", "code_editor"],
        default_scripts=["scripts/persona_hooks/engineer_notebook.py"],
    ),
}


def list_personas() -> List[PersonaProfile]:
    """Return all persona profiles."""
    return list(PERSONA_REGISTRY.values())


def get_persona(key: str) -> PersonaProfile:
    """Retrieve a persona by key, raising KeyError if missing."""
    if key not in PERSONA_REGISTRY:
        raise KeyError(f"Unknown persona '{key}'")
    return PERSONA_REGISTRY[key]


def activate_persona(
    key: str, *, overrides: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """Return an activation payload for downstream routers.

    Downstream components (UI, VTuber rig, scripting layer) can call
    this helper to obtain a ready-to-use context bundle. The overrides
    parameter allows callers to inject runtime-specific data such as
    session IDs, animation seeds, or caretaker instructions.
    """
    persona = get_persona(key)
    payload = persona.as_payload()
    payload.update(
        {
            "session_state": {
                "persona_key": persona.key,
                "focus_fields": persona.focus_fields,
            }
        }
    )
    if overrides:
        payload.update(overrides)
    return payload
