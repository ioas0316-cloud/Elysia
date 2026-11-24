r"""
ELYSIA Embodied Desktop Companion Package
=========================================

Continuous Scale-Rotor perception, transition memory, Windows idle awareness,
and universal PC/browser actuation (zero external LLM APIs, zero per-site rules).
"""

from core.topology.unified_scale_rotor_hierarchy import (
    ScaleRotorHierarchy,
    ScaleRotorLink,
    ScaleRotorNode,
)
from core.embodiment.language_scale_rotor import (
    ExpressiveDeficit,
    LanguageScaleRotor,
    bind_korean_operator,
    compose_hangul_jamo,
    decompose_hangul_jamo,
    has_jongseong,
)
from core.embodiment.pc_sensory_motor_body import (
    AgencyMode,
    ScreenFrameSnapshot,
    ScreenRetinaSensor,
    UniversalMotorActuator,
    WindowsIdleSensor,
)
from core.embodiment.scale_rotor_embodied_field import (
    CausalTransitionRotor,
    ObservationRotor,
    ScaleRotorEmbodiedMind,
    ScaleRotorTransitionMemory,
)
from core.embodiment.world_expression_ingestor import (
    SituatedExpressionField,
    SituatedExpressionRotor,
    extract_dialogues_from_literature,
    fetch_public_literature_from_web,
    ingest_local_world_media,
    parse_srt_or_vtt_subtitles,
)

__all__ = [
    "AgencyMode",
    "ScreenFrameSnapshot",
    "ScreenRetinaSensor",
    "UniversalMotorActuator",
    "WindowsIdleSensor",
    "CausalTransitionRotor",
    "ObservationRotor",
    "ScaleRotorEmbodiedMind",
    "ScaleRotorTransitionMemory",
    "SituatedExpressionField",
    "SituatedExpressionRotor",
    "extract_dialogues_from_literature",
    "fetch_public_literature_from_web",
    "ingest_local_world_media",
    "parse_srt_or_vtt_subtitles",
    "ExpressiveDeficit",
    "LanguageScaleRotor",
    "bind_korean_operator",
    "compose_hangul_jamo",
    "decompose_hangul_jamo",
    "has_jongseong",
    "ScaleRotorHierarchy",
    "ScaleRotorLink",
    "ScaleRotorNode",
]
