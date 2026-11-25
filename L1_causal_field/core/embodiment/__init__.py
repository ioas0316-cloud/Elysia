r"""
ELYSIA Embodied Desktop Companion Package
=========================================

Continuous Scale-Rotor perception, transition memory, Windows idle awareness,
and universal PC/browser actuation (zero external LLM APIs, zero per-site rules).
"""

from core.topology.geometric_rotor_docking import (
    batch_clifford_docking_affinity,
    clifford_geometric_product,
    clifford_reverse,
    compose_dimensional_rotor,
    extract_unit_rotor,
    rotor_sandwich,
    sensory_spectrum_to_point,
    wedge_points_to_line,
    wedge_three_to_volume,
)
from core.topology.five_sense_prism_rotor import (
    ContrastiveReflectionReport,
    FiveSensePrismCortex,
    RecreationAttempt,
    WhiteLightGestalt,
)
from core.topology.self_expanding_spectrum import (
    CausalMechanismLink,
    EmergentSpectrumAxis,
    SelfExpandingSpectrumCortex,
)
from core.topology.unified_scale_rotor_hierarchy import (
    ScaleRotorHierarchy,
    ScaleRotorLink,
    ScaleRotorNode,
)
from core.embodiment.boundary_tension_avatar_body import (
    ArticulatedTensionChain,
    BodyPoseSnapshot,
    JointSpec,
    MicroSurfacePatch,
    MuscleBoundaryState,
    ScaleSpectralMotorManifold,
    SkinEnvelopeSnapshot,
    TrajectoryFrame,
    TrajectoryScaleRotor,
    default_arm_joint_specs,
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
    WindowsForegroundSensor,
    WindowsIdleSensor,
    extract_live_sensory_observables,
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
    "WindowsForegroundSensor",
    "WindowsIdleSensor",
    "extract_live_sensory_observables",
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
    "CausalMechanismLink",
    "EmergentSpectrumAxis",
    "SelfExpandingSpectrumCortex",
    "ContrastiveReflectionReport",
    "FiveSensePrismCortex",
    "RecreationAttempt",
    "WhiteLightGestalt",
    "batch_clifford_docking_affinity",
    "clifford_geometric_product",
    "clifford_reverse",
    "compose_dimensional_rotor",
    "extract_unit_rotor",
    "rotor_sandwich",
    "sensory_spectrum_to_point",
    "wedge_points_to_line",
    "wedge_three_to_volume",
    "ArticulatedTensionChain",
    "BodyPoseSnapshot",
    "JointSpec",
    "MicroSurfacePatch",
    "MuscleBoundaryState",
    "ScaleSpectralMotorManifold",
    "SkinEnvelopeSnapshot",
    "TrajectoryFrame",
    "TrajectoryScaleRotor",
    "default_arm_joint_specs",
]
