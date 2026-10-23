"""
elysia_core Package Initialization
Re-exports primary interfaces from core/ and elysia_core/
"""

try:
    from core.physics.quantum_rotor_phase import Clifford5DRotorEngine, LandauGinzburgPotentialEngine, QuantumRotorPhaseIntegrator
    from core.physics.tensor_helmholtz_decomposer import TensorHelmholtzMetricDecomposer, MetricWaveDecompositionResult
    from core.physics.metric_rotor_pipeline import IntegratedMetricRotorPipeline, MetricToRotorBridge
    from core.physics.trainable_metric_rotor_loop import TrainableMetricRotorPipeline
    from core.memory.scale_wave_tensor_memory import ScaleWaveTensorMemory
    from core.memory.scale_wave_autograd import TrainableScaleWaveMemory, ScaleWaveMemoryAutogradFunction
    from core.embodied.swarm_lift_field import SwarmLiftField, DroneAgentState
    from core.physics.prismatic_axis_spectrum import PrismaticAxisSpectrumEngine, MultiLayeredWaveMemoryStack
except ImportError:
    pass

from core.knowledge_base.knowledge_bus_slot import KnowledgeBusSlot, KnowledgeChunk
from elysia_core.dressmaker_phase_lock_runtime import (
    DressmakerPhaseLockRuntime,
    FabricWaveSpectrum,
    CharacterBodyRotor,
)

from elysia_core.unified_system_architecture import (
    ComplexPhasor,
    PhasorField,
    PrismBifurcationOperator,
    DynamicMacroRotorStack,
    MacroRotor,
    RetrocausalExecutionLoop,
    ReversibleWaveTransformation,
    FractalResonanceUnit,
    UnifiedSystemArchitecture,
)
from elysia_core.unified_topological_phase_runtime import (
    SelfWorldBoundary,
    EnactiveReachGrid,
    FractalPhaseNode3x3x3,
    UnifiedTopologicalPhaseRuntime,
)
from elysia_core.perceptual_probe_skill import (
    PerceptualProbeSkill,
    RelationalVoidProbe,
    SkillAdaptabilityMap,
    PerceptualProbeEngine,
    ProbeFeedback,
)

__all__ = [
    "KnowledgeBusSlot",
    "KnowledgeChunk",
    "DressmakerPhaseLockRuntime",
    "FabricWaveSpectrum",
    "CharacterBodyRotor",
    "ComplexPhasor",
    "PhasorField",
    "PrismBifurcationOperator",
    "DynamicMacroRotorStack",
    "MacroRotor",
    "RetrocausalExecutionLoop",
    "ReversibleWaveTransformation",
    "FractalResonanceUnit",
    "UnifiedSystemArchitecture",
    "SelfWorldBoundary",
    "EnactiveReachGrid",
    "FractalPhaseNode3x3x3",
    "UnifiedTopologicalPhaseRuntime",
    "PerceptualProbeSkill",
    "RelationalVoidProbe",
    "SkillAdaptabilityMap",
    "PerceptualProbeEngine",
    "ProbeFeedback",
]
