"""
elysia_core Package Initialization
Re-exports primary interfaces from core/
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

__all__ = [
    "ComplexPhasor",
    "PhasorField",
    "PrismBifurcationOperator",
    "DynamicMacroRotorStack",
    "MacroRotor",
    "RetrocausalExecutionLoop",
    "ReversibleWaveTransformation",
    "FractalResonanceUnit",
    "UnifiedSystemArchitecture",
]
