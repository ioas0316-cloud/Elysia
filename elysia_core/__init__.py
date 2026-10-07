"""
elysia_core Package Initialization
Re-exports primary interfaces from core/
"""

from core.physics.quantum_rotor_phase import Clifford5DRotorEngine, LandauGinzburgPotentialEngine, QuantumRotorPhaseIntegrator
from core.physics.tensor_helmholtz_decomposer import TensorHelmholtzMetricDecomposer, MetricWaveDecompositionResult
from core.physics.metric_rotor_pipeline import IntegratedMetricRotorPipeline, MetricToRotorBridge
from core.physics.trainable_metric_rotor_loop import TrainableMetricRotorPipeline
from core.memory.scale_wave_tensor_memory import ScaleWaveTensorMemory
from core.memory.scale_wave_autograd import TrainableScaleWaveMemory, ScaleWaveMemoryAutogradFunction
from core.embodied.swarm_lift_field import SwarmLiftField, DroneAgentState

__all__ = [
    "Clifford5DRotorEngine",
    "LandauGinzburgPotentialEngine",
    "QuantumRotorPhaseIntegrator",
    "TensorHelmholtzMetricDecomposer",
    "MetricWaveDecompositionResult",
    "IntegratedMetricRotorPipeline",
    "MetricToRotorBridge",
    "TrainableMetricRotorPipeline",
    "ScaleWaveTensorMemory",
    "TrainableScaleWaveMemory",
    "ScaleWaveMemoryAutogradFunction",
    "SwarmLiftField",
    "DroneAgentState",
]
