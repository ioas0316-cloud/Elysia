"""Elysia Core - Physics / Kinematics Subsystem"""

from core.physics.counterfactual_branching import CounterfactualBranchingEngine
from core.physics.frame_controller import ObservationalFrameController
from core.physics.constructive_causal_spacetime import (
    ConstructiveSpacetimeAxis,
    ConstructiveLogicDiscriminator,
    HierarchicalScaleCoupler,
)
from core.physics.quantum_rotor_phase import (
    Clifford5DRotorEngine,
    LandauGinzburgPotentialEngine,
    QuantumRotorPhaseIntegrator
)
from core.physics.tensor_helmholtz_decomposer import (
    TensorHelmholtzMetricDecomposer,
    MetricWaveDecompositionResult
)
from core.physics.metric_rotor_pipeline import (
    IntegratedMetricRotorPipeline,
    MetricToRotorBridge
)
from core.physics.trainable_metric_rotor_loop import TrainableMetricRotorPipeline

try:
    from core.physics.conceptual_causal_tensor_engine import ConceptualCausalTensorEngine
except ImportError:
    ConceptualCausalTensorEngine = None

__all__ = [
    "CounterfactualBranchingEngine",
    "ObservationalFrameController",
    "ConstructiveSpacetimeAxis",
    "ConstructiveLogicDiscriminator",
    "HierarchicalScaleCoupler",
    "Clifford5DRotorEngine",
    "LandauGinzburgPotentialEngine",
    "QuantumRotorPhaseIntegrator",
    "TensorHelmholtzMetricDecomposer",
    "MetricWaveDecompositionResult",
    "IntegratedMetricRotorPipeline",
    "MetricToRotorBridge",
    "TrainableMetricRotorPipeline",
]

if ConceptualCausalTensorEngine is not None:
    __all__.append("ConceptualCausalTensorEngine")
