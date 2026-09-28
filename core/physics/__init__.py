"""Elysia Core - Physics / Kinematics Subsystem"""

from core.physics.counterfactual_branching import CounterfactualBranchingEngine
from core.physics.frame_controller import ObservationalFrameController
from core.physics.constructive_causal_spacetime import (
    ConstructiveSpacetimeAxis,
    ConstructiveLogicDiscriminator,
    HierarchicalScaleCoupler,
)

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
]

if ConceptualCausalTensorEngine is not None:
    __all__.append("ConceptualCausalTensorEngine")
