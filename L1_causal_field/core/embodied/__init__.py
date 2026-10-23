"""
Elysia Embodied Cognition Package Initialization
"""

from .schemas import ProtoVector, QualitySymbol, SensoryFrame
from .projector import SymbolPrototype, VQSensoryProjector
from .plasticity import DualPlasticityPipeline
from .emergence import OODLatentBuffer, AxiomInducer, ConsistencyFilter, CrossModalCausalFilter, ResidualCompressibilityFilter
from .topological_dimensional_mirror import (
    BodySchemaState,
    DimensionalTopologicalEngine,
    DualEnvironmentMirrorLoop,
    EnvironmentCondition,
    LivingCognitiveSchemaEngine,
    LivingSchema,
    TopologicalDiscrepancy,
)

__all__ = [
    "ProtoVector",
    "QualitySymbol",
    "SensoryFrame",
    "SymbolPrototype",
    "VQSensoryProjector",
    "DualPlasticityPipeline",
    "OODLatentBuffer",
    "ConsistencyFilter",
    "CrossModalCausalFilter",
    "ResidualCompressibilityFilter",
    "AxiomInducer",
    "EnvironmentCondition",
    "BodySchemaState",
    "TopologicalDiscrepancy",
    "LivingSchema",
    "DimensionalTopologicalEngine",
    "DualEnvironmentMirrorLoop",
    "LivingCognitiveSchemaEngine",
]
