"""
Elysia Embodied Cognition Package Initialization
"""

from .schemas import ProtoVector, QualitySymbol, SensoryFrame
from .projector import SymbolPrototype, VQSensoryProjector
from .plasticity import DualPlasticityPipeline
from .emergence import OODLatentBuffer, AxiomInducer, ConsistencyFilter, CrossModalCausalFilter, ResidualCompressibilityFilter

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
]
