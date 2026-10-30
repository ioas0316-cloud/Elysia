"""
ELYSIA Core Package: Scale Principle Hierarchy Architecture

Exposes scale principle layers (Layer 0 Substrate, Layer 1 Constraint, Layer 2 Resonant,
Layer 3 Macro, and Orchestration) along with backward compatibility facades for existing modules.
"""

# Export new Scale Principle Hierarchy Modules
from L1_causal_field.core.layer_1_constraint.sensory_prismatic_gate import SensoryPrismaticGate, ContinuousWaveformLens
from L1_causal_field.core.layer_2_resonant.surface_tension_attractor import SurfaceTensionAttractor, PhaseRelaxationEngine

__all__ = [
    "SensoryPrismaticGate",
    "ContinuousWaveformLens",
    "SurfaceTensionAttractor",
    "PhaseRelaxationEngine",
]
