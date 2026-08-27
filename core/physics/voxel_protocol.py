"""
voxel_protocol.py — Unified Field Lens Protocol
=================================================
[Informational Continuity: Protocol-Based Coupling]

Defines the FieldLens protocol — the contract that every engine must satisfy
to participate in the unified CausalField as an observer/modulator.

Instead of engines maintaining separate state spaces, each engine becomes a
"lens" that observes and modulates the shared InformationVoxel state.

This maintains the 4 Continuities:
- Relationship: Engines define boundaries through potential field coupling, not hard limits
- Connectivity: Changes propagate through ConnectivityBeam networks
- Mobility: Every modulation is a vector delta (direction + magnitude)
- Informational Continuity: Lenses operate on the same dt-stepped field
"""
import numpy as np
from typing import Protocol, Dict, List, Any, Optional, runtime_checkable
from dataclasses import dataclass


@dataclass
class FieldDelta:
    """
    [Mobility: Conservation of Impact]
    Represents a continuous change vector applied to the unified field.
    Every action has a corresponding impact vector — nothing changes without a trace.
    """
    # Which voxel(s) are affected
    target_voxel_ids: List[str]
    
    # Continuous delta vectors (None means no change to that property)
    delta_potential: Optional[float] = None
    delta_velocity: Optional[np.ndarray] = None
    delta_conductance: Optional[float] = None
    delta_temperature: Optional[float] = None
    delta_phase_angle: Optional[float] = None
    delta_coordination_margin: Optional[float] = None
    delta_curiosity_charge: Optional[float] = None
    delta_chromatic: Optional[np.ndarray] = None  # [dR, dB, dY]
    
    # The lens that produced this delta (for causal tracing)
    source_lens: str = "unknown"
    
    # Magnitude of this intervention (for energy accounting)
    @property
    def energy(self) -> float:
        """Total energy of this delta — conservation law."""
        e = 0.0
        if self.delta_potential is not None:
            e += abs(self.delta_potential)
        if self.delta_velocity is not None:
            e += float(np.linalg.norm(self.delta_velocity))
        if self.delta_conductance is not None:
            e += abs(self.delta_conductance)
        if self.delta_temperature is not None:
            e += abs(self.delta_temperature)
        if self.delta_chromatic is not None:
            e += float(np.linalg.norm(self.delta_chromatic))
        return e


@runtime_checkable
class FieldLens(Protocol):
    """
    [Relationship: Coupled Potential Field Interface]
    
    A FieldLens is any engine that observes and modulates the unified CausalField.
    Instead of maintaining its own state space, it reads from and writes to
    the shared InformationVoxel state through continuous FieldDelta vectors.
    
    This is the structural replacement for discrete function-call pipelines:
    - Old: result = engine.process(input_data)  # discrete translation
    - New: deltas = lens.observe_and_modulate(field)  # continuous field coupling
    """
    
    @property
    def lens_id(self) -> str:
        """Unique identifier for causal tracing."""
        ...
    
    def observe(self, voxels: Dict[str, Any], beams: List[Any]) -> Dict[str, Any]:
        """
        [Observation Phase]
        Reads the current field state through this lens's particular perspective.
        Returns lens-specific observables (e.g., 2D projection, phase spectrum, etc.)
        Does NOT modify the field.
        """
        ...
    
    def modulate(self, voxels: Dict[str, Any], beams: List[Any], dt: float) -> List[FieldDelta]:
        """
        [Modulation Phase]
        Based on the lens's observation, produces a list of FieldDelta vectors
        to be applied to the unified field.
        
        The deltas are additive and composable — multiple lenses can modulate
        the same voxel simultaneously, and their effects combine linearly.
        This ensures causal simultaneity instead of sequential pipeline order.
        """
        ...


class LensRegistry:
    """
    [Connectivity: Topology of Lenses]
    Manages registered FieldLens instances and applies their deltas to the field.
    This replaces the 40+ import ConsciousnessLoop pattern.
    """
    
    def __init__(self):
        self._lenses: Dict[str, FieldLens] = {}
        self._delta_history: List[List[FieldDelta]] = []  # Informational Continuity
        self._total_energy_budget: float = 0.0
    
    def register(self, lens: FieldLens) -> None:
        """Register a lens to participate in the unified field evolution."""
        self._lenses[lens.lens_id] = lens
    
    def unregister(self, lens_id: str) -> None:
        """Remove a lens from the field."""
        self._lenses.pop(lens_id, None)
    
    @property
    def registered_lenses(self) -> Dict[str, FieldLens]:
        return dict(self._lenses)
    
    def collect_all_deltas(
        self,
        voxels: Dict[str, Any],
        beams: List[Any],
        dt: float
    ) -> List[FieldDelta]:
        """
        [Causal Simultaneity]
        Collects deltas from ALL registered lenses in one pass.
        All lenses observe the SAME pre-step state, ensuring no ordering bias.
        """
        all_deltas: List[FieldDelta] = []
        for lens in self._lenses.values():
            deltas = lens.modulate(voxels, beams, dt)
            all_deltas.extend(deltas)
        self._delta_history.append(all_deltas)
        return all_deltas
    
    def apply_deltas(
        self,
        voxels: Dict[str, Any],
        deltas: List[FieldDelta]
    ) -> float:
        """
        [Mobility: Additive Composition]
        Applies all collected deltas to the voxel state.
        Returns total energy injected (for conservation tracking).
        """
        total_energy = 0.0
        
        for delta in deltas:
            for vid in delta.target_voxel_ids:
                if vid not in voxels:
                    continue
                v = voxels[vid]
                
                if delta.delta_potential is not None:
                    v.potential += delta.delta_potential
                if delta.delta_velocity is not None:
                    v.velocity += delta.delta_velocity
                if delta.delta_conductance is not None:
                    v.conductance = max(1e-6, v.conductance + delta.delta_conductance)
                if delta.delta_temperature is not None:
                    v.temperature = max(0.01, v.temperature + delta.delta_temperature)
                if delta.delta_phase_angle is not None:
                    v.phase_angle = (v.phase_angle + delta.delta_phase_angle) % (2.0 * np.pi)
                if delta.delta_coordination_margin is not None:
                    v.coordination_margin = np.clip(
                        v.coordination_margin + delta.delta_coordination_margin, 0.0, 1.0
                    )
                if delta.delta_curiosity_charge is not None:
                    v.curiosity_charge += delta.delta_curiosity_charge
                if delta.delta_chromatic is not None:
                    v.chromatic_vector += delta.delta_chromatic
                    # Re-normalize chromatic: sum to 1, all >= 0
                    v.chromatic_vector = np.maximum(v.chromatic_vector, 0.0)
                    total = float(np.sum(v.chromatic_vector))
                    if total > 0:
                        v.chromatic_vector /= total
                
                total_energy += delta.energy
        
        self._total_energy_budget += total_energy
        return total_energy
