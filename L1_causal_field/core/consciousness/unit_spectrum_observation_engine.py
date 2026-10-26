"""
unit_spectrum_observation_engine.py: Unit Spectrum Observation Engine & Grounded Cognitive Unit '1'.

Embodying symbol grounding and scale translation in Elysia:
- The concept of "1" (Unit '1') is not a fixed, abstract 0/1 scalar symbol, but a dynamic,
  emergent boundary formed when an observation scale/lens binds a cluster of micro-rotor particles
  sharing coherent phase synchronization and somatic friction/resistance.
- GroundedCognitiveUnit provides seamless 1:1 interchangeability between physical units
  (somatic friction, pressure, mass) and mental/cognitive units (semantic mass, memory, value).
- UnitSpectrumObservationEngine translates microscopic rotor friction into macro cognitive units,
  allowing dynamic clustering, formation, and dissolution of "1" across scales.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Any, Optional
from L1_causal_field.core.topology.particle_rotor_matrix_sandbox import MicroRotorParticle, ParticleRotorMatrixSandbox


@dataclass
class GroundedCognitiveUnit:
    """
    Grounded Unit '1' representing a coherent cognitive-physical entity.
    Acts as an interchangeable bridge between physical unit metrics and mental cognitive metrics.
    """
    unit_id: str
    bound_particle_ids: List[int]
    observation_scale: float            # Scale zoom factor (0.1 = microscopic, 1.0 = mesoscopic, 10.0 = macroscopic)
    physical_mass: float                 # Total somatic/physical mass sum (kg / Ms)
    somatic_friction: float              # Total somatic friction / resistance
    phase_coherence: float               # Phase lock degree R in [0, 1]
    centroid: np.ndarray                 # Spatial center of the unit
    semantic_mass: float                 # Mental semantic mass Ms = physical_mass * (1 + phase_coherence)
    value_density: float                 # Value/importance density = semantic_mass / (volume or footprint)
    memory_engram_weight: float          # Engram weight accumulated through somatic friction
    sensory_footprint: Dict[str, float]  # Aggregated somatic profile (pressure, weight, texture)

    def to_physical_dict(self) -> Dict[str, Any]:
        """
        Convert Grounded Unit '1' into physical unit system metrics.
        """
        return {
            "unit_id": self.unit_id,
            "physical_mass_kg": self.physical_mass,
            "somatic_friction_N": self.somatic_friction,
            "pressure_Pa": self.sensory_footprint.get("pressure", 0.0),
            "centroid_pos": self.centroid.tolist(),
            "particle_count": len(self.bound_particle_ids),
        }

    def to_mental_dict(self) -> Dict[str, Any]:
        """
        Convert Grounded Unit '1' into mental cognitive system metrics.
        Demonstrates seamless interchangeability between physical unit and mental unit.
        """
        return {
            "unit_id": self.unit_id,
            "cognitive_unit_symbol": "1_grounded",
            "semantic_mass_Ms": self.semantic_mass,
            "value_density": self.value_density,
            "memory_engram_weight": self.memory_engram_weight,
            "phase_coherence_R": self.phase_coherence,
            "observation_scale": self.observation_scale,
        }


class UnitSpectrumObservationEngine:
    """
    Observes the ParticleRotorMatrixSandbox across a continuous Unit Spectrum,
    binding micro-rotor particles into grounded '1' units or dissolving them
    based on observation scale, distance thresholds, and phase lock coherence.
    """

    def __init__(
        self,
        observation_scale: float = 1.0,
        coherence_threshold: float = 0.3,
        cluster_radius: float = 2.5,
    ):
        self.observation_scale = observation_scale
        self.coherence_threshold = coherence_threshold
        self.cluster_radius = cluster_radius
        self.grounded_units: Dict[str, GroundedCognitiveUnit] = {}

    def set_observation_scale(self, scale: float) -> None:
        """
        Adjust observation scale lens.
        Larger scale binds wider clusters into single '1' macro units.
        Smaller scale dissolves macro '1' units into fine micro-rotor components.
        """
        self.observation_scale = max(0.01, float(scale))

    def observe_and_bind_units(self, sandbox: ParticleRotorMatrixSandbox) -> List[GroundedCognitiveUnit]:
        """
        Examine sandbox particles, construct adjacency by effective cluster radius (scaled by observation_scale),
        and bind phase-coherent connected components into GroundedCognitiveUnits ('1').
        """
        particles = sandbox.particles
        N = len(particles)
        if N == 0:
            self.grounded_units.clear()
            return []

        effective_radius = self.cluster_radius * self.observation_scale
        positions = np.array([p.position for p in particles])
        phases = np.array([p.theta for p in particles])

        # Find connected components (simple BFS / flood fill)
        visited = np.zeros(N, dtype=bool)
        clusters: List[List[int]] = []

        for i in range(N):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            queue = [i]

            while queue:
                curr = queue.pop(0)
                diff = positions - positions[curr]
                dist = np.linalg.norm(diff, axis=1)

                # Neighbors within effective radius
                neighbors = np.where((dist < effective_radius) & (~visited))[0]
                for nbr in neighbors:
                    # Check phase coherence with cluster
                    phase_diff = np.abs(np.angle(np.exp(1j * (phases[nbr] - phases[curr]))))
                    if phase_diff < np.pi / 2: # Phase alignment condition
                        visited[nbr] = True
                        cluster.append(nbr)
                        queue.append(nbr)

            clusters.append(cluster)

        # Build GroundedCognitiveUnits for each bound cluster
        new_units: Dict[str, GroundedCognitiveUnit] = {}
        for idx, cluster_indices in enumerate(clusters):
            unit_id = f"unit_scale_{self.observation_scale:.2f}_cluster_{idx}"
            cluster_particles = [particles[i] for i in cluster_indices]

            # Aggregate physical properties
            tot_mass = sum(p.mass for p in cluster_particles)
            tot_friction = sum(p.friction for p in cluster_particles)

            cluster_positions = np.array([p.position for p in cluster_particles])
            centroid = np.mean(cluster_positions, axis=0)

            # Phase coherence R of cluster
            cluster_phases = np.array([p.theta for p in cluster_particles])
            phase_coherence = float(np.abs(np.mean(np.exp(1j * cluster_phases))))

            # Aggregated somatic footprint
            agg_pressure = sum(p.sensory_profile.get("pressure", 0.0) for p in cluster_particles)
            agg_weight = sum(p.sensory_profile.get("weight", 0.0) for p in cluster_particles)
            agg_texture = float(np.mean([p.sensory_profile.get("texture", 0.0) for p in cluster_particles]))

            sensory_footprint = {
                "pressure": float(agg_pressure),
                "weight": float(agg_weight),
                "texture": float(agg_texture),
            }

            # Calculate mental cognitive metrics
            # Semantic mass Ms scales with physical mass and phase coherence resonance
            semantic_mass = float(tot_mass * (1.0 + phase_coherence))

            # Spatial footprint / volume estimate
            if len(cluster_particles) > 1:
                spread = float(np.max(np.linalg.norm(cluster_positions - centroid, axis=1))) + 0.1
            else:
                spread = 1.0

            value_density = float(semantic_mass / spread)
            memory_engram_weight = float(tot_friction * (1.0 + 2.0 * phase_coherence))

            unit = GroundedCognitiveUnit(
                unit_id=unit_id,
                bound_particle_ids=[p.id for p in cluster_particles],
                observation_scale=self.observation_scale,
                physical_mass=tot_mass,
                somatic_friction=tot_friction,
                phase_coherence=phase_coherence,
                centroid=centroid,
                semantic_mass=semantic_mass,
                value_density=value_density,
                memory_engram_weight=memory_engram_weight,
                sensory_footprint=sensory_footprint,
            )
            new_units[unit_id] = unit

        self.grounded_units = new_units
        return list(self.grounded_units.values())

    def interchange_unit_representation(self, unit_id: str, target_mode: str = "mental") -> Dict[str, Any]:
        """
        Seamlessly interchange a Grounded Cognitive Unit '1' between 'physical' and 'mental' representations.
        """
        if unit_id not in self.grounded_units:
            raise KeyError(f"Unit {unit_id} not found in observation engine.")

        unit = self.grounded_units[unit_id]
        if target_mode == "physical":
            return unit.to_physical_dict()
        elif target_mode == "mental":
            return unit.to_mental_dict()
        else:
            raise ValueError(f"Unknown target_mode {target_mode}. Choose 'physical' or 'mental'.")
