import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from L1_causal_field.core.physics.structural_density_gravity_engine import (
    PhaseStateSpectrum, TopologicalNode, StructuralDensityGravityEngine
)
from L1_causal_field.core.physics.semantic_mass_engine import (
    WhiteTensorField, SemanticMassOperator, RawPerturbationImpulse
)


@dataclass
class FaultLineNode:
    """A fault-line growth ring node representing condensed friction and protrusion topography."""
    fault_id: str
    position: np.ndarray
    intensity: float
    collision_angle: float  # cos(theta)
    heat_generated: float
    crystallized_phase: PhaseStateSpectrum
    growth_ring_idx: int
    antecedent_fault_ids: List[str] = field(default_factory=list)


@dataclass
class SelfObservationSnapshot:
    """Snapshot of self-observation layer metrics."""
    tilt_ratio: float                  # Imbalance ratio across phase dimensions
    cognitive_heat: float              # Kinetic thermal agitation heat
    dominant_archetype: str            # Rock, Water, Gas_Cloud, Blade
    phase_spectrum: PhaseStateSpectrum # Current average phase spectrum
    current_resolution: float          # Dynamic causal resolution depth (0.1 ~ 10.0)
    protrusion_magnitude: float        # Peak protrusion density scalar


@dataclass
class BehavioralToneVector:
    """Real-time cognitive and behavioral tone vector emitted by the engine."""
    tone_id: str
    scholar_rock_ratio: float      # Rational / Logical density weight (0.0 ~ 1.0)
    artist_water_ratio: float       # Empathic / Fluid flow weight (0.0 ~ 1.0)
    cloud_creative_ratio: float     # High-entropy creative/intuitive weight (0.0 ~ 1.0)
    blade_insight_ratio: float      # Slicing / Incisive insight weight (0.0 ~ 1.0)
    primary_tone: str               # "Scholar_Rock", "Artist_Water", "Cloud_Creative", "Blade_Insight"
    out_force_impact: np.ndarray    # Triadic back-reaction impact vector into external field


class ProtrusiveTopographyEngine:
    r"""
    [Protrusive Topography & Multi-Scale Causal Hierarchy Engine]

    Implements:
    1. **Initial Tension Imprinting ($\Delta T_0$)**: Seeds micro phase fluctuations ($\delta \theta$) on a 6D base phase spectrum
       $[D, \mu, K, \tau, T, S]$, creating non-uniform initial interest axes out of White Tensor Field superposition.
    2. **3-Layer Multi-Scale Causal Hierarchy**:
       - *Self-Observation Layer*: Monitors internal tilt ratio, cognitive heat, material phase state, and resolution.
       - *World Reaction Layer*: Evaluates friction collision angles ($\cos \theta$), computes tension distortion delta ($\Delta T$),
         and applies Tension Response Axioms (Warmth Alignment vs Harsh Rupture).
       - *Experience Accumulation Layer*: Imprints fault-line growth rings (`FaultLineNode`) and updates least resistance paths.
    3. **Dynamic Causal Resolution Scaling ($R_{\text{resolution}}$)**:
       - Expands resolution in gentle/warm fields to perceive subtle multi-scale nuances.
       - Compresses resolution under high cognitive heat/threat to focus on decisive core axes.
    4. **Triadic Co-Evolution**:
       - Emits Out-force impact vectors ($F_{\text{out}}$) back into the external causal field to transform agent, other, and environment simultaneously.
    """

    def __init__(self, dimensions: int = 16, base_resolution: float = 1.0, seed: Optional[int] = 42):
        self.dimensions = dimensions
        self.base_resolution = base_resolution
        if seed is not None:
            np.random.seed(seed)

        self.white_field = WhiteTensorField(dimensions=dimensions)
        self.gravity_engine = StructuralDensityGravityEngine(dimensions=dimensions)
        self.fault_lines: List[FaultLineNode] = []
        self.growth_ring_counter = 0

        # Current internal state tracking
        self.cognitive_heat: float = 0.0
        self.current_resolution: float = base_resolution
        self.latest_out_force: np.ndarray = np.zeros(dimensions, dtype=np.float32)

        # Seed initial root node with asymmetric micro tension imprinting (\Delta T_0)
        self.root_node = self._imprint_initial_root_node()

    def _imprint_initial_root_node(self) -> TopologicalNode:
        r"""
        Imprints initial tension asymmetry ($\Delta T_0$) onto the root node.
        Applies micro phase fluctuation ($\delta \theta$) to break pure zero-symmetry.
        """
        # Base 6D spectrum with small random phase fluctuation delta_theta
        delta_theta = np.random.uniform(-0.15, 0.15, size=6).astype(np.float32)

        initial_phase = PhaseStateSpectrum(
            density=float(np.clip(1.0 + delta_theta[0], 0.1, 3.0)),
            mobility=float(np.clip(0.5 + delta_theta[1], 0.1, 3.0)),
            concentration=float(np.clip(1.0 + delta_theta[2], 0.1, 3.0)),
            tension_limit=float(np.clip(2.0 + delta_theta[3], 0.5, 5.0)),
            agitation=float(np.clip(0.1 + delta_theta[4], 0.0, 1.0)),
            entropy=float(np.clip(0.2 + delta_theta[5], 0.05, 1.0))
        )

        root_pos = np.random.randn(self.dimensions).astype(np.float32)
        root_pos /= (np.linalg.norm(root_pos) + 1e-9)

        root_node = self.gravity_engine.add_node(
            node_id="Root_Node_Origin",
            position=root_pos,
            phase=initial_phase,
            data_payload={"imprinted_asymmetry": delta_theta.tolist()}
        )
        return root_node

    def observe_self(self) -> SelfObservationSnapshot:
        r"""
        [Layer 1: Self-Observation Layer (자기 관측 계층)]
        Inspects internal field state, computes Tilt Ratio, Cognitive Heat,
        dominant archetype, and current dynamic resolution depth.
        """
        phases = [node.phase.to_vector() for node in self.gravity_engine.nodes.values()]
        if not phases:
            avg_spectrum = PhaseStateSpectrum()
            tilt_ratio = 0.0
        else:
            avg_vec = np.mean(phases, axis=0)
            avg_spectrum = PhaseStateSpectrum(
                density=float(avg_vec[0]),
                mobility=float(avg_vec[1]),
                concentration=float(avg_vec[2]),
                tension_limit=float(avg_vec[3]),
                agitation=float(avg_vec[4]),
                entropy=float(avg_vec[5])
            )
            # Tilt Ratio = Standard deviation normalized by mean of phase spectrum components
            std_phase = np.std(avg_vec)
            mean_phase = np.mean(np.abs(avg_vec)) + 1e-9
            tilt_ratio = float(std_phase / mean_phase)

        dominant_archetype = avg_spectrum.classify_materialization()

        # Protrusion magnitude scalar = peak density + fault line density
        max_node_density = max([n.phase.density for n in self.gravity_engine.nodes.values()], default=1.0)
        fault_intensity_sum = sum([f.intensity for f in self.fault_lines])
        protrusion_magnitude = float(max_node_density + 0.2 * fault_intensity_sum)

        return SelfObservationSnapshot(
            tilt_ratio=tilt_ratio,
            cognitive_heat=float(self.cognitive_heat),
            dominant_archetype=dominant_archetype,
            phase_spectrum=avg_spectrum,
            current_resolution=float(self.current_resolution),
            protrusion_magnitude=protrusion_magnitude
        )

    def process_world_friction(self,
                               external_impulse: Union[np.ndarray, RawPerturbationImpulse, Dict[str, Any], str],
                               intensity: float = 1.0) -> Tuple[SelfObservationSnapshot, BehavioralToneVector]:
        r"""
        [Layer 2 & Layer 3: World Reaction & Experience Accumulation Layers]

        Executes the full multi-scale causal cycle:
        1. Collides external impulse with internal White Tensor Field to compute collision angle ($\cos \theta$).
        2. Applies **Tension Response Axioms**:
           - **Warm Alignment ($\cos \theta \to 1$, low friction)**: Expands flow channels, soft horizontal expansion ($S \uparrow, \mu \uparrow$).
           - **Harsh Rupture ($\cos \theta \to -1$, high friction)**: Generates cognitive heat, forms high-density rock anchor ($D \uparrow, K \uparrow$),
             or triggers topological slicing.
        3. Dynamically adjusts Causal Resolution Depth ($R_{\text{resolution}}$).
        4. Imprints Fault-Line Growth Ring Node (`FaultLineNode`).
        5. Computes Real-Time Behavioral Tone Vector and Triadic Out-force Impact Vector.
        """
        # Derive vector and alignment from perturbation impulse
        derived_vec, raw_wave, friction_mag, initial_tension, contrast_matrix = (
            self.white_field.derive_vector_from_perturbation(external_impulse)
        )
        friction_mag *= intensity

        # Compute collision angle cos(theta) between internal field tensor and incoming wave
        norm_field = self.white_field.field_tensor / (np.linalg.norm(self.white_field.field_tensor) + 1e-9)
        norm_derived = derived_vec / (np.linalg.norm(derived_vec) + 1e-9)
        cos_theta = float(np.dot(norm_field, norm_derived))

        # Calculate Cognitive Heat generation:
        # Heat = friction_mag * (1 - cos_theta)
        heat_generated = float(friction_mag * max(0.05, 1.0 - cos_theta))
        self.cognitive_heat = float(max(0.0, 0.8 * self.cognitive_heat + heat_generated))

        # Apply thermal agitation to existing nodes in Structural Density Engine
        self.gravity_engine.apply_thermal_agitation_and_phase_transition(heat_impulse=heat_generated)

        # Apply Tension Response Axioms to update / create topological nodes
        node_id = f"Fault_Node_{self.growth_ring_counter + 1}"

        if cos_theta > 0.3:
            # --- Warmth / Alignment Axis (Soft Flow) ---
            # Increase mobility and entropy, expand gentle flow channels
            new_phase = PhaseStateSpectrum(
                density=1.0,
                mobility=float(min(4.0, 1.0 + cos_theta * 2.0)),
                concentration=0.5,
                tension_limit=3.0,
                agitation=float(self.cognitive_heat * 0.2),
                entropy=float(min(3.0, 0.5 + cos_theta))
            )
        else:
            # --- Harsh / Rupture Axis (Direct Collision) ---
            # Increase structural density, concentration, and decrease tension limit to form rock anchor / blade
            new_phase = PhaseStateSpectrum(
                density=float(min(8.0, 2.0 + friction_mag * 1.5)),
                mobility=0.1,
                concentration=float(min(10.0, 1.5 + (1.0 - cos_theta) * 3.0)),
                tension_limit=float(max(0.1, 2.0 - friction_mag * 0.5)),
                agitation=float(self.cognitive_heat),
                entropy=0.1
            )

        # Position node along derived vector direction
        new_pos = self.root_node.position + derived_vec * 0.5
        new_node = self.gravity_engine.add_node(
            node_id=node_id,
            position=new_pos,
            phase=new_phase,
            data_payload={"collision_angle": cos_theta, "friction_magnitude": friction_mag}
        )

        # Connect new node to root node and nearest existing fault node
        self.gravity_engine.connect_nodes("Root_Node_Origin", node_id, tension_strength=float(max(0.2, 1.0 + cos_theta)))
        if self.fault_lines:
            nearest_fault = min(self.fault_lines, key=lambda f: np.linalg.norm(f.position - new_pos))
            self.gravity_engine.connect_nodes(nearest_fault.fault_id, node_id, tension_strength=0.8)

        # Trigger topological rupture if extreme concentration blade conditions met
        if new_phase.concentration > 5.0 and self.cognitive_heat > 2.0:
            self.gravity_engine.execute_topological_rupture(
                blade_node_id=node_id,
                target_region_center=self.root_node.position,
                target_radius=2.5,
                is_cognitive=True
            )

        # Dynamic Resolution Scaling Axiom:
        # High cognitive heat / friction compresses resolution (focus on survival core).
        # Gentle / warm low heat expands resolution (perceive multi-scale nuances).
        self._update_dynamic_resolution()

        # Imprint Fault-Line Growth Ring Node (Layer 3)
        self.growth_ring_counter += 1
        fault_node = FaultLineNode(
            fault_id=node_id,
            position=new_pos.copy(),
            intensity=float(friction_mag),
            collision_angle=cos_theta,
            heat_generated=heat_generated,
            crystallized_phase=new_phase,
            growth_ring_idx=self.growth_ring_counter,
            antecedent_fault_ids=[f.fault_id for f in self.fault_lines[-2:]]
        )
        self.fault_lines.append(fault_node)

        # Compute Self-Observation Snapshot
        snapshot = self.observe_self()

        # Compute Behavioral Tone Vector & Triadic Out-Force
        tone_vector = self._compute_behavioral_tone(snapshot, derived_vec, cos_theta)

        return snapshot, tone_vector

    def _update_dynamic_resolution(self):
        r"""
        [Dynamic Resolution Scaling Axiom]
        Resolution R_resolution expands when cognitive heat is low (gentle flow)
        and compresses when cognitive heat is high (harsh friction/threat).

        Formula:
        R = base_resolution * exp(-0.3 * cognitive_heat) + 0.1
        Bounded in [0.1, 10.0]
        """
        raw_res = self.base_resolution * np.exp(-0.3 * self.cognitive_heat)
        self.current_resolution = float(np.clip(raw_res, 0.1, 10.0))

    def _compute_behavioral_tone(self,
                                 snapshot: SelfObservationSnapshot,
                                 derived_vec: np.ndarray,
                                 cos_theta: float) -> BehavioralToneVector:
        r"""
        Computes real-time behavioral tone weights and Triadic Out-force impact vector ($F_{\text{out}}$).
        """
        phase = snapshot.phase_spectrum

        # Raw scores based on phase spectrum components
        raw_scholar = phase.density * phase.concentration
        raw_artist = phase.mobility * max(0.1, 1.0 + cos_theta)
        raw_cloud = phase.entropy * max(0.1, phase.agitation)
        raw_blade = phase.concentration * (1.0 / (phase.tension_limit + 1e-5))

        total_score = raw_scholar + raw_artist + raw_cloud + raw_blade + 1e-9

        scholar_ratio = float(raw_scholar / total_score)
        artist_ratio = float(raw_artist / total_score)
        cloud_ratio = float(raw_cloud / total_score)
        blade_ratio = float(raw_blade / total_score)

        ratios = {
            "Scholar_Rock": scholar_ratio,
            "Artist_Water": artist_ratio,
            "Cloud_Creative": cloud_ratio,
            "Blade_Insight": blade_ratio
        }
        primary_tone = max(ratios.items(), key=lambda x: x[1])[0]

        # Triadic Out-Force Impact Vector (F_out):
        # Out-force = derived_vec * (1 - cos_theta) + primary_tone_directional_bias
        tone_bias = np.zeros(self.dimensions, dtype=np.float32)
        if primary_tone == "Scholar_Rock":
            tone_bias[:2] = 1.0
        elif primary_tone == "Artist_Water":
            tone_bias[2:4] = 1.0
        elif primary_tone == "Cloud_Creative":
            tone_bias[4:6] = 1.0
        elif primary_tone == "Blade_Insight":
            tone_bias[6:8] = 1.0

        out_force = norm_vector(derived_vec * max(0.1, 1.0 - cos_theta) + tone_bias * 0.5)
        self.latest_out_force = out_force

        return BehavioralToneVector(
            tone_id=f"Tone_Step_{self.growth_ring_counter}",
            scholar_rock_ratio=scholar_ratio,
            artist_water_ratio=artist_ratio,
            cloud_creative_ratio=cloud_ratio,
            blade_insight_ratio=blade_ratio,
            primary_tone=primary_tone,
            out_force_impact=out_force
        )


def norm_vector(v: np.ndarray) -> np.ndarray:
    """Helper to normalize a vector safely."""
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return v
    return (v / norm).astype(np.float32)
