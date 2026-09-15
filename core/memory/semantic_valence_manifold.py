"""
Semantic Valence Manifold & Cognitive Lens Engine for Elysia.

This module implements:
1. Composite Valence Potential Field V(x, t) = w_core(x) V_core + w_eff V_eff + w_fric V_fric
   with non-linear core weight divergence w_core(x) near r_core.
2. Valence-weighted metric tensor g_ij(x) and continuous gradient flow motion:
   m * d^2 x / dt^2 + gamma * dx / dt = grad V(x, t)
3. Sensory Refraction via CognitiveLens (E_ext -> L_cog -> T_wave).
4. Concept Attractor formation, Cognitive Mitosis (unmapped novel state -> cell division),
   and Cognitive Fusion (spatiotemporal/frequency synchronization -> abstraction).
5. Biological Boundary Modules (dopamine scaling, refractory period, LTP scarring engine).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .closed_loop_valence import SensoryFeedbackEngine, ClosedLoopValenceField


# ============================================================================
# 1. Signals & Data Structures
# ============================================================================

@dataclass
class FieldFluxSignal:
    """Flux signal passing through a biological boundary surface (∂Ω)."""
    amplitude: float         # Wave/Tension amplitude (Energy)
    frequency: float         # Resonant frequency (Hz)
    gradient: np.ndarray     # Potential gradient vector (∇V)


@dataclass
class ExternalSensorySignal:
    """External physical sensory signal E_ext (e.g., optical, acoustic, tactile)."""
    sensory_type: str         # 'optical', 'acoustic', 'tactile'
    raw_spectrum: np.ndarray  # Raw physical spectrum values
    intensity: float          # Physical stimulus intensity
    external_pos: np.ndarray  # External 3D coordinate [x, y, z]


@dataclass
class RefractedTensionWave:
    """Internal tension wave T_wave refracted through CognitiveLens L_cog."""
    internal_frequency: float  # Refracted semantic resonant frequency (Hz)
    amplitude: float           # Refracted tension amplitude
    refraction_dir: np.ndarray # Refraction direction vector in thought space


@dataclass
class ConceptAttractor:
    """Concept Attractor Well on the semantic valence potential field."""
    concept_id: str
    center_pos: np.ndarray       # Geometric position [x, y, z] in thought space
    resonant_freq: float        # Resonant frequency (Hz)
    well_depth: float           # Potential well depth (valence magnitude)
    well_radius: float          # Potential attraction radius
    generation: int = 0         # Mitosis / Fusion generation lineage
    fused_origins: List[str] = field(default_factory=list) # History of merged concept IDs


# ============================================================================
# 2. Biological Boundary Modules (∂Ω Interfaces)
# ============================================================================

class BoundaryModule(ABC):
    """Abstract interface for biological boundary conditions at ∂Ω."""

    @abstractmethod
    def apply_boundary_condition(
        self,
        signal: FieldFluxSignal,
        current_time: float
    ) -> FieldFluxSignal:
        """Transforms flux signal passing through the boundary interface."""
        pass


class DopamineScalerModule(BoundaryModule):
    """Dynamically scales potential gradient (∇V) and amplitude based on dopamine level."""

    def __init__(self, initial_dopamine: float = 1.0):
        self.dopamine_level = max(0.0, initial_dopamine)

    def set_dopamine_level(self, level: float):
        self.dopamine_level = max(0.0, level)

    def apply_boundary_condition(
        self,
        signal: FieldFluxSignal,
        current_time: float
    ) -> FieldFluxSignal:
        scaled_amp = signal.amplitude * self.dopamine_level
        scaled_grad = signal.gradient * self.dopamine_level
        return FieldFluxSignal(
            amplitude=scaled_amp,
            frequency=signal.frequency,
            gradient=scaled_grad
        )


class RefractoryPeriodModule(BoundaryModule):
    """Surges temporary impedance (Z_temp) post-activation to block influx for a duration."""

    def __init__(self, refractory_duration: float = 0.5, activation_threshold: float = 1.0):
        self.refractory_duration = refractory_duration
        self.activation_threshold = activation_threshold
        self.last_activation_time: float = -999.0

    def apply_boundary_condition(
        self,
        signal: FieldFluxSignal,
        current_time: float
    ) -> FieldFluxSignal:
        if (current_time - self.last_activation_time) < self.refractory_duration:
            return FieldFluxSignal(
                amplitude=0.0,
                frequency=signal.frequency,
                gradient=np.zeros_like(signal.gradient)
            )

        if signal.amplitude >= self.activation_threshold:
            self.last_activation_time = current_time

        return signal


class LTPScarringEngine(BoundaryModule):
    """
    Long-Term Potentiation (LTP) Scarring Engine.
    Integrates resonance energy over time. When accumulated energy exceeds threshold E_th,
    it permanently imprints a Scar Weight (W_scar) onto the local metric tensor field.
    """

    def __init__(self, energy_threshold: float = 5.0, plasticity_rate: float = 0.2, decay_rate: float = 0.01):
        self.energy_threshold = energy_threshold
        self.plasticity_rate = plasticity_rate
        self.decay_rate = decay_rate
        self.accumulated_energy: Dict[float, float] = {} # freq -> energy
        self.scar_weights: Dict[float, float] = {}       # freq -> W_scar

    def apply_boundary_condition(
        self,
        signal: FieldFluxSignal,
        current_time: float
    ) -> FieldFluxSignal:
        freq = round(float(signal.frequency), 1)
        prev_e = self.accumulated_energy.get(freq, 0.0)

        # Integrate flux energy: E_acc += amplitude^2
        new_e = prev_e + (signal.amplitude ** 2) * 0.1
        self.accumulated_energy[freq] = new_e

        # Plastic deformation transition
        if new_e >= self.energy_threshold:
            prev_scar = self.scar_weights.get(freq, 0.0)
            self.scar_weights[freq] = prev_scar + self.plasticity_rate * (new_e - self.energy_threshold)
            # Reset integrated energy excess
            self.accumulated_energy[freq] = self.energy_threshold * 0.5

        # Imprinted scar weight boosts outgoing signal flow (reduces impedance)
        scar = self.scar_weights.get(freq, 0.0)
        boosted_amp = signal.amplitude * (1.0 + 0.5 * scar)

        return FieldFluxSignal(
            amplitude=boosted_amp,
            frequency=signal.frequency,
            gradient=signal.gradient * (1.0 + 0.3 * scar)
        )

    def get_scar_weight(self, freq: float) -> float:
        return self.scar_weights.get(round(float(freq), 1), 0.0)


# ============================================================================
# 3. Cognitive Lens & Refraction Engine
# ============================================================================

class CognitiveLens:
    """
    Cognitive Lens (L_cog) defining subjective embodiment and sensory limits.
    Refracts raw external sensory signals E_ext into internal tension waves T_wave.
    """

    def __init__(
        self,
        sensory_range: Tuple[float, float] = (100.0, 10000.0),
        refractive_index: float = 1.618,
        seed: int = 42
    ):
        self.sensory_range = sensory_range
        self.refractive_index = refractive_index
        np.random.seed(seed)
        self.spatial_refraction_matrix = np.random.randn(3, 3) * 0.4

    def refract(self, external_signal: ExternalSensorySignal) -> RefractedTensionWave:
        """Refracts E_ext through L_cog into RefractedTensionWave T_wave."""
        # 1. Sensory embodied range filtering
        valid_spectrum = external_signal.raw_spectrum[
            (external_signal.raw_spectrum >= self.sensory_range[0]) &
            (external_signal.raw_spectrum <= self.sensory_range[1])
        ]

        if len(valid_spectrum) == 0:
            effective_freq = self.sensory_range[0]
            effective_intensity = external_signal.intensity * 0.05
        else:
            effective_freq = float(np.mean(valid_spectrum))
            effective_intensity = external_signal.intensity

        # 2. Frequency refraction into semantic manifold spectrum (0 ~ 200 Hz)
        internal_freq = (effective_freq / self.refractive_index) % 200.0

        # 3. Directional refraction via lens curvature matrix
        refracted_dir = np.dot(external_signal.external_pos, self.spatial_refraction_matrix)
        norm = np.linalg.norm(refracted_dir)
        if norm > 1e-8:
            refracted_dir = refracted_dir / norm

        # 4. Refracted amplitude formation
        refracted_amp = effective_intensity * self.refractive_index

        return RefractedTensionWave(
            internal_frequency=internal_freq,
            amplitude=refracted_amp,
            refraction_dir=refracted_dir
        )


# ============================================================================
# 4. Cognitive Mitosis & Fusion Engines
# ============================================================================

class CognitiveMitosisEngine:
    """
    Cognitive Mitosis (Cell Division) Engine.
    Triggers spontaneous formation of a new ConceptAttractor when a thought trajectory
    converges on an unmapped position outside existing concept attraction radii.
    """

    def __init__(self, base_radius: float = 1.0, initial_depth_factor: float = 0.8):
        self.base_radius = base_radius
        self.initial_depth_factor = initial_depth_factor
        self.mitosis_counter = 0

    def trigger_mitosis(
        self,
        converged_pos: np.ndarray,
        wave: RefractedTensionWave,
        nearest_attractor: Optional[ConceptAttractor] = None
    ) -> ConceptAttractor:
        self.mitosis_counter += 1
        new_id = f"CONCEPT_MITO_{self.mitosis_counter:03d}"

        if nearest_attractor:
            inherited_freq = (nearest_attractor.resonant_freq + wave.internal_frequency) / 2.0
            generation = nearest_attractor.generation + 1
        else:
            inherited_freq = wave.internal_frequency
            generation = 1

        initial_depth = max(1.0, wave.amplitude * self.initial_depth_factor)
        dynamic_radius = self.base_radius / (1.0 + np.log1p(wave.amplitude))

        return ConceptAttractor(
            concept_id=new_id,
            center_pos=converged_pos.copy(),
            resonant_freq=inherited_freq,
            well_depth=initial_depth,
            well_radius=dynamic_radius,
            generation=generation
        )


class CognitiveFusionEngine:
    """
    Cognitive Fusion Engine.
    Scans the potential field for interfering concept attractors with high spatial proximity
    and frequency synchronization index J(i, j), fusing them into higher-level abstract attractors.
    """

    def __init__(
        self,
        spatial_threshold: float = 1.2,
        freq_threshold: float = 15.0,
        mass_conservation_ratio: float = 0.85
    ):
        self.spatial_threshold = spatial_threshold
        self.freq_threshold = freq_threshold
        self.mass_conservation_ratio = mass_conservation_ratio
        self.fusion_counter = 0

    def calculate_sync_index(self, att1: ConceptAttractor, att2: ConceptAttractor) -> float:
        """Calculates synchronization index J(i, j) based on spatial & frequency affinity."""
        spatial_dist = np.linalg.norm(att1.center_pos - att2.center_pos)
        freq_diff = abs(att1.resonant_freq - att2.resonant_freq)

        if spatial_dist > self.spatial_threshold or freq_diff > self.freq_threshold:
            return 0.0

        spatial_affinity = np.exp(-spatial_dist / self.spatial_threshold)
        freq_affinity = np.exp(-freq_diff / (self.freq_threshold * 0.5))
        return float(spatial_affinity * freq_affinity)

    def fuse_attractors(self, att1: ConceptAttractor, att2: ConceptAttractor) -> ConceptAttractor:
        """Fuses two concept attractors based on center-of-mass law."""
        self.fusion_counter += 1
        fused_id = f"CONCEPT_FUSED_{self.fusion_counter:03d}"

        m1, m2 = att1.well_depth, att2.well_depth
        total_m = m1 + m2 + 1e-8

        fused_pos = (att1.center_pos * m1 + att2.center_pos * m2) / total_m
        fused_freq = (att1.resonant_freq * m1 + att2.resonant_freq * m2) / total_m
        fused_depth = total_m * self.mass_conservation_ratio

        max_reach = max(
            np.linalg.norm(fused_pos - att1.center_pos) + att1.well_radius,
            np.linalg.norm(fused_pos - att2.center_pos) + att2.well_radius
        )
        fused_radius = float(max_reach * 0.9)

        origins = []
        origins.extend(att1.fused_origins if att1.fused_origins else [att1.concept_id])
        origins.extend(att2.fused_origins if att2.fused_origins else [att2.concept_id])

        return ConceptAttractor(
            concept_id=fused_id,
            center_pos=fused_pos,
            resonant_freq=fused_freq,
            well_depth=fused_depth,
            well_radius=fused_radius,
            generation=max(att1.generation, att2.generation) + 1,
            fused_origins=origins
        )


# ============================================================================
# 5. Composite Semantic Valence Potential Field & Metric Tensor
# ============================================================================

class SemanticValenceManifold:
    """
    Semantic Valence Manifold V(x, t) and Metric Tensor Field g_ij(x).

    Composite Field:
      V(x, t) = w_core(x) * V_core(x) + w_eff * V_eff(x) + w_fric * V_fric(x) + Sum(V_attractor)

    Non-linear core weight divergence:
      w_core(x) = w0 + alpha / max(r - r_core, epsilon)  when r -> r_core

    Metric tensor:
      g_ij(x) = exp(-beta * V(x)) * I + Scar_Term
    """

    def __init__(
        self,
        core_origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
        core_radius: float = 0.8,
        w0_core: float = 1.0,
        alpha_core: float = 2.0,
        w_eff: float = 0.5,
        w_fric: float = 0.3,
        mass: float = 1.0,
        friction_gamma: float = 0.5,
        beta_metric: float = 0.2
    ):
        self.core_origin = core_origin
        self.core_radius = core_radius
        self.w0_core = w0_core
        self.alpha_core = alpha_core
        self.w_eff = w_eff
        self.w_fric = w_fric
        self.mass = mass
        self.friction_gamma = friction_gamma
        self.beta_metric = beta_metric

        self.attractors: List[ConceptAttractor] = []
        self.mitosis_engine = CognitiveMitosisEngine()
        self.fusion_engine = CognitiveFusionEngine()

    def register_attractor(self, attractor: ConceptAttractor):
        self.attractors.append(attractor)

    def calculate_w_core(self, pos: np.ndarray) -> float:
        """Calculates non-linear core weight w_core(x) with explosive divergence near r_core."""
        r = np.linalg.norm(pos - self.core_origin)
        dist_to_core = r - self.core_radius
        if dist_to_core <= 0.01:
            return 1e4 # Divergent barrier
        return self.w0_core + (self.alpha_core / dist_to_core)

    def compute_potential(self, pos: np.ndarray, t: float = 0.0) -> float:
        """Computes composite scalar potential V(pos, t)."""
        r = np.linalg.norm(pos - self.core_origin)

        # 1. Core potential: origin attractor well + log barrier near r_core
        dist_to_core = max(0.001, r - self.core_radius)
        v_core_well = 5.0 * np.exp(- (r**2) / (2 * (1.5**2)))
        v_core_barrier = - 1.0 / dist_to_core if r < self.core_radius * 1.5 else 0.0
        v_core = v_core_well + v_core_barrier
        w_core_val = self.calculate_w_core(pos)

        # 2. Computational efficiency potential (smooth quadratic well)
        v_eff = 2.0 / (1.0 + 0.1 * (r ** 2))

        # 3. External friction attenuation potential
        v_fric = 1.0 * np.exp(-0.2 * r)

        # 4. Sum of registered concept attractor wells
        v_attractors = 0.0
        for att in self.attractors:
            d = np.linalg.norm(pos - att.center_pos)
            if d < att.well_radius * 3.0:
                v_attractors += att.well_depth * np.exp(- (d**2) / (2 * (att.well_radius**2)))

        return (w_core_val * v_core) + (self.w_eff * v_eff) + (self.w_fric * v_fric) + v_attractors

    def compute_gradient(self, pos: np.ndarray, t: float = 0.0, eps: float = 1e-4) -> np.ndarray:
        """Computes potential gradient ∇V(pos) via finite differencing."""
        grad = np.zeros(3)
        for i in range(3):
            p_plus, p_minus = pos.copy(), pos.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            grad[i] = (self.compute_potential(p_plus, t) - self.compute_potential(p_minus, t)) / (2 * eps)
        return grad

    def compute_metric_tensor(self, pos: np.ndarray, ltp_engine: Optional[LTPScarringEngine] = None) -> np.ndarray:
        """
        Computes valence-weighted metric tensor g_ij(x).
        g_ij(x) = exp(-beta * V(x)) * I_3x3 + Scar_Tensor
        Higher V(x) -> smaller ds (less resistance, rapid convergence).
        Lower V(x) -> larger ds (strong causal resistance).
        """
        v_val = self.compute_potential(pos)
        scale = np.exp(-self.beta_metric * v_val)
        g_tensor = np.eye(3) * scale

        # Apply LTP scar weight deformation if available
        if ltp_engine and ltp_engine.scar_weights:
            total_scar = sum(ltp_engine.scar_weights.values())
            # Scar reduces spatial impedance in local channel
            g_tensor *= (1.0 / (1.0 + 0.2 * total_scar))

        return g_tensor

    def step_motion_rk4(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        wave: Optional[RefractedTensionWave] = None,
        dt: float = 0.05,
        t: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes one RK4 step for valence gradient motion equation:
        m * d^2x/dt^2 + gamma * dx/dt = ∇V(x, t) + F_resonance + F_refraction
        No discrete conditionals (no if-else reasoning).
        """
        def acceleration(p: np.ndarray, v: np.ndarray) -> np.ndarray:
            grad_v = self.compute_gradient(p, t)

            # Resonance pull force from attractors
            res_force = np.zeros(3)
            if wave:
                for att in self.attractors:
                    dist = np.linalg.norm(p - att.center_pos)
                    res_ratio = 1.0 / (1.0 + 0.1 * (wave.internal_frequency - att.resonant_freq)**2)
                    if res_ratio > 0.3 and dist > 0.01:
                        dir_to_att = (att.center_pos - p) / dist
                        res_force += dir_to_att * res_ratio * att.well_depth * 0.8

            refract_force = (wave.refraction_dir * wave.amplitude * 0.2) if wave else np.zeros(3)

            total_force = grad_v + res_force + refract_force - (self.friction_gamma * v)
            return total_force / self.mass

        # RK4 integration
        k1_v = acceleration(pos, vel)
        k1_p = vel

        k2_v = acceleration(pos + 0.5 * dt * k1_p, vel + 0.5 * dt * k1_v)
        k2_p = vel + 0.5 * dt * k1_v

        k3_v = acceleration(pos + 0.5 * dt * k2_p, vel + 0.5 * dt * k2_v)
        k3_p = vel + 0.5 * dt * k2_v

        k4_v = acceleration(pos + dt * k3_p, vel + dt * k3_v)
        k4_p = vel + dt * k3_v

        next_pos = pos + (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        next_vel = vel + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        return next_pos, next_vel

    def process_and_perceive(
        self,
        wave: RefractedTensionWave,
        start_pos: np.ndarray,
        steps: int = 40,
        dt: float = 0.05
    ) -> Tuple[np.ndarray, ConceptAttractor, bool]:
        """
        Executes continuous gradient flow thought trajectory relaxation.
        If converged point lies outside all existing concept attractor wells,
        triggers Cognitive Mitosis to form a new ConceptAttractor automatically.
        """
        pos = start_pos.copy()
        vel = np.zeros(3)

        for step in range(steps):
            pos, vel = self.step_motion_rk4(pos, vel, wave=wave, dt=dt, t=step*dt)

        # Evaluate convergence against attractors
        matched_attractor = None
        nearest_attractor = None
        min_dist = float('inf')

        for att in self.attractors:
            d = np.linalg.norm(pos - att.center_pos)
            if d < min_dist:
                min_dist = d
                nearest_attractor = att
            if d <= att.well_radius:
                matched_attractor = att

        if matched_attractor is not None:
            return pos, matched_attractor, False

        # Unmapped state reached -> Cognitive Mitosis
        new_attractor = self.mitosis_engine.trigger_mitosis(
            converged_pos=pos,
            wave=wave,
            nearest_attractor=nearest_attractor
        )
        self.register_attractor(new_attractor)
        return pos, new_attractor, True

    def consolidate_field(self) -> List[Tuple[str, str, str]]:
        """Scans and consolidates interfering concept attractors via Cognitive Fusion."""
        fused_events = []
        merged_indices = set()
        new_attractors = []

        n = len(self.attractors)
        for i in range(n):
            if i in merged_indices:
                continue

            current_att = self.attractors[i]
            target_to_fuse = None
            max_sync_idx = 0.0

            for j in range(i + 1, n):
                if j in merged_indices:
                    continue

                sync_idx = self.fusion_engine.calculate_sync_index(current_att, self.attractors[j])
                if sync_idx > 0.4 and sync_idx > max_sync_idx:
                    max_sync_idx = sync_idx
                    target_to_fuse = j

            if target_to_fuse is not None:
                partner_att = self.attractors[target_to_fuse]
                fused_att = self.fusion_engine.fuse_attractors(current_att, partner_att)

                merged_indices.add(i)
                merged_indices.add(target_to_fuse)
                new_attractors.append(fused_att)

                fused_events.append((current_att.concept_id, partner_att.concept_id, fused_att.concept_id))
            else:
                new_attractors.append(current_att)

        self.attractors = new_attractors
        return fused_events
