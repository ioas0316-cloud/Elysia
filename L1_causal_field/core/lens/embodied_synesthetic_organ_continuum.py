#!/usr/bin/env python3
"""
embodied_synesthetic_organ_continuum.py

Layer 1 Core Lens: Embodied Intentional Organ Continuum Engine.

Embodying the non-reductionist, enactive phenomenon where the body ($0_{\text{self}}$) and world
interact through multi-sensory synesthetic resonance:
- $0_{\text{self}}$ (Internal Lack Baseline) and $\Delta V$ (World Phase Divergence) drive intentionality.
- Somatic Friction ($Q_{\text{heat}}$, postural tension), Acoustic Resonant Timbre (`SoundAcousticsLens`),
  Prismatic Refraction (`SensoryPrismaticGate`), Dynamic Aperture & Focal Depth Lens (`CausalLensDifferential`),
  and Focal Plane Image Formation & Grounded Unit '1' Engine (`UnitSpectrumObservationEngine`) form a single
  organ continuum.
- Scale Horizon expands and contracts (Micro <-> Meso <-> Macro) dynamically while conserving the invariant backbone.
"""

from dataclasses import dataclass, field
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Import component organ lenses
from L1_causal_field.core.layer_1_constraint.sensory_prismatic_gate import SensoryPrismaticGate
from L1_causal_field.core.physics.causal_lens_differential import (
    PrimitiveSubstrate,
    StructuralMechanics,
    SoundAcousticsLens,
    VisionOpticsLens,
    ModalityType,
)
from L1_causal_field.core.consciousness.unit_spectrum_observation_engine import (
    UnitSpectrumObservationEngine,
    GroundedCognitiveUnit,
)
from L1_causal_field.core.topology.particle_rotor_matrix_sandbox import (
    ParticleRotorMatrixSandbox,
    MicroRotorParticle,
)


@dataclass
class EmbodiedOrganState:
    """
    Represents the holistic, enactive physical-mental state of the body organ continuum.
    """
    internal_lack_zero: float          # 0_self baseline energy deficit / tension
    world_phase_divergence: float      # Delta V: Phase divergence between world and internal archetype
    postural_tension: float            # Somatic muscle/neck/body posture tension
    somatic_friction_heat: float       # Q_heat: Friction heat generated at boundary
    acoustic_resonance_phase: float    # Acoustic timbre & phase lock angle
    focal_depth: float                 # Aperture focal depth (scale zoom factor)
    aperture_size: float               # Aperture opening ratio (0.1 = narrow focus, 1.0 = wide field)
    chromatic_spectrum: np.ndarray     # Prismatic Red/Blue/Yellow spectral energy
    image_focal_plane: np.ndarray      # Formed image / mental field on focal plane
    bound_grounded_units: List[GroundedCognitiveUnit] = field(default_factory=list)
    invariant_backbone_fidelity: float = 1.0


class EmbodiedSynestheticOrganContinuum:
    """
    Enactive Organ Continuum Engine where intentionality ($0_{\text{self}}$) and world friction ($\Delta V$)
    trigger multi-sensory synesthetic resonance across Somatic, Acoustic, Optical, and Scale dimensions.
    """

    def __init__(
        self,
        internal_lack_zero: float = 0.5,
        base_focal_length: float = 1.0,
        num_particles: int = 100,
    ):
        self.internal_lack_zero = internal_lack_zero
        self.base_focal_length = base_focal_length

        # Sub-organ engines
        self.prismatic_gate = SensoryPrismaticGate(num_spectral_bands=3, lens_density=1000)
        self.sound_lens = SoundAcousticsLens(intent_weight=1.0)
        self.vision_lens = VisionOpticsLens(intent_weight=1.0)
        self.observation_engine = UnitSpectrumObservationEngine(
            observation_scale=base_focal_length, coherence_threshold=0.3
        )

        # Micro-rotor particle sandbox representing somatic substrate
        self.sandbox = ParticleRotorMatrixSandbox(dimensions=2, bounds=10.0)
        self._init_somatic_sandbox(num_particles)

    def _init_somatic_sandbox(self, num_particles: int) -> None:
        """Initialize somatic sandbox with micro-rotor particles."""
        np.random.seed(42)
        for i in range(num_particles):
            pos = np.random.uniform(-8.0, 8.0, size=2)
            vel = np.random.randn(2) * 0.1
            theta = float(np.random.uniform(0, 2 * np.pi))
            omega = float(np.random.uniform(0.5, 2.0))
            p = MicroRotorParticle(
                id=i,
                position=pos,
                velocity=vel,
                mass=float(np.random.uniform(0.8, 1.5)),
                friction=float(np.random.uniform(0.05, 0.2)),
                theta=theta,
                omega=omega,
                type_id=i % 3,
            )
            self.sandbox.add_particle(p)

    def perceive_and_resonate(self, external_world_wave: np.ndarray) -> EmbodiedOrganState:
        """
        Process incoming world wave field through the enactive organ continuum:
        1. Measure Phase Divergence ($\Delta V$) between world wave and internal state.
        2. Driven by $0_{\text{self}}$ + $\Delta V$, adjust postural tension and somatic friction ($Q_{\text{heat}}$).
        3. Refract wave through Prismatic Gate into chromatic spectral bands.
        4. Pass through SoundAcousticsLens to extract acoustic timbre phase resonance.
        5. Adjust aperture size and focal depth (scale zoom) dynamically.
        6. Form image on Focal Plane and bind Grounded Cognitive Units ('1').
        """
        raw_wave = np.atleast_1d(external_world_wave).astype(float)
        if len(raw_wave) == 0:
            raw_wave = np.sin(np.linspace(0, 2 * np.pi, 1000))

        # Step 1: Measure Phase Divergence Delta V
        internal_phase = self.sandbox.get_macro_order_parameter()
        world_energy = float(np.mean(raw_wave**2))
        world_freq = float(np.abs(np.fft.rfft(raw_wave)[1])) if len(raw_wave) > 1 else 1.0
        world_phase = float(np.sin(world_freq))

        delta_v = float(abs(internal_phase - world_phase) + 0.1 * world_energy)

        # Step 2: Postural tension and Somatic Friction (Q_heat)
        # Higher lack (0_self) and divergence (Delta V) induce somatic friction & tension
        somatic_tension = math.tanh(self.internal_lack_zero + delta_v)
        q_heat = float(delta_v * (1.0 + somatic_tension) * 2.5)

        # Step 3: Advance somatic sandbox with friction perturbation
        for p in self.sandbox.particles:
            p.friction = float(np.clip(p.friction + 0.01 * q_heat, 0.01, 1.0))
            p.velocity += np.random.randn(2) * 0.05 * somatic_tension
        sandbox_res = self.sandbox.step()

        # Step 4: Prismatic Refraction Gate
        prismatic_res = self.prismatic_gate.refract_external_aperture(raw_wave)
        chromatic_spec = prismatic_res["chromatic_spectrum"]

        # Step 5: Sound & Acoustics Lens
        substrate = PrimitiveSubstrate(
            substrate_id="somatic_sub",
            energy_density=world_energy,
            tension_potential=somatic_tension,
            repulsion_potential=0.5 * delta_v,
            state_vector=np.array([world_energy, somatic_tension, delta_v, q_heat, 0, 0, 0, 0][:8])
        )
        mechanics = StructuralMechanics(
            mechanism_id="somatic_mech",
            modality=ModalityType.SOUND_ACOUSTICS,
            tension_coefficient=somatic_tension,
            repulsion_coefficient=0.5,
            boundary_curvature=1.0 + somatic_tension
        )
        sound_construct = self.sound_lens.refract_substrate(substrate, mechanics)
        acoustic_resonance = float(sound_construct.form_tensor[1]) # resonance phase

        # Step 6: Dynamic Scale Zoom (Aperture & Focal Depth)
        # Deep focus when tension/divergence is high; wide macro field when relaxed
        dynamic_focal_scale = float(self.base_focal_length / (1.0 + 2.0 * somatic_tension))
        aperture_opening = float(np.clip(1.0 / (1.0 + delta_v), 0.1, 1.0))

        # Step 7: Focal Plane Image Formation & Grounded Unit '1' Binding
        self.observation_engine.set_observation_scale(dynamic_focal_scale)
        grounded_units = self.observation_engine.observe_and_bind_units(self.sandbox)

        image_focal_plane = prismatic_res["mental_imagery_field"] * aperture_opening

        # Invariant backbone fidelity (energy/coherence ratio)
        coherence = prismatic_res["phase_coherence"]
        invariant_fidelity = float(np.clip(1.0 - 0.02 * (1.0 - coherence) - 0.01 * q_heat, 0.90, 1.0))

        return EmbodiedOrganState(
            internal_lack_zero=self.internal_lack_zero,
            world_phase_divergence=delta_v,
            postural_tension=somatic_tension,
            somatic_friction_heat=q_heat,
            acoustic_resonance_phase=acoustic_resonance,
            focal_depth=dynamic_focal_scale,
            aperture_size=aperture_opening,
            chromatic_spectrum=chromatic_spec,
            image_focal_plane=image_focal_plane,
            bound_grounded_units=grounded_units,
            invariant_backbone_fidelity=invariant_fidelity,
        )


if __name__ == "__main__":
    organ = EmbodiedSynestheticOrganContinuum(internal_lack_zero=0.6)
    test_wave = np.sin(np.linspace(0, 8 * np.pi, 1000)) + 0.3 * np.random.randn(1000)
    state = organ.perceive_and_resonate(test_wave)

    print("=== Embodied Synesthetic Organ Continuum Verification ===")
    print(f"Internal Lack (0_self): {state.internal_lack_zero:.4f}")
    print(f"World Phase Divergence (Delta V): {state.world_phase_divergence:.4f}")
    print(f"Postural Tension: {state.postural_tension:.4f}")
    print(f"Somatic Friction Heat (Q_heat): {state.somatic_friction_heat:.4f}")
    print(f"Acoustic Resonance Phase: {state.acoustic_resonance_phase:.4f}")
    print(f"Dynamic Focal Depth (Scale): {state.focal_depth:.4f}")
    print(f"Aperture Size: {state.aperture_size:.4f}")
    print(f"Chromatic Spectrum (Red/Blue/Yellow): {state.chromatic_spectrum}")
    print(f"Bound Grounded Units ('1'): {len(state.bound_grounded_units)}")
    print(f"Invariant Backbone Fidelity: {state.invariant_backbone_fidelity * 100:.2f}%")
