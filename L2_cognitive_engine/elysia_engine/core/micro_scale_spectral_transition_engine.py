"""
L2_cognitive_engine/elysia_engine/core/micro_scale_spectral_transition_engine.py

Micro-Scale Spectral Transition & Continuous Phase Flow Engine
==============================================================
Principles (from THE_ABSOLUTE_COMMANDMENT & Causal Field):
1. No Hell of Dots & Zero Pixel Storage:
   - Does not store or accumulate massive discrete pixel arrays (RGB raster databases).
   - Governs reality through micro-scale transition rules and multi-scale relational scale rotors (w-axis).
   - Infinite resolution crystallization on demand directly from the underlying generating law Theta.
2. Micro-Scale Spectral Resolution as Transition Rules:
   - "High resolution" in luminance/spectral transition does not mean cramming in more pixels.
   - It means how finely and continuously the transition steps and gradient phases can be parsed.
   - Microscopic texture gradients, optical edge cliff formation, and macroscopic illumination flows
     are unified into a single continuous phase transition dynamics.
3. Scale Rotor & Dimension Control:
   - Not a toy knob to juggle pixels, but a multi-scale dimensional operator spanning micro to macro.
   - Dialing the scale knob smoothly traverses between micro-texture diffraction, meso-structural edge boundaries,
     and macro-environmental illumination waves.
4. Causal Anomaly & Phase Rupture Friction (Q_heat):
   - Natural reality preserves continuous phase conservation and smooth gradient transitions.
   - Artificial distortions, structural joint breaks, or temporal glitches produce non-differentiable
     discontinuities, phase ruptures, and high friction tension Q_heat = ||Delta_phase|| * Tension.
   - Serves as the fundamental metric to discern genuine physical reality from artificial hallucinations.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math
import numpy as np

from elysia_engine.core.hyper_dimensional_blister_nebula import (
    Clifford4DRotor,
    HyperDimensionalOpticalRotor,
    HyperDimensionalBlisterKernel,
    OpticalHyperSlicer
)
from elysia_engine.core.field_seed_manifest import (
    FieldSeed,
    RotorSeedDescriptor,
    FieldKernelDescriptor
)


@dataclass
class MicroTransitionLaw:
    """
    Generating Law Theta for continuous micro-scale spectral transition.
    Pure invariant law without storing any pixel arrays.
    """
    spectral_frequency: float = 4.2      # Frequency of micro-gradient oscillations
    damping_factor: float = 0.85          # Natural spatial attenuation coefficient
    chromatic_coherence: float = 0.92     # Triadic phase locking ratio (Red-Green-Blue coupling)
    stiction_threshold: float = 0.15      # Boundary stiction preventing hunting noise
    edge_cliff_exponent: float = 3.2      # Sharp optical edge gradient preservation exponent
    temporal_frequency: float = 1.0       # Temporal phase progression speed


@dataclass
class PhaseTransitionAssessment:
    """
    Assessment result of field continuity and causal integrity.
    """
    is_causally_natural: bool
    phase_rupture_count: int
    mean_phase_friction: float
    max_phase_friction: float
    spectral_entropy: float
    energy_conservation_score: float
    anomaly_map: Optional[np.ndarray] = None


class MicroScaleSpectralTransitionEngine:
    """
    Micro-Scale Spectral Transition & Continuous Phase Flow Engine.
    Employs Clifford bivectors and multi-scale rotors to project arbitrary-resolution
    continuous optical spectrum fields without storing pixel arrays.
    """

    def __init__(
        self,
        law: Optional[MicroTransitionLaw] = None,
        base_w: float = 1.0
    ):
        self.law = law or MicroTransitionLaw()
        self.scale_w = base_w # Relational Scale Rotor position
        self.phase_theta = 0.0 # Global phase flow state

    def rotate_scale_rotor(self, delta_w: float) -> float:
        """
        Adjusts the relational scale rotor (w-axis).
        Transitions smoothly from micro-texture to macroscopic illumination field.
        """
        self.scale_w = max(0.01, min(100.0, self.scale_w + delta_w))
        return self.scale_w

    def set_scale_rotor(self, w: float) -> float:
        """Directly sets the relational scale rotor value."""
        self.scale_w = max(0.01, min(100.0, float(w)))
        return self.scale_w

    def advance_phase(self, dt: float) -> float:
        """Advances global temporal phase."""
        self.phase_theta = (self.phase_theta + dt * self.law.temporal_frequency) % (2.0 * math.pi)
        return self.phase_theta

    def crystallize_field(
        self,
        resolution_wh: Tuple[int, int] = (128, 72),
        t: float = 0.0,
        rotors: Optional[Sequence[RotorSeedDescriptor]] = None,
        custom_scale_w: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Crystallizes continuous optical field on-demand at target resolution directly
        from the underlying generating law Theta, without referencing stored pixel buffers.

        Returns:
            Dict containing:
                - "framebuffer_rgb": (H, W, 3) uint8 image
                - "phase_gradient": (H, W) float32 magnitude
                - "spectral_field": (H, W, 3) float32 normalized spectrum
        """
        width, height = resolution_wh
        w_scale = custom_scale_w if custom_scale_w is not None else self.scale_w

        # Continuous normalized coordinate grid [-1, 1]
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)

        # Distance from center and angular coordinate
        radius = np.sqrt(grid_x**2 + grid_y**2)
        angle = np.arctan2(grid_y, grid_x)

        # Micro-scale wave propagation governed by Law Theta & Scale Rotor w
        # Multi-scale decomposition: Micro wavelets modulated by macro envelope
        effective_freq = self.law.spectral_frequency * w_scale
        base_phase = self.phase_theta + t * self.law.temporal_frequency

        # Continuous phase field formulation
        phase_r = np.sin(effective_freq * radius - base_phase + angle * 2.0)
        phase_g = np.sin(effective_freq * radius * 1.15 - base_phase * 1.08 + angle * 2.0 + 2.0 * math.pi / 3.0)
        phase_b = np.sin(effective_freq * radius * 1.30 - base_phase * 0.95 + angle * 2.0 + 4.0 * math.pi / 3.0)

        # Incorporate rotor seeds if provided (Clifford bivector harmonics)
        if rotors:
            for i, r_desc in enumerate(rotors):
                pos = r_desc.position
                dist_r = np.sqrt((grid_x - pos[0])**2 + (grid_y - pos[1])**2) + 0.001
                # Non-linear blister kernel stiction
                envelope = np.exp(-self.law.damping_factor * dist_r * w_scale)
                bivector_phase = r_desc.bivector_angles[0] + r_desc.bivector_angles[1] + base_phase
                harmonic = np.cos(r_desc.dominant_frequency * dist_r - bivector_phase)

                # Coupled chromatic triad injection
                phase_r += float(r_desc.triad_weights[0]) * harmonic * envelope
                phase_g += float(r_desc.triad_weights[1]) * harmonic * envelope
                phase_b += float(r_desc.triad_weights[2]) * harmonic * envelope

        # Apply Edge Cliff Non-linear Shaping (preserves sharp optical boundaries without blur)
        def _shape_cliff(val: np.ndarray) -> np.ndarray:
            norm_val = 0.5 * (val + 1.0)
            norm_val = np.clip(norm_val, 0.0, 1.0)
            # S-curve cliff transition
            return np.where(
                norm_val > 0.5,
                1.0 - 0.5 * ((2.0 * (1.0 - norm_val)) ** self.law.edge_cliff_exponent),
                0.5 * ((2.0 * norm_val) ** self.law.edge_cliff_exponent)
            )

        spec_r = _shape_cliff(phase_r)
        spec_g = _shape_cliff(phase_g)
        spec_b = _shape_cliff(phase_b)

        spectral_field = np.stack([spec_r, spec_g, spec_b], axis=-1).astype(np.float32)

        # Calculate phase gradient magnitude for edge continuity check
        grad_y, grad_x = np.gradient(spec_r)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Transduce to RGB [0, 255]
        framebuffer_rgb = (spectral_field * 255.0).clip(0, 255).astype(np.uint8)

        return {
            "framebuffer_rgb": framebuffer_rgb,
            "phase_gradient": grad_mag.astype(np.float32),
            "spectral_field": spectral_field
        }

    def assess_causal_continuity(
        self,
        frame_t0: np.ndarray,
        frame_t1: np.ndarray,
        expected_flow_rate: float = 1.0,
        stiction_friction_threshold: float = 0.02
    ) -> PhaseTransitionAssessment:
        """
        Assesses whether the transition between frame_t0 and frame_t1 conforms to
        natural physical phase continuity or suffers from artificial rupture/causal anomalies.

        Inputs:
            frame_t0, frame_t1: RGB or normalized spectral arrays of shape (H, W, 3).
        """
        # Normalize to float [0, 1] if uint8
        f0 = frame_t0.astype(np.float32) / (255.0 if frame_t0.dtype == np.uint8 else 1.0)
        f1 = frame_t1.astype(np.float32) / (255.0 if frame_t1.dtype == np.uint8 else 1.0)

        # Spectral difference vector
        diff = f1 - f0

        # Spatial gradient of the difference (measures phase rupture / non-smooth discontinuity)
        diff_lum = np.mean(diff, axis=-1)
        dy, dx = np.gradient(diff_lum)
        rupture_gradient = np.sqrt(dx**2 + dy**2)

        # Local friction tension Q_heat = ||Delta_phase|| * Tension
        phase_friction = rupture_gradient * (np.abs(diff_lum) + 0.1)

        mean_friction = float(np.mean(phase_friction))
        max_friction = float(np.max(phase_friction))

        # Detect rupture points where phase gradient exceeds natural stiction barrier
        rupture_mask = phase_friction > stiction_friction_threshold
        rupture_count = int(np.sum(rupture_mask))

        # Spectral entropy (spread of unexpected frequency noise)
        hist, _ = np.histogram(phase_friction, bins=20, range=(0.0, 1.0), density=True)
        hist = hist[hist > 0]
        spectral_entropy = float(-np.sum(hist * np.log2(hist + 1e-12)) * (1.0 / 20.0))

        # Energy conservation score: smooth natural flows preserve energy, ruptures bleed/spike
        flux_delta = np.abs(np.mean(f1) - np.mean(f0))
        energy_conservation = max(0.0, 1.0 - float(flux_delta * 3.0 + mean_friction * 2.0))

        # Natural physical flow criterion
        is_natural = (rupture_count < 0.05 * (f0.shape[0] * f0.shape[1])) and (mean_friction < stiction_friction_threshold)

        return PhaseTransitionAssessment(
            is_causally_natural=is_natural,
            phase_rupture_count=rupture_count,
            mean_phase_friction=mean_friction,
            max_phase_friction=max_friction,
            spectral_entropy=spectral_entropy,
            energy_conservation_score=energy_conservation,
            anomaly_map=rupture_mask.astype(np.uint8) * 255
        )

    def inject_causal_distortion_benchmark(
        self,
        clean_frame: np.ndarray,
        distortion_type: str = "joint_collapse"
    ) -> np.ndarray:
        """
        Creates synthetic benchmark frames containing known artificial generative distortions
        (e.g., collapsed anatomical joints, non-physical boundary tearing, or salt-and-pepper pixel noise)
        to calibrate and verify the causal anomaly detection engine.
        """
        distorted = clean_frame.copy()
        h, w = distorted.shape[:2]

        if distortion_type == "joint_collapse":
            # An unnatural dislocation / sharp discontinuity cutting across boundaries
            x_mid = w // 2
            y_mid = h // 2
            patch_size = min(h, w) // 4
            # Break phase continuity by shifting a block non-isomorphically
            patch = distorted[y_mid:y_mid+patch_size, x_mid:x_mid+patch_size].copy()
            distorted[y_mid:y_mid+patch_size, x_mid:x_mid+patch_size] = np.roll(patch, shift=patch_size // 2, axis=0)
            # Add artificial sharp seam
            distorted[y_mid:y_mid+patch_size, x_mid] = 255 if clean_frame.dtype == np.uint8 else 1.0

        elif distortion_type == "spectral_noise":
            # Discrete high-frequency pixel hash (the 'hell of dots' noise)
            noise = (np.random.rand(h, w, 3) > 0.95).astype(clean_frame.dtype)
            if clean_frame.dtype == np.uint8:
                distorted = np.where(noise > 0, 255, distorted)
            else:
                distorted = np.where(noise > 0, 1.0, distorted)

        elif distortion_type == "phase_tearing":
            # Artificial tearing across scanlines
            for y in range(0, h, 8):
                shift = int(10 * math.sin(y))
                distorted[y:y+4] = np.roll(distorted[y:y+4], shift=shift, axis=1)

        return distorted
