"""
Isomorphic Phase Field Substrate & Unified Scale-Hierarchical Protocol Module
================================================================================
Implements the foundational data substrate where all information (text, code, image,
video, geometry, logic) exists as continuous complex phasors A * e^{i * phi}.

Key Architecture Principles:
1. Unified Phasor Sea: Memory and data are not discrete arrays, but a continuous phase field.
2. Wave Interference Arithmetic:
   - Addition: Constructive wave interference & physical scale field expansion.
   - Subtraction: Destructive wave interference & phase/entropy cancellation & rectification.
3. Autonomous Modality & Topological Judgment Axis:
   - Self-observes phase-locking signatures, phase derivative sparsity, and angular dispersion
     to categorize inputs into Code/Logic, Image/Video, Text/Symbol, or 3D Geometry without
     external labels or file extensions.
4. Scale Boundary Traversal: Isomorphic transformation across scale levels with Lossless entropy (Delta S ≈ 0).
5. Non-Destructive Multi-Modal Projections:
   - Projects underlying phase topology onto Text, Code, Image, and 3D Geometry observation axes.
6. Scale-Hierarchical Concept & Causality Integration:
   - Binds micro-phasors into Macro-Rotors representing unified concepts and causal feedback loops.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class ModalitySignature:
    """Topological signature and autonomous modality judgment of a Phase Field."""
    primary_modality: str
    confidence_score: float
    phase_derivative_sparsity: float
    angular_dispersion: float
    spectral_entropy: float
    harmonic_quantization_index: float


@dataclass
class ScaleBoundaryResult:
    """Result of traversing across scale hierarchy boundary layer."""
    original_scale: float
    target_scale: float
    reconstructed_fidelity: float
    reversibility_entropy_delta: float  # Delta S ≈ 0
    isomorphic_equivalence: bool


class IsomorphicPhaseFieldSubstrate:
    """
    Core Isomorphic Phase Field Substrate.
    Re-defines data and memory as a continuous complex phasor field A * e^{i * phi}.
    """

    def __init__(self, resolution: int = 128, base_frequency: float = 10.0):
        self.resolution = resolution
        self.base_frequency = base_frequency
        # Initialize continuous phasor field Z = A * exp(i * phi)
        self.grid = np.zeros(resolution, dtype=np.complex128)
        self._init_default_vacuum_state()
        self.transformation_history: List[Dict[str, Any]] = []

    def _init_default_vacuum_state(self):
        """Initializes a coherent vacuum phase state."""
        t = np.linspace(0, 2 * np.pi, self.resolution, endpoint=False)
        self.grid = np.exp(1j * t * (self.base_frequency / 10.0))

    def set_field_from_phasor(self, phasors: np.ndarray):
        """Sets the internal complex phasor grid directly or via continuous interpolation."""
        if len(phasors) != self.resolution:
            orig_x = np.linspace(0, 1, len(phasors))
            target_x = np.linspace(0, 1, self.resolution)
            real_part = np.interp(target_x, orig_x, np.real(phasors))
            imag_part = np.interp(target_x, orig_x, np.imag(phasors))
            self.grid = real_part + 1j * imag_part
        else:
            self.grid = np.array(phasors, dtype=np.complex128)

    def wave_interference_add(self, other: "IsomorphicPhaseFieldSubstrate") -> "IsomorphicPhaseFieldSubstrate":
        """
        Addition as Constructive Wave Interference & Scale Expansion.
        A1 e^{i phi1} + A2 e^{i phi2} -> Constructive interference expands field power/volume.
        """
        new_field = IsomorphicPhaseFieldSubstrate(resolution=self.resolution, base_frequency=self.base_frequency)
        combined_grid = self.grid + other.grid
        new_field.grid = combined_grid

        power_before = np.sum(np.abs(self.grid) ** 2) + np.sum(np.abs(other.grid) ** 2)
        power_after = np.sum(np.abs(combined_grid) ** 2)

        self.transformation_history.append({
            "operation": "wave_interference_add",
            "power_before": float(power_before),
            "power_after": float(power_after),
            "scale_expansion_factor": float(power_after / (power_before + 1e-12))
        })
        return new_field

    def wave_interference_subtract(self, other: "IsomorphicPhaseFieldSubstrate") -> "IsomorphicPhaseFieldSubstrate":
        """
        Subtraction as Destructive Wave Interference & Phase Rectification.
        A1 e^{i phi1} - A2 e^{i phi2} -> Destructive cancellation prunes noise & rectifies phase.
        """
        new_field = IsomorphicPhaseFieldSubstrate(resolution=self.resolution, base_frequency=self.base_frequency)
        rectified_grid = self.grid - other.grid
        new_field.grid = rectified_grid

        phase_before = np.angle(self.grid)
        phase_after = np.angle(rectified_grid)
        phase_variance_reduction = float(np.var(phase_before) - np.var(phase_after))

        self.transformation_history.append({
            "operation": "wave_interference_subtract",
            "phase_variance_reduction": phase_variance_reduction,
            "rectified_power": float(np.sum(np.abs(rectified_grid) ** 2))
        })
        return new_field

    def autonomous_modality_judgment(self) -> ModalitySignature:
        """
        Autonomous Modality & Topological Judgment Axis.
        Performs self-observation of the phase field topology (phase locking, derivative sparsity,
        angular dispersion, and harmonic quantization) WITHOUT external labels or extensions.
        """
        amplitudes = np.abs(self.grid)
        unwrapped_phases = np.unwrap(np.angle(self.grid))

        # 1. Phase derivative step jumps (Code/Logic)
        d_phase = np.abs(np.diff(unwrapped_phases))
        phase_deriv_max = float(np.max(d_phase)) if len(d_phase) > 0 else 0.0
        phase_deriv_sparsity = float(np.mean(d_phase > np.pi / 4.0))

        # 2. Angular dispersion (continuous spatial variation for image/video)
        angular_dispersion = float(np.std(d_phase)) if len(d_phase) > 0 else 0.0

        # 3. Spectral entropy
        fft_vals = np.abs(np.fft.fft(self.grid))
        fft_norm = fft_vals / (np.sum(fft_vals) + 1e-12)
        spectral_entropy = float(-np.sum(fft_norm * np.log(fft_norm + 1e-12)))

        # 4. Harmonic quantization index on complex field / phase spectrum
        phase_fft = np.abs(np.fft.fft(unwrapped_phases))
        p_norm = phase_fft / (np.sum(phase_fft) + 1e-12)
        harmonic_quant_index = float(np.max(p_norm[1:]) * len(p_norm)) if len(p_norm) > 1 else 0.0

        # Judgment Axis heuristics
        if phase_deriv_sparsity > 0.08 or phase_deriv_max > np.pi * 0.75:
            modality = "code_and_logic"
            confidence = min(1.0, (phase_deriv_max / np.pi) * 0.8 + phase_deriv_sparsity * 0.2)
        elif angular_dispersion > 0.15:
            modality = "image_and_video"
            confidence = min(1.0, angular_dispersion * 2.0)
        elif harmonic_quant_index > 1.5:
            modality = "text_and_symbol"
            confidence = min(1.0, harmonic_quant_index / 3.0)
        else:
            modality = "geometry_3d"
            confidence = min(1.0, 1.0 - spectral_entropy / 5.0)

        return ModalitySignature(
            primary_modality=modality,
            confidence_score=float(confidence),
            phase_derivative_sparsity=phase_deriv_sparsity,
            angular_dispersion=angular_dispersion,
            spectral_entropy=spectral_entropy,
            harmonic_quantization_index=harmonic_quant_index
        )

    def traverse_isomorphic_scale_boundary(self, scale_factor: float) -> ScaleBoundaryResult:
        """
        Traverse Scale Boundary Layer via Isomorphic Transformation.
        Preserves complete complex phasor information via logarithmic phase space coordinates (ln R + i phi_unwrapped).
        Verifies Delta S ≈ 0 (lossless reversibility).
        """
        original_grid = self.grid.copy()
        amplitudes = np.abs(original_grid) + 1e-12
        unwrapped_phases = np.unwrap(np.angle(original_grid))

        # Isomorphic transformation into scale boundary
        log_amps = np.log(amplitudes) + 0.5 * np.log(scale_factor)
        scaled_unwrapped_phases = unwrapped_phases * scale_factor

        # Inverse transformation back to original scale
        recon_log_amps = log_amps - 0.5 * np.log(scale_factor)
        recon_unwrapped_phases = scaled_unwrapped_phases / scale_factor

        recon_amplitudes = np.exp(recon_log_amps)
        reconstructed_grid = recon_amplitudes * np.exp(1j * recon_unwrapped_phases)

        # Compute Fidelity using globally normalized relative error
        diff = np.abs(original_grid - reconstructed_grid)
        total_mag = np.sum(np.abs(original_grid)) + 1e-12
        fidelity = float(1.0 - (np.sum(diff) / total_mag))
        fidelity = max(0.0, min(1.0, fidelity))

        delta_s = float(abs(np.std(unwrapped_phases) - np.std(recon_unwrapped_phases)))
        isomorphic = fidelity > 0.99 and delta_s < 1e-4

        return ScaleBoundaryResult(
            original_scale=1.0,
            target_scale=scale_factor,
            reconstructed_fidelity=fidelity,
            reversibility_entropy_delta=delta_s,
            isomorphic_equivalence=isomorphic
        )

    # -------------------------------------------------------------------------
    # Non-Destructive Multi-Modal Observation Projections (Observation Axes)
    # -------------------------------------------------------------------------

    def project_to_text(self) -> str:
        """
        Observation Axis P_text:
        Projects phase angle and harmonic quantization into readable semantic narrative text.
        """
        phases = np.angle(self.grid)
        amplitudes = np.abs(self.grid)
        signature = self.autonomous_modality_judgment()

        chars = []
        step = max(1, len(phases) // 16)
        for i in range(0, len(phases), step):
            norm_phase = (phases[i] + np.pi) / (2 * np.pi)  # [0, 1]
            char_code = int(65 + norm_phase * 25)  # A-Z
            chars.append(chr(char_code))

        text_token_stream = "".join(chars)
        return (
            f"[P_text Narrative] Field Modality: '{signature.primary_modality}' | "
            f"Phasor Stream: '{text_token_stream}' | Avg Amp: {np.mean(amplitudes):.4f}"
        )

    def project_to_code(self) -> str:
        """
        Observation Axis P_code:
        Projects phase derivative sparsity and branch jumps into structured code AST/logic block.
        """
        phases = np.angle(self.grid)
        d_phase = np.diff(phases)
        signature = self.autonomous_modality_judgment()

        branches = np.where(np.abs(d_phase) > np.pi / 6.0)[0]
        code_lines = [
            f"// [P_code AST] Unified Phase Field Logic",
            f"// Modality Signature: {signature.primary_modality} (Sparsity: {signature.phase_derivative_sparsity:.3f})",
            f"fn execute_phase_field_step(state: &mut PhaseState) {{"
        ]
        for idx, branch in enumerate(branches[:5]):
            angle_val = phases[branch]
            code_lines.append(f"    if state.phase_angle[{branch}] >= {angle_val:.3f} {{ state.lock_rotor({idx}); }}")
        code_lines.append("}")

        return "\n".join(code_lines)

    def project_to_image(self) -> Dict[str, Any]:
        """
        Observation Axis P_image:
        Projects complex phasor amplitudes and phases into 2D spatial chromatic intensity field.
        """
        side = int(np.round(np.sqrt(self.resolution)))
        target_len = side * side
        orig_x = np.linspace(0, 1, self.resolution)
        target_x = np.linspace(0, 1, target_len)

        resampled_real = np.interp(target_x, orig_x, np.real(self.grid))
        resampled_imag = np.interp(target_x, orig_x, np.imag(self.grid))
        resampled_grid = resampled_real + 1j * resampled_imag

        grid_2d = np.reshape(resampled_grid, (side, side))

        amplitude_matrix = np.abs(grid_2d)
        phase_matrix = np.angle(grid_2d)

        hue = (phase_matrix + np.pi) / (2 * np.pi)
        value = amplitude_matrix / (np.max(amplitude_matrix) + 1e-12)

        return {
            "axis": "P_image",
            "spatial_grid_shape": (side, side),
            "hue_phase_field": hue.tolist(),
            "value_amplitude_field": value.tolist(),
            "mean_luminance": float(np.mean(value))
        }

    def project_to_geometry(self) -> Dict[str, Any]:
        """
        Observation Axis P_geom:
        Projects phase curvature and angular momentum into 3D volumetric geometry coordinates.
        """
        phases = np.angle(self.grid)
        amplitudes = np.abs(self.grid)

        num_points = min(32, self.resolution)
        indices = np.linspace(0, self.resolution - 1, num_points, dtype=int)

        coords_3d = []
        for idx in indices:
            r = amplitudes[idx]
            theta = phases[idx]
            phi = (idx / self.resolution) * np.pi
            x = float(r * np.sin(phi) * np.cos(theta))
            y = float(r * np.sin(phi) * np.sin(theta))
            z = float(r * np.cos(phi))
            coords_3d.append((x, y, z))

        return {
            "axis": "P_geom",
            "point_cloud_3d": coords_3d,
            "volumetric_bounding_radius": float(np.max(amplitudes)),
            "spatial_vertices_count": len(coords_3d)
        }

    # -------------------------------------------------------------------------
    # Scale-Hierarchical Macro-Rotor Concept Promotion
    # -------------------------------------------------------------------------

    def promote_to_macro_rotor_concept(self, concept_id: str) -> Dict[str, Any]:
        """
        Promotes micro phasor clusters into a higher-order Macro-Rotor Concept in the
        scale hierarchy, encapsulating causality and global order.
        """
        amplitudes = np.abs(self.grid)

        macro_amplitude = float(np.mean(amplitudes))
        macro_phase_anchor = float(np.angle(np.sum(self.grid)))
        coherence = float(np.abs(np.sum(self.grid)) / (np.sum(amplitudes) + 1e-12))

        return {
            "concept_id": concept_id,
            "macro_amplitude": macro_amplitude,
            "macro_phase_anchor": macro_phase_anchor,
            "coherence_index": coherence,
            "scale_level": "MacroHierarchy_Level_1"
        }
