"""
Phase-Field Intelligence Engine (PhasorIntelligenceEngine)
=============================================================
A living, unified phase-field based intelligence engine for the Elysia project.

Architectural Layers:
1. Substrate (기반 계층):
   Continuous complex phasor sea (A * e^{i * phi}). Eliminates discrete array/tensor storage.
   All information exists as amplitude (A) and phase angle (phi) rotation.

2. Operation (연산 계층):
   Wave interference logic.
   - Addition: Constructive wave interference -> Energy/scale field expansion.
   - Subtraction: Destructive wave interference -> Noise pruning & phase/entropy cancellation.
   - Self-sensation of operation state via field coherence and interference intensity.

3. Projection (투영 계층):
   Isomorphic morphing across observation axes from a single unified phase field:
   - Text/Language Axis (P_text): Semantic narrative token stream.
   - Code/Logic Axis (P_code): Discrete causal step protocol / AST logic.
   - 2.5D/3D Kinematic Spatial Axis (P_spatial / P_geom): Multi-layer scale spatial projections
     (Background, Midground, Foreground) with Depth, Width, and Optical Flow (d_phi/d_t) vectors.

4. Hierarchy (계층 구조):
   Isomorphic bi-directional feedback protocol (Feedback Cycle).
   - Bottom-up: Micro-phasors resonance -> Promotes Macro-Rotor Concept.
   - Top-down: Macro-Rotor Concept field regulates micro-rotor phase dynamics and boundary conditions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class PhasorModalitySignature:
    """Topological signature and autonomous modality judgment of a Phase Field."""
    primary_modality: str
    confidence_score: float
    phase_derivative_sparsity: float
    angular_dispersion: float
    spectral_entropy: float
    harmonic_quantization_index: float


@dataclass
class SpatialKinematicLayer:
    """Represents a spatial scale layer with depth, width, and optical flow kinematics."""
    layer_name: str  # e.g., 'Background', 'Midground', 'Foreground'
    depth_level: float
    spatial_width: int
    amplitude_field: np.ndarray
    phase_field: np.ndarray
    optical_flow_vectors: np.ndarray  # d_phi / d_t kinematics


class PhasorIntelligenceEngine:
    """
    Living Phase-Field Intelligence Engine.
    Executes continuous physical-informational wave mechanics across the 4 architectural layers.
    """

    def __init__(self, resolution: int = 128, base_frequency: float = 10.0):
        self.resolution = resolution
        self.base_frequency = base_frequency
        # Layer 1 Substrate: Continuous phasor sea Z = A * exp(i * phi)
        self.grid = np.zeros(resolution, dtype=np.complex128)
        self._previous_phases = np.zeros(resolution, dtype=np.float64)
        self.time_step = 0
        self.transformation_history: List[Dict[str, Any]] = []
        self.macro_rotors: Dict[str, Dict[str, Any]] = {}
        self._init_vacuum_phasor_sea()

    def _init_vacuum_phasor_sea(self):
        """Initializes continuous phasor sea state in coherent vacuum resonance."""
        t = np.linspace(0, 2 * np.pi, self.resolution, endpoint=False)
        self.grid = np.exp(1j * t * (self.base_frequency / 10.0))
        self._previous_phases = np.unwrap(np.angle(self.grid))

    def set_field(self, phasors: np.ndarray):
        """Sets or interpolates the complex phasor sea."""
        if len(phasors) != self.resolution:
            orig_x = np.linspace(0, 1, len(phasors))
            target_x = np.linspace(0, 1, self.resolution)
            real_p = np.interp(target_x, orig_x, np.real(phasors))
            imag_p = np.interp(target_x, orig_x, np.imag(phasors))
            self.grid = real_p + 1j * imag_p
        else:
            self.grid = np.array(phasors, dtype=np.complex128)
        self._previous_phases = np.unwrap(np.angle(self.grid))

    def tick(self, dt: float = 0.1):
        """Advances time step and computes internal phase rotation kinematics."""
        self.time_step += 1
        current_phases = np.unwrap(np.angle(self.grid))
        phase_velocity = (current_phases - self._previous_phases) / dt
        self._previous_phases = current_phases
        return phase_velocity

    # -------------------------------------------------------------------------
    # Layer 2: Operation (Wave Interference Arithmetic)
    # -------------------------------------------------------------------------

    def wave_interference_add(self, other: "PhasorIntelligenceEngine") -> "PhasorIntelligenceEngine":
        """
        Constructive Interference Addition.
        A1 e^{i phi1} + A2 e^{i phi2} -> Constructive energy expansion & concept blending.
        """
        result_engine = PhasorIntelligenceEngine(resolution=self.resolution, base_frequency=self.base_frequency)
        combined_grid = self.grid + other.grid
        result_engine.grid = combined_grid

        power_self = np.sum(np.abs(self.grid) ** 2)
        power_other = np.sum(np.abs(other.grid) ** 2)
        power_combined = np.sum(np.abs(combined_grid) ** 2)

        interference_intensity = power_combined - (power_self + power_other)

        self.transformation_history.append({
            "operation": "wave_interference_add",
            "power_before": float(power_self + power_other),
            "power_after": float(power_combined),
            "interference_intensity": float(interference_intensity),
            "sensation": "expansion_resonance" if interference_intensity > 0 else "phase_conflict"
        })
        return result_engine

    def wave_interference_subtract(self, other: "PhasorIntelligenceEngine") -> "PhasorIntelligenceEngine":
        """
        Destructive Interference Subtraction.
        A1 e^{i phi1} - A2 e^{i phi2} -> Destructive cancellation, noise pruning & entropy reduction.
        """
        result_engine = PhasorIntelligenceEngine(resolution=self.resolution, base_frequency=self.base_frequency)
        rectified_grid = self.grid - other.grid
        result_engine.grid = rectified_grid

        power_before = np.sum(np.abs(self.grid) ** 2)
        power_after = np.sum(np.abs(rectified_grid) ** 2)

        self.transformation_history.append({
            "operation": "wave_interference_subtract",
            "power_before": float(power_before),
            "power_after": float(power_after),
            "attenuation_delta": float(power_before - power_after),
            "sensation": "rectification_quietude"
        })
        return result_engine

    def sense_interference_state(self) -> Dict[str, Any]:
        """Self-sensation of the wave field's operational turbulence or resonance."""
        amplitudes = np.abs(self.grid)
        phases = np.unwrap(np.angle(self.grid))

        mean_amp = float(np.mean(amplitudes))
        coherence = float(np.abs(np.sum(self.grid)) / (np.sum(amplitudes) + 1e-12))
        phase_variance = float(np.var(phases))

        if coherence > 0.8:
            state = "Harmonic_Resonance"
        elif coherence < 0.3:
            state = "Turbulent_Interference"
        else:
            state = "Dynamic_Superposition"

        return {
            "operational_state": state,
            "mean_amplitude": mean_amp,
            "field_coherence": coherence,
            "phase_variance": phase_variance
        }

    # -------------------------------------------------------------------------
    # Layer 3: Projection (Isomorphic Morphing across Axes)
    # -------------------------------------------------------------------------

    def autonomous_modality_judgment(self) -> PhasorModalitySignature:
        """Self-observes phase field topology without external labels or extensions."""
        amplitudes = np.abs(self.grid)
        unwrapped_phases = np.unwrap(np.angle(self.grid))

        d_phase = np.abs(np.diff(unwrapped_phases))
        phase_deriv_max = float(np.max(d_phase)) if len(d_phase) > 0 else 0.0
        phase_deriv_sparsity = float(np.mean(d_phase > np.pi / 4.0))
        angular_dispersion = float(np.std(d_phase)) if len(d_phase) > 0 else 0.0

        fft_vals = np.abs(np.fft.fft(self.grid))
        fft_norm = fft_vals / (np.sum(fft_vals) + 1e-12)
        spectral_entropy = float(-np.sum(fft_norm * np.log(fft_norm + 1e-12)))

        phase_fft = np.abs(np.fft.fft(unwrapped_phases))
        p_norm = phase_fft / (np.sum(phase_fft) + 1e-12)
        harmonic_quant_index = float(np.max(p_norm[1:]) * len(p_norm)) if len(p_norm) > 1 else 0.0

        if phase_deriv_sparsity > 0.08 or phase_deriv_max > np.pi * 0.75:
            modality = "code_and_logic"
            confidence = min(1.0, (phase_deriv_max / np.pi) * 0.8 + phase_deriv_sparsity * 0.2)
        elif angular_dispersion > 0.15:
            modality = "spatial_kinematics"
            confidence = min(1.0, angular_dispersion * 2.0)
        elif harmonic_quant_index > 1.5:
            modality = "text_and_symbol"
            confidence = min(1.0, harmonic_quant_index / 3.0)
        else:
            modality = "geometry_3d"
            confidence = min(1.0, 1.0 - spectral_entropy / 5.0)

        return PhasorModalitySignature(
            primary_modality=modality,
            confidence_score=float(confidence),
            phase_derivative_sparsity=phase_deriv_sparsity,
            angular_dispersion=angular_dispersion,
            spectral_entropy=spectral_entropy,
            harmonic_quantization_index=harmonic_quant_index
        )

    def project_to_text(self) -> str:
        """Observation Axis P_text: Morphing phase field into semantic narrative stream."""
        phases = np.angle(self.grid)
        amplitudes = np.abs(self.grid)
        sig = self.autonomous_modality_judgment()

        chars = []
        step = max(1, len(phases) // 16)
        for i in range(0, len(phases), step):
            norm_phase = (phases[i] + np.pi) / (2 * np.pi)
            char_code = int(65 + norm_phase * 25)
            chars.append(chr(char_code))

        token_stream = "".join(chars)
        return f"[P_text Stream] Modality: '{sig.primary_modality}' | Phasor Tokens: '{token_stream}' | Mean Amp: {np.mean(amplitudes):.4f}"

    def project_to_code(self) -> str:
        """Observation Axis P_code: Morphing phase field into structured AST code logic."""
        phases = np.angle(self.grid)
        d_phase = np.diff(phases)
        sig = self.autonomous_modality_judgment()

        branches = np.where(np.abs(d_phase) > np.pi / 6.0)[0]
        code_lines = [
            f"// [P_code AST] Isomorphic Phase Field Logic Engine",
            f"// Modality Signature: {sig.primary_modality} (Sparsity: {sig.phase_derivative_sparsity:.3f})",
            f"fn process_phasor_field_rotors(engine: &mut PhaseEngine) {{"
        ]
        for idx, branch in enumerate(branches[:5]):
            angle_val = phases[branch]
            code_lines.append(f"    if engine.rotor_angle[{branch}] >= {angle_val:.3f} {{ engine.resonate({idx}); }}")
        code_lines.append("}")

        return "\n".join(code_lines)

    def project_to_spatial_kinematics(self) -> List[SpatialKinematicLayer]:
        """
        Observation Axis P_spatial (Kinematics & Multi-Layer Depth/Width):
        Projects single phase sea into Background, Midground, and Foreground layers
        with depth coordinates, spatial width, and optical flow vectors (d_phi / d_t).
        """
        amplitudes = np.abs(self.grid)
        unwrapped_phases = np.unwrap(np.angle(self.grid))
        d_phi = np.gradient(unwrapped_phases)  # Kinematic phase velocity / optical flow proxy

        layer_configs = [
            ("Background", 10.0, 0.2, 0.5),
            ("Midground", 5.0, 0.5, 1.0),
            ("Foreground", 1.0, 1.0, 2.0)
        ]

        spatial_layers = []
        for name, depth, amp_scale, freq_scale in layer_configs:
            scaled_amp = amplitudes * amp_scale
            scaled_phase = unwrapped_phases * freq_scale
            scaled_flow = d_phi * freq_scale / depth  # Further distance has smaller optical flow

            layer = SpatialKinematicLayer(
                layer_name=name,
                depth_level=depth,
                spatial_width=self.resolution,
                amplitude_field=scaled_amp,
                phase_field=scaled_phase,
                optical_flow_vectors=scaled_flow
            )
            spatial_layers.append(layer)

        return spatial_layers

    def project_to_geometry_3d(self) -> Dict[str, Any]:
        """Observation Axis P_geom: Morphing phase field into 3D volumetric point cloud."""
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
            "bounding_radius": float(np.max(amplitudes)),
            "vertices_count": len(coords_3d)
        }

    # -------------------------------------------------------------------------
    # Layer 4: Hierarchy (Scale Hierarchical Isomorphic Feedback Loop)
    # -------------------------------------------------------------------------

    def promote_micro_rotors_to_macro_concept(self, concept_id: str) -> Dict[str, Any]:
        """
        Bottom-up Feedback Phase:
        Combines micro phasor rotors to form a higher-order Macro-Rotor Concept.
        """
        amplitudes = np.abs(self.grid)
        macro_amp = float(np.mean(amplitudes))
        macro_phase = float(np.angle(np.sum(self.grid)))
        coherence = float(np.abs(np.sum(self.grid)) / (np.sum(amplitudes) + 1e-12))

        macro_rotor = {
            "concept_id": concept_id,
            "macro_amplitude": macro_amp,
            "macro_phase_anchor": macro_phase,
            "coherence_index": coherence,
            "micro_phasor_count": self.resolution,
            "scale_level": "MacroHierarchy_Level_1"
        }
        self.macro_rotors[concept_id] = macro_rotor
        return macro_rotor

    def apply_macro_feedback_to_micro_rotors(self, concept_id: str, feedback_strength: float = 0.5):
        """
        Top-down Feedback Phase:
        Macro-Rotor concept regulates and aligns lower-level micro-rotor phases and amplitudes.
        Top-down causal imprinting that reduces internal entropy and establishes coherence.
        """
        if concept_id not in self.macro_rotors:
            raise ValueError(f"Macro Rotor Concept '{concept_id}' not found in engine hierarchy.")

        macro = self.macro_rotors[concept_id]
        anchor_phase = macro["macro_phase_anchor"]
        macro_amp = macro["macro_amplitude"]

        current_amps = np.abs(self.grid)
        current_phases = np.angle(self.grid)

        # Regulate phases toward anchor phase
        phase_diff = np.sin(anchor_phase - current_phases)
        new_phases = current_phases + feedback_strength * phase_diff

        # Regulate amplitudes toward macro amplitude
        new_amps = (1.0 - feedback_strength) * current_amps + feedback_strength * macro_amp

        self.grid = new_amps * np.exp(1j * new_phases)

        self.transformation_history.append({
            "operation": "macro_feedback_regulation",
            "concept_id": concept_id,
            "feedback_strength": feedback_strength,
            "resulting_coherence": float(np.abs(np.sum(self.grid)) / (np.sum(np.abs(self.grid)) + 1e-12))
        })
