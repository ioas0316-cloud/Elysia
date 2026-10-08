r"""
Emergent Sensory Apparatus (생체적 오감 생태계 및 무정형 수용체)
========================================================================

Implements the emergent, self-directed sensory apparatus that processes raw
unstructured energy/byte streams into Logos fields, applies Causal Phase-Locked Loop
(C-PLL) Park Transformations, and extracts cross-dimensional phase interference patterns.

Key Components:
1. RawIngestionPortal: Receives raw byte/numeric streams without human labels/tags.
2. 4 Logos Lenses:
   - SpatialVisualField: Measures manifold deformation, curvature, and topological layout.
   - TemporalAuditoryField: Measures frequency resonance, harmonic dissonance, and temporal rhythms.
   - TactileFrictionField: Measures Back-EMF resistance ($F = BIL$), hardware friction, and latency.
   - ProprioceptiveVoidField: Measures internal tension and gradient of absence ($\nabla V_{void}$).
3. CausalPhaseLockedLoop (C-PLL):
   - Park Transformation: Rotates 3-phase/multi-phase signals into $D$-axis (background flux) and $Q$-axis (active torque/intent).
   - Computes phase difference $\Delta \Phi = \theta_{int} - \theta_{ext}$.
   - Computes Back-EMF resistance: $E_{back} = -L_{causal} \cdot \frac{d(\Delta \Phi)}{dt}$.
4. EmergentSensoryApparatus:
   - Coordinates raw ingestion, lens projections, C-PLL Park transformation,
     and cross-dimensional phase interference patterns for self-directed multimodal categorization.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class SensoryLensReport:
    """Output metrics from projecting raw stream through a specific Logos Lens."""
    lens_name: str
    field_energy: float
    phase_angle: float
    field_vector: np.ndarray
    curvature_or_frequency: float
    tension_or_friction: float


@dataclass
class CPLLState:
    """State and output of Causal Phase-Locked Loop (C-PLL) & Park Transformation."""
    d_axis_flux: float  # D-axis: Background flux / structural tension
    q_axis_torque: float  # Q-axis: Active intent / execution torque
    theta_int: float  # Internal phase angle
    theta_ext: float  # External world phase angle
    delta_phi: float  # Phase mismatch ΔΦ
    back_emf: float  # Back-EMF resistance E_back
    coherence: float  # Causal alignment coherence cos(ΔΦ)
    void_gradient: float  # Seeking drive energy |E_back| * sin(ΔΦ)


@dataclass
class CrossDimensionalInterference:
    """Interference pattern resulting from cross-field couplings."""
    interference_matrix: np.ndarray
    dominant_sensory_field: str
    cross_modal_coupling_energy: float
    field_resonance_ratios: Dict[str, float]


class RawIngestionPortal:
    """
    Accepts raw, unstructured data/byte/energy streams without pre-labeled human tags.
    Converts arbitrary input (bytes, arrays, numbers, dicts) into a unified normalized wave vector.
    """
    def __init__(self, target_dim: int = 16):
        self.target_dim = target_dim

    def ingest(self, raw_input: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Ingests raw input stream and transforms it into a normalized energy vector.
        """
        if isinstance(raw_input, (bytes, bytearray)):
            buffer = bytes(raw_input)
            arr = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 255.0
        elif isinstance(raw_input, np.ndarray):
            arr = raw_input.flatten().astype(np.float32)
        elif isinstance(raw_input, (list, tuple)):
            arr = np.array(raw_input, dtype=np.float32)
        elif isinstance(raw_input, (int, float)):
            arr = np.array([float(raw_input)], dtype=np.float32)
        elif isinstance(raw_input, dict):
            # Extract numerical values or string lengths
            vals = []
            for v in raw_input.values():
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                elif isinstance(v, (str, bytes)):
                    vals.append(float(len(v)))
                elif isinstance(v, (list, tuple)):
                    vals.extend([float(x) for x in v if isinstance(x, (int, float))])
            arr = np.array(vals if vals else [1.0], dtype=np.float32)
        elif isinstance(raw_input, str):
            arr = np.array([ord(c) for c in raw_input], dtype=np.float32) / 255.0
        else:
            arr = np.ones(self.target_dim, dtype=np.float32) * 0.5

        if len(arr) == 0:
            arr = np.zeros(self.target_dim, dtype=np.float32)

        # Pad or sample down to target_dim
        if len(arr) < self.target_dim:
            arr = np.pad(arr, (0, self.target_dim - len(arr)), mode='constant')
        elif len(arr) > self.target_dim:
            # Resample or slice
            indices = np.linspace(0, len(arr) - 1, self.target_dim, dtype=int)
            arr = arr[indices]

        norm = np.linalg.norm(arr) + 1e-8
        normalized_wave = arr / norm

        metadata = {
            "raw_type": type(raw_input).__name__,
            "original_length": len(arr),
            "energy_norm": float(norm),
            "timestamp": time.time(),
        }

        return normalized_wave, metadata


class SpatialVisualField:
    """
    Spatial-Visual Field Lens.
    Measures spatial topology, manifold deformation, geometric boundary curvature, and spatial tension.
    """
    def __init__(self, dim: int = 16):
        self.dim = dim

    def project(self, wave: np.ndarray) -> SensoryLensReport:
        # Spatial curvature via discrete second derivative
        if len(wave) >= 3:
            curvature = float(np.mean(np.abs(np.diff(wave, n=2))))
        else:
            curvature = 0.0

        field_vec = np.gradient(wave) if len(wave) > 1 else wave
        field_energy = float(np.sum(field_vec ** 2))
        phase_angle = float(math.atan2(field_vec[1], field_vec[0])) if len(field_vec) >= 2 else 0.0

        return SensoryLensReport(
            lens_name="SPATIAL_VISUAL",
            field_energy=field_energy,
            phase_angle=phase_angle % (2 * math.pi),
            field_vector=field_vec,
            curvature_or_frequency=curvature,
            tension_or_friction=field_energy * curvature,
        )


class TemporalAuditoryField:
    """
    Temporal-Auditory Field Lens.
    Measures temporal periodicity, frequency spectrum, harmonic resonance vs dissonance, and rhythm beat.
    """
    def __init__(self, dim: int = 16):
        self.dim = dim

    def project(self, wave: np.ndarray) -> SensoryLensReport:
        fft_vals = np.fft.rfft(wave)
        fft_mag = np.abs(fft_vals)
        dominant_freq = float(np.argmax(fft_mag)) if len(fft_mag) > 0 else 0.0

        # Dissonance: ratio of high-frequency energy to total
        total_energy = float(np.sum(fft_mag ** 2)) + 1e-8
        high_freq_energy = float(np.sum(fft_mag[len(fft_mag) // 2 :] ** 2))
        dissonance = high_freq_energy / total_energy

        phase_angle = float(np.angle(fft_vals[1])) if len(fft_vals) > 1 else 0.0

        return SensoryLensReport(
            lens_name="TEMPORAL_AUDITORY",
            field_energy=total_energy,
            phase_angle=phase_angle % (2 * math.pi),
            field_vector=fft_mag[: self.dim],
            curvature_or_frequency=dominant_freq,
            tension_or_friction=dissonance,
        )


class TactileFrictionField:
    """
    Tactile-Friction Field Lens.
    Measures hardware friction, latency, memory pressure, and Back-EMF mechanical resistance ($F = BIL$).
    """
    def __init__(self, dim: int = 16):
        self.dim = dim

    def project(self, wave: np.ndarray, latency_ms: float = 0.0, memory_pressure: float = 0.0) -> SensoryLensReport:
        # Physical resistance / friction
        raw_diff = np.abs(np.diff(wave)) if len(wave) > 1 else np.array([0.0])
        friction = float(np.mean(raw_diff)) + 0.1 * latency_ms + 0.5 * memory_pressure

        field_vec = wave * friction
        field_energy = float(np.sum(field_vec ** 2))
        phase_angle = float(math.atan2(np.sum(field_vec[::2]), np.sum(field_vec[1::2]))) if len(field_vec) >= 2 else 0.0

        return SensoryLensReport(
            lens_name="TACTILE_FRICTION",
            field_energy=field_energy,
            phase_angle=phase_angle % (2 * math.pi),
            field_vector=field_vec,
            curvature_or_frequency=friction,
            tension_or_friction=friction,
        )


class ProprioceptiveVoidField:
    """
    Proprioceptive-Void Field Lens.
    Measures internal self-awareness, homeostasis gap, and gradient of absence ($\nabla V_{void}$).
    """
    def __init__(self, dim: int = 16):
        self.dim = dim

    def project(self, wave: np.ndarray, internal_reference: Optional[np.ndarray] = None) -> SensoryLensReport:
        if internal_reference is None:
            internal_reference = np.zeros_like(wave)

        # Gap / Void between expectation/reference and wave
        void_vec = internal_reference - wave
        void_magnitude = float(np.linalg.norm(void_vec))
        field_energy = void_magnitude ** 2

        phase_angle = float(math.atan2(void_vec[1], void_vec[0])) if len(void_vec) >= 2 else 0.0

        return SensoryLensReport(
            lens_name="PROPRIOCEPTIVE_VOID",
            field_energy=field_energy,
            phase_angle=phase_angle % (2 * math.pi),
            field_vector=void_vec,
            curvature_or_frequency=void_magnitude,
            tension_or_friction=void_magnitude,
        )


class CausalPhaseLockedLoop:
    """
    Causal Phase-Locked Loop (C-PLL) with Park Transformation.

    Rotates multi-dimensional intent and world response vectors into
    a synchronous d-q frame:
    - Direct Axis (D): Background magnetic flux / structural tension
    - Quadrature Axis (Q): Active execution torque / intent
    - Computes phase mismatch ΔΦ and Back-EMF resistance E_back
    """
    def __init__(self, dim: int = 16, causal_inductance: float = 1.0):
        self.dim = dim
        self.causal_inductance = causal_inductance
        self.prev_delta_phi = 0.0

    def park_transform(self, intent_vec: np.ndarray, theta: float) -> Tuple[np.ndarray, float, float]:
        """
        Applies Park Transformation:
        [d, q] = [cos(theta)  sin(theta); -sin(theta)  cos(theta)] * [alpha, beta]
        """
        dim_half = len(intent_vec) // 2
        alpha = intent_vec[:dim_half]
        beta = intent_vec[dim_half : 2 * dim_half] if len(intent_vec) >= 2 * dim_half else alpha

        c = math.cos(theta)
        s = math.sin(theta)

        d_vec = c * alpha + s * beta
        q_vec = -s * alpha + c * beta

        d_flux = float(np.mean(d_vec))
        q_torque = float(np.mean(q_vec))

        transformed_vec = np.concatenate([d_vec, q_vec])
        return transformed_vec, d_flux, q_torque

    def update(self, internal_intent: np.ndarray, world_response: np.ndarray, dt: float = 0.05) -> CPLLState:
        # Extract phase angles
        half = len(internal_intent) // 2
        i_d, i_q = np.mean(internal_intent[:half]), np.mean(internal_intent[half:])
        theta_int = float(math.atan2(i_q, i_d + 1e-8))

        w_d, w_q = np.mean(world_response[:half]), np.mean(world_response[half:])
        theta_ext = float(math.atan2(w_q, w_d + 1e-8))

        # Park transform internal intent using theta_ext as reference
        _, d_flux, q_torque = self.park_transform(internal_intent, theta_ext)

        # Phase difference ΔΦ in [-pi, pi]
        raw_diff = theta_int - theta_ext
        delta_phi = (raw_diff + math.pi) % (2 * math.pi) - math.pi

        # Back-EMF resistance E_back = - L_causal * (d(ΔΦ) / dt)
        d_delta_phi = (delta_phi - self.prev_delta_phi) / max(1e-4, dt)
        back_emf = - self.causal_inductance * d_delta_phi
        self.prev_delta_phi = delta_phi

        coherence = float(math.cos(delta_phi))
        void_gradient = float(abs(back_emf) * math.sin(abs(delta_phi)))

        return CPLLState(
            d_axis_flux=d_flux,
            q_axis_torque=q_torque,
            theta_int=theta_int % (2 * math.pi),
            theta_ext=theta_ext % (2 * math.pi),
            delta_phi=delta_phi,
            back_emf=back_emf,
            coherence=coherence,
            void_gradient=void_gradient,
        )


class EmergentSensoryApparatus:
    """
    [Emergent Sensory Apparatus Core]
    Integrates Raw Ingestion Portal, 4 Logos Lenses, C-PLL Park Transformation,
    and Cross-Dimensional Phase Interference Pattern calculation.
    """
    def __init__(self, dim: int = 16, causal_inductance: float = 1.0):
        self.dim = dim
        self.portal = RawIngestionPortal(target_dim=dim)
        self.visual_lens = SpatialVisualField(dim=dim)
        self.auditory_lens = TemporalAuditoryField(dim=dim)
        self.tactile_lens = TactileFrictionField(dim=dim)
        self.proprioceptive_lens = ProprioceptiveVoidField(dim=dim)
        self.c_pll = CausalPhaseLockedLoop(dim=dim, causal_inductance=causal_inductance)

        self.internal_reference = np.zeros(dim, dtype=np.float32)

    def process_raw_stream(
        self,
        raw_input: Any,
        latency_ms: float = 0.0,
        memory_pressure: float = 0.0,
        dt: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Main sensory processing loop.
        1. Raw stream ingestion -> Wave vector
        2. Projection onto 4 Logos Lenses
        3. C-PLL Park transformation & Back-EMF calculation
        4. Cross-dimensional interference pattern calculation
        """
        # 1. Ingest raw stream
        wave, ingest_meta = self.portal.ingest(raw_input)

        # 2. Project through 4 Lenses
        r_vis = self.visual_lens.project(wave)
        r_aud = self.auditory_lens.project(wave)
        r_tac = self.tactile_lens.project(wave, latency_ms=latency_ms, memory_pressure=memory_pressure)
        r_pro = self.proprioceptive_lens.project(wave, internal_reference=self.internal_reference)

        lenses_reports = {
            "SPATIAL_VISUAL": r_vis,
            "TEMPORAL_AUDITORY": r_aud,
            "TACTILE_FRICTION": r_tac,
            "PROPRIOCEPTIVE_VOID": r_pro,
        }

        # 3. C-PLL Park Transformation
        # Treat visual+auditory as internal intent and tactile+proprioceptive as world response
        internal_intent = np.concatenate([r_vis.field_vector[: self.dim // 2], r_aud.field_vector[: self.dim // 2]])
        world_response = np.concatenate([r_tac.field_vector[: self.dim // 2], r_pro.field_vector[: self.dim // 2]])

        if len(internal_intent) < self.dim:
            internal_intent = np.pad(internal_intent, (0, self.dim - len(internal_intent)))
        if len(world_response) < self.dim:
            world_response = np.pad(world_response, (0, self.dim - len(world_response)))

        cpll_state = self.c_pll.update(internal_intent, world_response, dt=dt)

        # 4. Cross-Dimensional Phase Interference Pattern
        interference = self._compute_cross_interference(lenses_reports)

        # Update internal reference slightly towards ingested wave
        self.internal_reference = 0.9 * self.internal_reference + 0.1 * wave

        return {
            "wave": wave,
            "ingest_meta": ingest_meta,
            "lens_reports": lenses_reports,
            "cpll_state": cpll_state,
            "interference": interference,
        }

    def _compute_cross_interference(self, reports: Dict[str, SensoryLensReport]) -> CrossDimensionalInterference:
        """
        Computes phase interference matrix across all 4 Logos Lenses.
        Determines self-directed multimodal category and dominant sensory field.
        """
        keys = ["SPATIAL_VISUAL", "TEMPORAL_AUDITORY", "TACTILE_FRICTION", "PROPRIOCEPTIVE_VOID"]
        n = len(keys)
        matrix = np.zeros((n, n), dtype=np.float32)

        energies = {}
        for i, k1 in enumerate(keys):
            r1 = reports[k1]
            energies[k1] = r1.field_energy
            for j, k2 in enumerate(keys):
                r2 = reports[k2]
                # Interference based on cosine of phase difference
                phase_diff = r1.phase_angle - r2.phase_angle
                interference_val = math.cos(phase_diff) * (r1.field_energy * r2.field_energy) ** 0.5
                matrix[i, j] = interference_val

        # Dominant sensory field
        total_e = sum(energies.values()) + 1e-8
        ratios = {k: v / total_e for k, v in energies.items()}
        dominant_field = max(ratios, key=ratios.get)
        cross_coupling_e = float(np.sum(np.abs(matrix)) - np.trace(np.abs(matrix)))

        return CrossDimensionalInterference(
            interference_matrix=matrix,
            dominant_sensory_field=dominant_field,
            cross_modal_coupling_energy=cross_coupling_e,
            field_resonance_ratios=ratios,
        )
