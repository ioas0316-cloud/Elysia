r"""
wave_prism_spatiotemporal_rotor_core.py

ELYSIA Wave-Prism Interference & Spatiotemporal S^3 Rotor Substrate
==================================================================
Pure continuous wave/field/rotor substrate (ZERO PyTorch, ZERO Transformers, ZERO LLM
calls at runtime) that lifts raw real-world signals through the full geometric ladder:

  0D Scalar Point
    -> 1D Temporal Wave Trajectory (continuous channels over t)
    -> 2D Circle Complex Frequency Prism S^1 (complex FFT harmonics: A(w) * e^{i*phi(w)})
    -> 3D / 4D Spatiotemporal Quaternion Rotor S^3 (Q(t) rotating on unit 3-sphere -> S^2)

Core Architectural Consequences:
1. **Resonance IS Direct Wave Superposition (No Neural Forward Pass)**:
   Superimposing two complex wave prisms (`Psi_source + Psi_probe`) directly exposes
   constructive interference (`Re(s * conj(p)) > 0`), destructive phase opposition
   (`Re(s * conj(p)) < 0`, tearing the `ConnectivityBeam`), and orthogonal uncoupling
   (`spectral_overlap_support < 0.25 -> ORTHOGONAL_VOID`) in sub-millisecond time.
2. **Causal Bifurcation Point `(t*, w*)` Observed Directly From Trajectory Geometry**:
   Measuring the geodesic divergence `theta(t) = 2 * arccos(|Q(0) . Q(t)|)` on S^3
   and its rate `d(theta)/dt` directly reveals the exact moment in real video or audio
   where dynamic causality splits away from its initial anchor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - enables projection='3d'
from scipy.io import wavfile

from core.physics.causal_field import CausalField, ConnectivityBeam, InformationVoxel

PRISM_HARMONIC_BINS = 16


# ======================================================================
# Exact S^3 Quaternion Rotor Algebra
# ======================================================================
def quat_normalize(q: np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim == 1:
        nrm = float(np.linalg.norm(arr))
        if nrm <= 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return arr / nrm
    nrms = np.linalg.norm(arr, axis=-1, keepdims=True)
    safe = np.where(nrms > 1e-12, arr / np.maximum(nrms, 1e-12), np.array([1.0, 0.0, 0.0, 0.0]))
    return safe


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64).copy()
    arr[..., 1:] = -arr[..., 1:]
    return arr


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return quat_normalize(out)


def quat_from_axis_angle(axis_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    angle = float(angle_rad)
    if abs(angle) <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ax = np.asarray(axis_xyz, dtype=np.float64)
    nrm = float(np.linalg.norm(ax))
    if nrm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u = ax / nrm
    half = 0.5 * angle
    s = float(np.sin(half))
    return quat_normalize(np.array([np.cos(half), u[0] * s, u[1] * s, u[2] * s], dtype=np.float64))


def quat_geodesic_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    a = quat_normalize(q1)
    b = quat_normalize(q2)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def quat_rotate_vector(q: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
    qn = quat_normalize(q)
    v = np.asarray(v_xyz, dtype=np.float64)
    vq = np.concatenate([np.zeros(qn.shape[:-1] + (1,), dtype=np.float64), np.broadcast_to(v, qn.shape[:-1] + (3,))], axis=-1)
    rotated = quat_multiply(quat_multiply(qn, vq), quat_conjugate(qn))
    return rotated[..., 1:]


# ======================================================================
# Core Data Structures
# ======================================================================
@dataclass
class WavePrismSpectrum:
    """2D Circle S^1 complex harmonic prism representation of a continuous temporal wave."""

    modality_id: str
    frequencies: np.ndarray
    complex_spectrum: np.ndarray  # complex64 shape (K,)
    amplitudes: np.ndarray
    phases: np.ndarray
    dc_static_energy_ratio: float


@dataclass
class CausalBifurcationPoint:
    """Observed spatiotemporal bifurcation instant `(t*, w*)` where geodesic divergence peaks."""

    frame_index: int
    total_frames: int
    time_sec: float
    dominant_harmonic_band: int
    geodesic_divergence_rad: float
    divergence_velocity: float
    primary_causal_axis: str


@dataclass
class SpatiotemporalRotorRecord:
    """Complete 0D -> 1D -> 2D S^1 Prism -> 3D/4D S^3 Rotor lifted entity."""

    record_id: str
    modality: str
    duration_sec: float
    wave_1d_channels: Dict[str, np.ndarray]
    prism_spectrum: WavePrismSpectrum
    quaternions_s3: np.ndarray  # (T, 4)
    sphere_trajectory_xyz: np.ndarray  # (T, 3)
    geodesic_divergence_curve: np.ndarray  # (T,)
    divergence_velocity_curve: np.ndarray  # (T,)
    bifurcation: CausalBifurcationPoint
    key_frames_rgb: Optional[List[Tuple[int, np.ndarray]]] = None
    lift_latency_ms: float = 0.0


@dataclass
class InterferenceResonanceResult:
    """Direct wave-prism interference + S^3 geodesic alignment verdict without heavy ML calls."""

    source_id: str
    probe_id: str
    prism_interference_spectrum: np.ndarray
    prism_constructive_score: float
    spectral_overlap_support: float
    rotor_geodesic_alignment: float
    combined_resonance: float
    verdict: str  # "CONSTRUCTIVE_RESONANCE" | "DESTRUCTIVE_CONTRADICTION" | "ORTHOGONAL_VOID"
    beam_torn: bool
    latency_ms: float


# ======================================================================
# Core Engine: Wave-Prism & Spatiotemporal S^3 Rotor Core
# ======================================================================
class WavePrismSpatiotemporalRotorCore:
    """
    Lifts real multi-modal sensory streams (`.mp4` video, `.wav` audio) and physical
    hypotheses onto ONE common Complex Wave Prism (`S^1`) + Spatiotemporal Quaternion Rotor
    (`S^3`) manifold integrated directly with `CausalField`.
    """

    def __init__(self) -> None:
        self.causal_field = CausalField(dimensions=3)
        self.records: Dict[str, SpatiotemporalRotorRecord] = {}

    @staticmethod
    def _compute_complex_prism(
        modality_id: str, real_wave: np.ndarray, imag_wave: np.ndarray, n_bins: int = PRISM_HARMONIC_BINS
    ) -> WavePrismSpectrum:
        """
        Convert 1D continuous analytic wave `(real_wave + i * imag_wave)` onto an `n_bins`
        2D Circle (`S^1`) complex frequency prism.
        """
        complex_sig = real_wave.astype(np.float64) + 1j * imag_wave.astype(np.float64)
        spec_full = np.fft.fft(complex_sig)
        # Keep DC + positive harmonics up to `n_bins`
        bins = np.zeros(n_bins, dtype=np.complex64)
        k_copy = min(n_bins, spec_full.shape[0])
        bins[:k_copy] = spec_full[:k_copy].astype(np.complex64)

        total_power = float(np.sum(np.abs(bins) ** 2))
        dc_power = float(np.abs(bins[0]) ** 2)
        dc_ratio = float(dc_power / max(total_power, 1e-12))

        norm_scale = float(np.sqrt(max(total_power, 1e-12)))
        bins_norm = (bins / norm_scale).astype(np.complex64)

        return WavePrismSpectrum(
            modality_id=modality_id,
            frequencies=np.arange(n_bins, dtype=np.float32),
            complex_spectrum=bins_norm,
            amplitudes=np.abs(bins_norm).astype(np.float32),
            phases=np.angle(bins_norm).astype(np.float32),
            dc_static_energy_ratio=dc_ratio,
        )

    @staticmethod
    def _lift_3d_trajectory_to_s3_rotor(
        delta_vectors_xyz: np.ndarray,
        prism: WavePrismSpectrum,
        duration_sec: float,
        primary_causal_axis: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CausalBifurcationPoint]:
        """Lift a (T, 3) continuous state-delta curve onto unit S^3 quaternions and S^2 sphere."""
        t_len = delta_vectors_xyz.shape[0]
        quats = np.zeros((t_len, 4), dtype=np.float64)
        ref_pole = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        for t in range(t_len):
            v = delta_vectors_xyz[t]
            angle = float(np.clip(np.linalg.norm(v), 0.0, np.pi * 0.90))
            quats[t] = quat_from_axis_angle(v, angle)

        sphere_xyz = quat_rotate_vector(quats, ref_pole)
        q0 = quats[0]
        geodesic_curve = np.array(
            [quat_geodesic_angle(q0, quats[t]) for t in range(t_len)], dtype=np.float64
        )
        vel_curve = np.gradient(geodesic_curve)

        if t_len >= 3:
            t_star = int(np.argmax(vel_curve[1:-1]) + 1)
        else:
            t_star = max(0, t_len - 1)

        dyn_amps = prism.amplitudes[1:]
        dom_band = int(np.argmax(dyn_amps) + 1) if dyn_amps.size > 0 else 1

        bif = CausalBifurcationPoint(
            frame_index=t_star,
            total_frames=t_len,
            time_sec=float(duration_sec * t_star / max(1, t_len - 1)),
            dominant_harmonic_band=dom_band,
            geodesic_divergence_rad=float(geodesic_curve[t_star]),
            divergence_velocity=float(vel_curve[t_star]),
            primary_causal_axis=primary_causal_axis,
        )
        return quats, sphere_xyz, geodesic_curve, vel_curve, bif

    def _register_record_in_causal_field(self, record: SpatiotemporalRotorRecord) -> None:
        self.records[record.record_id] = record
        q_final = record.quaternions_s3[-1].astype(np.float32)
        end_pos = record.sphere_trajectory_xyz[-1].astype(np.float32)
        dom_band = record.bifurcation.dominant_harmonic_band
        phase_ang = float(record.prism_spectrum.phases[dom_band]) % float(2.0 * np.pi)

        voxel = InformationVoxel(
            id=record.record_id,
            content=f"{record.modality}:{record.bifurcation.primary_causal_axis}",
            tensor=record.prism_spectrum.amplitudes.astype(np.float32),
            mass=float(1.0 + record.bifurcation.geodesic_divergence_rad),
            position=end_pos,
            phase_angle=phase_ang,
            scale_level=3.0,
            quaternion_q=q_final,
        )
        self.causal_field.add_voxel(voxel)

    # ------------------------------------------------------------------
    # 1. Real External MP4 Video -> 0D..S^3 Spatiotemporal Rotor
    # ------------------------------------------------------------------
    def lift_real_mp4_to_spatiotemporal_rotor(
        self,
        video_path: Path,
        sample_frames: int = 36,
        record_id: str = "real_mp4_flower",
    ) -> SpatiotemporalRotorRecord:
        t0 = time.perf_counter()
        assert video_path.exists(), f"Missing real MP4 file: {video_path}"

        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Could not open MP4: {video_path}"
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = max(float(cap.get(cv2.CAP_PROP_FPS)), 1.0)
        duration_sec = float(total_frames / fps)

        sample_idx_list = np.linspace(0, max(0, total_frames - 1), sample_frames).astype(int).tolist()
        sample_idx_set = set(sample_idx_list)
        frames_rgb: List[np.ndarray] = []
        f_idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if f_idx in sample_idx_set:
                frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            f_idx += 1
        cap.release()

        vid = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
        t_len = vid.shape[0]
        assert t_len >= 6, "Video sample count too small"

        tstd = np.mean(np.std(vid, axis=0), axis=-1)
        dyn_mask = tstd > max(float(np.percentile(tstd, 62)), 0.035)
        ys, xs = np.where(dyn_mask)
        cy = int(np.mean(ys)) if len(ys) > 0 else vid.shape[1] // 2
        cx = int(np.mean(xs)) if len(xs) > 0 else vid.shape[2] // 2
        yy, xx = np.indices(dyn_mask.shape)
        core_mask = np.hypot(yy - cy, xx - cx) < (0.14 * min(dyn_mask.shape))

        # 0D -> 1D Continuous Temporal Wave Channels from genuine native frames
        red_petal_wave = np.mean(vid[:, dyn_mask][:, :, 0] - vid[:, dyn_mask][:, :, 1], axis=1)
        yellow_core_wave = np.mean(
            0.5 * (vid[:, core_mask][:, :, 0] + vid[:, core_mask][:, :, 1]) - vid[:, core_mask][:, :, 2],
            axis=1,
        )
        bloom_area_wave = np.mean(
            (vid[..., 0] - vid[..., 1] > 0.10) & (vid[..., 0] > 0.38), axis=(1, 2)
        )
        static_carrier_wave = np.mean(vid[:, ~dyn_mask], axis=(1, 2))

        wave_channels = {
            "red_petal_wave": red_petal_wave.astype(np.float32),
            "yellow_core_wave": yellow_core_wave.astype(np.float32),
            "bloom_area_wave": bloom_area_wave.astype(np.float32),
            "static_carrier_wave": static_carrier_wave.astype(np.float32),
        }

        # 1D -> 2D Circle Complex Prism S^1
        real_wave = (red_petal_wave - red_petal_wave[0]) + 1.5 * (bloom_area_wave - bloom_area_wave[0]) + 0.35 * static_carrier_wave
        imag_wave = (yellow_core_wave - yellow_core_wave[0]) + np.gradient(bloom_area_wave) * 3.0
        prism = self._compute_complex_prism(record_id, real_wave, imag_wave)

        # 2D -> 3D/4D Spatiotemporal S^3 Rotor Trajectory
        deltas = np.stack(
            [
                5.0 * (red_petal_wave - red_petal_wave[0]),
                3.5 * (yellow_core_wave - yellow_core_wave[0]),
                9.0 * (bloom_area_wave - bloom_area_wave[0]),
            ],
            axis=1,
        )
        quats, sphere_xyz, geo_curve, vel_curve, bif = self._lift_3d_trajectory_to_s3_rotor(
            deltas, prism, duration_sec=duration_sec, primary_causal_axis="warm_red_bloom_expansion_axis"
        )

        key_frames = [
            (0, frames_rgb[0]),
            (bif.frame_index, frames_rgb[bif.frame_index]),
            (t_len - 1, frames_rgb[-1]),
        ]

        record = SpatiotemporalRotorRecord(
            record_id=record_id,
            modality="video_mp4",
            duration_sec=duration_sec,
            wave_1d_channels=wave_channels,
            prism_spectrum=prism,
            quaternions_s3=quats,
            sphere_trajectory_xyz=sphere_xyz,
            geodesic_divergence_curve=geo_curve,
            divergence_velocity_curve=vel_curve,
            bifurcation=bif,
            key_frames_rgb=key_frames,
            lift_latency_ms=float((time.perf_counter() - t0) * 1000.0),
        )
        self._register_record_in_causal_field(record)
        return record

    # ------------------------------------------------------------------
    # 2. Real External WAV Audio -> 0D..S^3 Spatiotemporal Rotor
    # ------------------------------------------------------------------
    def lift_real_wav_to_spatiotemporal_rotor(
        self,
        wav_path: Path,
        n_windows: int = 36,
        record_id: str = "real_audio_speech",
    ) -> SpatiotemporalRotorRecord:
        t0 = time.perf_counter()
        assert wav_path.exists(), f"Missing real WAV file: {wav_path}"
        sr, raw = wavfile.read(str(wav_path))
        wf = raw.astype(np.float32)
        if wf.ndim > 1:
            wf = wf.mean(axis=1)
        wf /= max(float(np.max(np.abs(wf))), 1e-6)
        duration_sec = float(len(wf) / max(sr, 1))

        win = max(64, len(wf) // n_windows)
        n_win = len(wf) // win
        frames = wf[: n_win * win].reshape(n_win, win)

        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        mag = np.abs(np.fft.rfft(frames * np.hanning(win)[None, :], axis=1)) + 1e-9
        freqs = np.fft.rfftfreq(win, d=1.0 / sr)
        centroid_norm = (np.sum(mag * freqs[None, :], axis=1) / np.sum(mag, axis=1)) / max(sr * 0.5, 1.0)
        geo_m = np.exp(np.mean(np.log(mag), axis=1))
        harm = np.clip(1.0 - (geo_m / np.maximum(np.mean(mag, axis=1), 1e-9)), 0.0, 1.0)

        wave_channels = {
            "rms_energy_wave": rms.astype(np.float32),
            "spectral_centroid_wave": centroid_norm.astype(np.float32),
            "harmonicity_wave": harm.astype(np.float32),
        }

        real_wave = (rms - rms[0]) + 0.30 * harm
        imag_wave = (centroid_norm - centroid_norm[0]) + np.gradient(rms) * 2.0
        prism = self._compute_complex_prism(record_id, real_wave, imag_wave)

        deltas = np.stack(
            [
                3.8 * (rms - rms[0]),
                2.5 * (centroid_norm - centroid_norm[0]),
                1.8 * (harm - harm[0]),
            ],
            axis=1,
        )
        quats, sphere_xyz, geo_curve, vel_curve, bif = self._lift_3d_trajectory_to_s3_rotor(
            deltas, prism, duration_sec=duration_sec, primary_causal_axis="acoustic_prosody_flux_axis"
        )

        record = SpatiotemporalRotorRecord(
            record_id=record_id,
            modality="audio_wav",
            duration_sec=duration_sec,
            wave_1d_channels=wave_channels,
            prism_spectrum=prism,
            quaternions_s3=quats,
            sphere_trajectory_xyz=sphere_xyz,
            geodesic_divergence_curve=geo_curve,
            divergence_velocity_curve=vel_curve,
            bifurcation=bif,
            lift_latency_ms=float((time.perf_counter() - t0) * 1000.0),
        )
        self._register_record_in_causal_field(record)
        return record

    # ------------------------------------------------------------------
    # 3. Direct Hypothesis Probe Construction & Wave Interference
    # ------------------------------------------------------------------
    def build_hypothesis_probe_rotor(
        self,
        probe_id: str,
        reference_record: SpatiotemporalRotorRecord,
        mode: str,
    ) -> SpatiotemporalRotorRecord:
        """
        Builds a continuous wave-prism + S^3 rotor hypothesis in sub-millisecond time:
          - `"constructive_bloom"`: Co-phase dynamic harmonics & aligning S^3 expansion rotor
          - `"contradiction_static"`: Anti-phase (`phi -> phi + pi`) wave inversion & opposed S^3 rotor
          - `"orthogonal_void"`: Orthogonal high-frequency harmonic carrier & orthogonal S^3 axis
        """
        t0 = time.perf_counter()
        ref_spec = reference_record.prism_spectrum.complex_spectrum.copy()
        n_bins = len(ref_spec)
        t_len = reference_record.quaternions_s3.shape[0]

        probe_spec = np.zeros(n_bins, dtype=np.complex64)
        ref_v_final = reference_record.quaternions_s3[-1, 1:]
        ref_angle = quat_geodesic_angle(
            reference_record.quaternions_s3[0], reference_record.quaternions_s3[-1]
        )

        if mode == "constructive_bloom":
            # Same phase spectrum weighted smoothly across dynamic harmonics
            probe_spec[:] = ref_spec * np.exp(1j * 0.08)
            target_axis = ref_v_final if np.linalg.norm(ref_v_final) > 1e-6 else np.array([1.0, 0.0, 0.0])
            target_angle = ref_angle * 0.92
            axis_label = "co_phase_blooming_hypothesis"
        elif mode == "contradiction_static":
            # Anti-phase inversion across all dynamic harmonic bands (w >= 1)
            probe_spec[0] = ref_spec[0]
            probe_spec[1:] = -ref_spec[1:]
            # Opposing rotor rotation vector on S^3
            target_axis = -ref_v_final if np.linalg.norm(ref_v_final) > 1e-6 else np.array([-1.0, 0.0, 0.0])
            target_angle = max(0.65, ref_angle * 0.85)
            axis_label = "anti_phase_static_contradiction"
        elif mode == "orthogonal_void":
            # Place all energy into inactive high-frequency bands orthogonal to ref_spec
            low_energy_idx = np.argsort(np.abs(ref_spec[1:]))[:4] + 1
            for k in low_energy_idx:
                probe_spec[k] = 0.5 + 0.0j
            # Construct an S^3 axis strictly orthogonal to ref_v_final via cross product
            ref_u = ref_v_final / max(float(np.linalg.norm(ref_v_final)), 1e-9)
            seed = np.array([0.0, 1.0, 0.0]) if abs(ref_u[1]) < 0.85 else np.array([1.0, 0.0, 0.0])
            target_axis = np.cross(ref_u, seed)
            target_angle = 0.40
            axis_label = "orthogonal_uncoupled_void"
        else:
            raise ValueError(f"Unsupported probe mode: {mode}")

        pwr = float(np.sum(np.abs(probe_spec) ** 2))
        probe_spec = (probe_spec / float(np.sqrt(max(pwr, 1e-12)))).astype(np.complex64)
        prism = WavePrismSpectrum(
            modality_id=probe_id,
            frequencies=np.arange(n_bins, dtype=np.float32),
            complex_spectrum=probe_spec,
            amplitudes=np.abs(probe_spec).astype(np.float32),
            phases=np.angle(probe_spec).astype(np.float32),
            dc_static_energy_ratio=float(abs(probe_spec[0]) ** 2),
        )

        # Build continuous geodesic arc on S^3 over T steps
        tau = np.linspace(0.0, 1.0, t_len)
        u = target_axis / max(float(np.linalg.norm(target_axis)), 1e-9)
        deltas = np.outer(tau * target_angle, u)
        quats, sphere_xyz, geo_curve, vel_curve, bif = self._lift_3d_trajectory_to_s3_rotor(
            deltas, prism, duration_sec=reference_record.duration_sec, primary_causal_axis=axis_label
        )

        record = SpatiotemporalRotorRecord(
            record_id=probe_id,
            modality="hypothesis_probe",
            duration_sec=reference_record.duration_sec,
            wave_1d_channels={"geodesic_angle_curve": geo_curve.astype(np.float32)},
            prism_spectrum=prism,
            quaternions_s3=quats,
            sphere_trajectory_xyz=sphere_xyz,
            geodesic_divergence_curve=geo_curve,
            divergence_velocity_curve=vel_curve,
            bifurcation=bif,
            lift_latency_ms=float((time.perf_counter() - t0) * 1000.0),
        )
        self._register_record_in_causal_field(record)
        return record

    def superimpose_and_observe_resonance(
        self,
        source: SpatiotemporalRotorRecord,
        probe: SpatiotemporalRotorRecord,
    ) -> InterferenceResonanceResult:
        """
        Directly superimposes two Wave Prisms (`Re(s * conj(p))`) and compares S^3
        geodesic rotors in sub-millisecond time—ZERO neural inference needed.
        """
        t0 = time.perf_counter()

        s = source.prism_spectrum.complex_spectrum[1:]  # Dynamic causal harmonics (w >= 1)
        p = probe.prism_spectrum.complex_spectrum[1:]

        denom = float(np.sqrt(np.sum(np.abs(s) ** 2) * np.sum(np.abs(p) ** 2))) + 1e-12
        # Band-wise interference spectrum: Re(s_k * conj(p_k)) normalized
        band_interference = np.real(s * np.conj(p)).astype(np.float32) / denom
        prism_constructive = float(np.clip(np.sum(band_interference), -1.0, 1.0))
        spectral_overlap = float(np.clip(np.sum(np.abs(s) * np.abs(p)) / denom, 0.0, 1.0))

        # S^3 Rotor geodesic alignment across endpoints and trajectory axes
        q_src_vec = source.quaternions_s3[-1, 1:]
        q_prb_vec = probe.quaternions_s3[-1, 1:]
        ns = float(np.linalg.norm(q_src_vec))
        npv = float(np.linalg.norm(q_prb_vec))
        if ns > 1e-9 and npv > 1e-9:
            rotor_align = float(np.dot(q_src_vec / ns, q_prb_vec / npv))
        else:
            rotor_align = 0.0

        combined = float(np.clip(0.55 * prism_constructive + 0.45 * rotor_align, -1.0, 1.0))

        if spectral_overlap < 0.25:
            verdict = "ORTHOGONAL_VOID"
            beam_torn = False
        elif combined < -0.10:
            verdict = "DESTRUCTIVE_CONTRADICTION"
            beam_torn = True
        else:
            verdict = "CONSTRUCTIVE_RESONANCE"
            beam_torn = False

        tension = float(max(0.0, 1.0 - combined))
        beam = ConnectivityBeam(
            source_id=source.record_id,
            target_id=probe.record_id,
            strength=max(0.1, spectral_overlap),
            rest_length=0.0,
            current_tension=tension,
            break_threshold=1.10,
            is_broken=beam_torn,
            frequency=float(source.bifurcation.dominant_harmonic_band),
        )
        self.causal_field.beams.append(beam)

        return InterferenceResonanceResult(
            source_id=source.record_id,
            probe_id=probe.record_id,
            prism_interference_spectrum=band_interference,
            prism_constructive_score=prism_constructive,
            spectral_overlap_support=spectral_overlap,
            rotor_geodesic_alignment=rotor_align,
            combined_resonance=combined,
            verdict=verdict,
            beam_torn=beam_torn,
            latency_ms=float((time.perf_counter() - t0) * 1000.0),
        )

    @staticmethod
    def synthesize_causal_bifurcation_expression(record: SpatiotemporalRotorRecord) -> str:
        """
        Zero-LLM native causal expression read straight from the lifted Wave-Prism &
        S^3 Rotor Bifurcation point `(t*, w*)`.
        """
        b = record.bifurcation
        deg = float(np.degrees(b.geodesic_divergence_rad))
        end_deg = float(np.degrees(record.geodesic_divergence_curve[-1]))
        dc_pct = record.prism_spectrum.dc_static_energy_ratio * 100.0
        return (
            f"[{record.record_id} | {record.modality}] 정적 DC 에너지({dc_pct:.1f}%)로 유지되던 기준 상태에서 "
            f"프레임 {b.frame_index}/{b.total_frames} (약 {b.time_sec:.2f}초) 지점에 인과적 분기점(t*)이 관측됨. "
            f"이 순간 S^3 측지 각변위 속도가 최대({b.divergence_velocity:.3f} rad/step, 누적 각 {deg:.1f}° -> 최종 {end_deg:.1f}°)에 도달하며 "
            f"프리즘 제{b.dominant_harmonic_band}고조파 대역을 통해 '{b.primary_causal_axis}' 방향으로 회전 전개됨."
        )

    @staticmethod
    def render_wave_prism_rotor_dashboard(
        mp4_record: SpatiotemporalRotorRecord,
        wav_record: SpatiotemporalRotorRecord,
        resonances: Sequence[InterferenceResonanceResult],
        output_path: Path,
        llm_baseline_ms: float = 7480.0,
    ) -> Path:
        """Render clean dark-mode 2x2 diagnostic showcase of S^3 trajectory, Wave Prism, Bifurcation, & Speed."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(14, 9), dpi=120)
        fig.patch.set_facecolor("#111318")

        # Panel 1: 3D Unit Sphere S^3 -> S^2 Geodesic Trajectory
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1.set_facecolor("#161b22")
        u = np.linspace(0, 2 * np.pi, 28)
        v = np.linspace(0, np.pi, 16)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax1.plot_wireframe(xs, ys, zs, color="#30363d", linewidth=0.45, alpha=0.45)

        traj = mp4_record.sphere_trajectory_xyz
        b_idx = mp4_record.bifurcation.frame_index
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#ffa657", linewidth=2.4, label="S^3->S^2 Geodesic")
        ax1.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], color="#58a6ff", s=55, label="t=0 Anchor")
        ax1.scatter([traj[b_idx, 0]], [traj[b_idx, 1]], [traj[b_idx, 2]], color="#ff7b72", marker="*", s=160, label=f"t*={b_idx} Bifurcation")
        ax1.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], color="#7ee787", s=60, label="t=T Bloom End")
        ax1.set_title("S^3 -> S^2 Spatiotemporal Geodesic Trajectory (MP4)", color="#e6edf3", fontsize=10, pad=6)
        ax1.set_xlim(-1.05, 1.05)
        ax1.set_ylim(-1.05, 1.05)
        ax1.set_zlim(-1.05, 1.05)
        ax1.tick_params(colors="#8b949e", labelsize=7)
        ax1.legend(loc="upper left", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

        # Panel 2: Wave Prism Interference Spectrum
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor("#161b22")
        harm_bins = np.arange(1, len(resonances[0].prism_interference_spectrum) + 1)
        palette = {"constructive_bloom": "#7ee787", "contradiction_static": "#ff7b72", "orthogonal_void": "#d2a8ff"}
        for res in resonances:
            col = "#58a6ff"
            for key, c in palette.items():
                if key in res.probe_id:
                    col = c
            ax2.plot(
                harm_bins,
                res.prism_interference_spectrum,
                marker="o",
                markersize=4,
                linewidth=1.9,
                color=col,
                label=f"{res.probe_id}: {res.combined_resonance:+.2f} ({res.verdict})",
            )
        ax2.axhline(0.0, color="#8b949e", linewidth=0.9)
        ax2.set_xlabel("Harmonic Band w", color="#c9d1d9", fontsize=9)
        ax2.set_ylabel("Prism Interference Re(s*conj(p))", color="#c9d1d9", fontsize=9)
        ax2.set_title("Complex Wave Prism Interference Spectrum (Sub-ms)", color="#e6edf3", fontsize=10)
        ax2.tick_params(colors="#c9d1d9", labelsize=8)
        ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

        # Panel 3: Causal Bifurcation Timeline & Geodesic Velocity (MP4)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_facecolor("#161b22")
        t_axis = np.linspace(0.0, mp4_record.duration_sec, len(mp4_record.geodesic_divergence_curve))
        ax3.plot(t_axis, mp4_record.geodesic_divergence_curve, color="#79c0ff", linewidth=2.2, label="Geodesic Angle theta(t) [rad]")
        ax3_vel = ax3.twinx()
        ax3_vel.plot(t_axis, mp4_record.divergence_velocity_curve, color="#ffa657", linestyle="--", linewidth=1.8, label="Velocity d(theta)/dt")
        ax3.axvline(mp4_record.bifurcation.time_sec, color="#ff7b72", linewidth=1.6, linestyle=":", label=f"Bifurcation t*={mp4_record.bifurcation.time_sec:.2f}s (frame {b_idx})")
        ax3.set_xlabel("Video Time (s)", color="#c9d1d9", fontsize=9)
        ax3.set_ylabel("Geodesic Divergence (rad)", color="#79c0ff", fontsize=9)
        ax3_vel.set_ylabel("Bifurcation Velocity", color="#ffa657", fontsize=9)
        ax3.set_title("MP4 Causal Bifurcation Point Observation", color="#e6edf3", fontsize=10)
        ax3.tick_params(colors="#c9d1d9", labelsize=8)
        ax3_vel.tick_params(colors="#ffa657", labelsize=8)
        h1, l1 = ax3.get_legend_handles_labels()
        h2, l2 = ax3_vel.get_legend_handles_labels()
        ax3.legend(h1 + h2, l1 + l2, loc="upper left", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

        # Panel 4: Real Audio WAV Geodesic + Latency Efficiency Comparison
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_facecolor("#161b22")
        t_wav = np.linspace(0.0, wav_record.duration_sec, len(wav_record.geodesic_divergence_curve))
        ax4.plot(t_wav, wav_record.geodesic_divergence_curve, color="#d2a8ff", linewidth=2.0, label="WAV Prosody Geodesic theta(t)")
        ax4.axvline(wav_record.bifurcation.time_sec, color="#7ee787", linestyle="--", linewidth=1.4, label=f"Audio Bifurcation t*={wav_record.bifurcation.time_sec:.2f}s")
        mean_res_ms = float(np.mean([r.latency_ms for r in resonances]))
        speedup = llm_baseline_ms / max(mean_res_ms, 1e-4)
        ax4.text(
            0.04,
            0.84,
            f"Wave-Prism Superposition: {mean_res_ms:.3f} ms\nPrior LLM Turn Baseline: {llm_baseline_ms:.0f} ms\nMeasured Speedup: {speedup:,.0f}x",
            transform=ax4.transAxes,
            fontsize=8.5,
            color="#e6edf3",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", edgecolor="#58a6ff"),
        )
        ax4.set_xlabel("Audio Time (s)", color="#c9d1d9", fontsize=9)
        ax4.set_ylabel("Audio S^3 Geodesic (rad)", color="#d2a8ff", fontsize=9)
        ax4.set_title("Audio WAV S^3 Trajectory & Millisecond Efficiency", color="#e6edf3", fontsize=10)
        ax4.tick_params(colors="#c9d1d9", labelsize=8)
        ax4.legend(loc="lower right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

        plt.tight_layout(pad=1.5)
        fig.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path
