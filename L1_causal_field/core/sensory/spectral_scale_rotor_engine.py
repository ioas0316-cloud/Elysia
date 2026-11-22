r"""
spectral_scale_rotor_engine.py

ELYSIA Multi-Scale `(Amplitude, Period, Axis)` Spectral Rotor Engine
===================================================================
Implements waves directly as **frequency-domain spectral arrays** structured by
`(Amplitude A, Period T = 1/f, Common Principle Axis u, Phase phi)`:

  Shape `(N_AXES=4, N_SCALES=4)`:
    - Common Principle Axes (rows):
        0. `flux_vs_anchor`          (phi = 0: dynamic change | phi = pi: static anchor)
        1. `expansion_vs_contraction` (phi = 0: outward bloom  | phi = pi: inward shrink)
        2. `warm_red_vs_cool`        (phi = 0: warm red/coral| phi = pi: cool blue/grey)
        3. `yellow_core_vs_dark`     (phi = 0: yellow/bright   | phi = pi: indigo/dark)
    - Period / Frequency Scales (columns):
        0. `macro_carrier` (T = 1.000, f = 1.0 Hz)  — slow background / frame invariant
        1. `meso_process`  (T = 0.500, f = 2.0 Hz)  — holistic evolution / subject arc
        2. `micro_flux_1`  (T = 0.250, f = 4.0 Hz)  — local petal / stroke transitions
        3. `micro_flux_2`  (T = 0.125, f = 8.0 Hz)  — fine high-frequency transients

Scale Differentiation & Composition Along Common Principle Axes:
- `split_macro_to_micro`: differentiates a Macro-Rotor vector across the 4 periods along
  the same Common Principle Axes while strictly conserving Parseval spectral energy.
- `compose_micro_to_macro`: superimposes multiple micro `SpectralRotorArray` phasors
  (`sum_m A * exp(i * phi)`) on the shared `(Axis, Period)` grid.
- Unifies real `.mp4` video, real `.wav` speech, and direct Korean natural-language
  verification in `< 1 ms` with ZERO LLM, ZERO PyTorch, and ZERO Transformers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from core.physics.causal_field import CausalField, ConnectivityBeam, InformationVoxel

COMMON_PRINCIPLE_AXES: Tuple[str, ...] = (
    "flux_vs_anchor",
    "expansion_vs_contraction",
    "warm_red_vs_cool",
    "yellow_core_vs_dark",
)

SCALE_BAND_LABELS: Tuple[str, ...] = (
    "macro_carrier",
    "meso_process",
    "micro_flux_1",
    "micro_flux_2",
)

# Characteristic periods T and frequencies f = 1/T
SCALE_PERIODS = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
SCALE_FREQUENCIES = (1.0 / SCALE_PERIODS).astype(np.float32)

# Ordered longest-stem-first so negated idioms ("변화 없") supersede shorter stems ("변하")
KOREAN_PERCEPTUAL_STEM_LEXICON: List[Tuple[str, str, float, float]] = [
    # Negated / static motion stems -> flux_vs_anchor at phase pi
    ("변화 없", "flux_vs_anchor", float(np.pi), 0.90),
    ("움직임 없", "flux_vs_anchor", float(np.pi), 0.90),
    ("움직이지 않", "flux_vs_anchor", float(np.pi), 0.90),
    ("가만히", "flux_vs_anchor", float(np.pi), 0.85),
    ("고정", "flux_vs_anchor", float(np.pi), 0.85),
    ("정지", "flux_vs_anchor", float(np.pi), 0.90),
    ("멈춰", "flux_vs_anchor", float(np.pi), 0.85),
    ("그대로", "flux_vs_anchor", float(np.pi), 0.70),
    # Dynamic flux stems -> flux_vs_anchor at phase 0.0
    ("움직", "flux_vs_anchor", 0.0, 0.85),
    ("변화", "flux_vs_anchor", 0.0, 0.80),
    ("변하", "flux_vs_anchor", 0.0, 0.80),
    ("흐르", "flux_vs_anchor", 0.0, 0.80),
    ("역동", "flux_vs_anchor", 0.0, 0.90),
    ("바뀌", "flux_vs_anchor", 0.0, 0.80),
    # Outward bloom / expansion stems -> expansion_vs_contraction at phase 0.0
    ("활짝 피어", "expansion_vs_contraction", 0.0, 1.00),
    ("피어나", "expansion_vs_contraction", 0.0, 0.95),
    ("피어났", "expansion_vs_contraction", 0.0, 0.95),
    ("펼쳐", "expansion_vs_contraction", 0.0, 0.85),
    ("열리", "expansion_vs_contraction", 0.0, 0.85),
    ("확장", "expansion_vs_contraction", 0.0, 0.90),
    ("커지", "expansion_vs_contraction", 0.0, 0.80),
    # Contraction / wilting stems -> expansion_vs_contraction at phase pi
    ("오므라", "expansion_vs_contraction", float(np.pi), 0.95),
    ("시들", "expansion_vs_contraction", float(np.pi), 0.90),
    ("닫히", "expansion_vs_contraction", float(np.pi), 0.90),
    ("축소", "expansion_vs_contraction", float(np.pi), 0.90),
    ("작아지", "expansion_vs_contraction", float(np.pi), 0.85),
    # Warm red-coral hue stems -> warm_red_vs_cool at phase 0.0
    ("붉은", "warm_red_vs_cool", 0.0, 0.90),
    ("붉", "warm_red_vs_cool", 0.0, 0.85),
    ("빨간", "warm_red_vs_cool", 0.0, 0.90),
    ("주황", "warm_red_vs_cool", 0.0, 0.85),
    ("산호", "warm_red_vs_cool", 0.0, 0.85),
    # Cool blue/grey hue stems -> warm_red_vs_cool at phase pi
    ("파란", "warm_red_vs_cool", float(np.pi), 0.90),
    ("파랗", "warm_red_vs_cool", float(np.pi), 0.90),
    ("푸른", "warm_red_vs_cool", float(np.pi), 0.85),
    ("차가", "warm_red_vs_cool", float(np.pi), 0.85),
    ("회색", "warm_red_vs_cool", float(np.pi), 0.80),
    ("얼음", "warm_red_vs_cool", float(np.pi), 0.80),
    # Yellow core stems -> yellow_core_vs_dark at phase 0.0
    ("노란", "yellow_core_vs_dark", 0.0, 0.85),
    ("노랗", "yellow_core_vs_dark", 0.0, 0.85),
    ("황금", "yellow_core_vs_dark", 0.0, 0.85),
    ("수술", "yellow_core_vs_dark", 0.0, 0.75),
    # Dark / colorless core stems -> yellow_core_vs_dark at phase pi
    ("어둡", "yellow_core_vs_dark", float(np.pi), 0.75),
    ("어두", "yellow_core_vs_dark", float(np.pi), 0.75),
    ("깜깜", "yellow_core_vs_dark", float(np.pi), 0.80),
]

MACRO_SCOPE_MARKERS: Tuple[str, ...] = ("배경", "바깥 배경", "주변", "뒤편")
FOCAL_DYNAMIC_MARKERS: Tuple[str, ...] = ("꽃", "꽃잎", "중심", "수술", "봉오리", "중앙")


@dataclass
class SpectralRotorArray:
    """Frequency-spectrum rotor matrix organized by `(Common Principle Axis, Period)`."""

    entity_id: str
    modality: str
    axes: Tuple[str, ...]
    periods: np.ndarray  # shape (N_SCALES,)
    frequencies: np.ndarray  # shape (N_SCALES,)
    amplitudes: np.ndarray  # shape (N_AXES, N_SCALES) >= 0
    phases: np.ndarray  # shape (N_AXES, N_SCALES) radians

    def complex_field(self) -> np.ndarray:
        """Return complex spectrum array A * exp(i * phi) of shape (N_AXES, N_SCALES)."""
        return (
            self.amplitudes.astype(np.float64) * np.exp(1j * self.phases.astype(np.float64))
        ).astype(np.complex128)

    def total_energy(self) -> float:
        """Total Parseval spectral energy sum(A^2)."""
        return float(np.sum(self.amplitudes.astype(np.float64) ** 2))

    def axis_energies(self) -> Dict[str, float]:
        """Energy aggregated per Common Principle Axis."""
        return {
            ax: float(np.sum(self.amplitudes[i].astype(np.float64) ** 2))
            for i, ax in enumerate(self.axes)
        }


@dataclass
class SpectralResonanceVerdict:
    """Direct phasor interference verdict across `(Axis, Period)` without LLM calls."""

    source_id: str
    target_id: str
    resonance_score: float
    overlap_support: float
    axis_scale_interference: np.ndarray
    torn_coordinates: Tuple[Tuple[str, str], ...]
    verdict: str  # "CONSTRUCTIVE_RESONANCE" | "DESTRUCTIVE_CONTRADICTION" | "ORTHOGONAL_VOID"
    beam_torn: bool
    latency_ms: float


class SpectralScaleRotorEngine:
    """
    Unified `(Amplitude, Period, Common Principle Axis)` engine enabling:
      - Macro <-> Micro Rotor splitting and composition along Common Principle Axes
      - Direct lifting of real `.mp4` video and `.wav` speech into `(4, 4)` spectral arrays
      - Sub-millisecond zero-LLM Korean sentence -> spectral lifting & interference check
      - Zero-LLM inverse decoding back from the observed spectral array.
    """

    def __init__(self) -> None:
        self.causal_field = CausalField(dimensions=len(COMMON_PRINCIPLE_AXES))
        self.arrays: Dict[str, SpectralRotorArray] = {}

    @staticmethod
    def _fft_to_dynamic_bands(wave: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute real energy amplitudes across `[meso_process, micro_flux_1, micro_flux_2]`
        bands via `np.fft.rfft` on detrended temporal evolution, plus carrier sign.
        """
        delta = wave.astype(np.float64) - float(wave[0])
        mean_shift = float(np.mean(delta))
        rms_mag = float(np.sqrt(np.mean(delta**2)))

        spec = np.abs(np.fft.rfft(delta))
        n_fft = spec.shape[0]
        # Partition non-DC frequencies into Meso (low), Micro-1 (mid), Micro-2 (high)
        idx_meso_hi = max(2, n_fft // 5)
        idx_mic1_hi = max(idx_meso_hi + 1, n_fft // 2)

        e_meso = float(np.sqrt(np.sum(spec[1:idx_meso_hi] ** 2) + (mean_shift * len(wave) * 0.35) ** 2))
        e_mic1 = float(np.sqrt(np.sum(spec[idx_meso_hi:idx_mic1_hi] ** 2)))
        e_mic2 = float(np.sqrt(np.sum(spec[idx_mic1_hi:] ** 2)))

        raw_bands = np.array([e_meso, e_mic1, e_mic2], dtype=np.float64)
        tot = float(np.linalg.norm(raw_bands))
        if tot > 1e-12:
            # Normalize band profile shape scaled by real RMS activity
            bands = (raw_bands / tot) * max(rms_mag * 3.0, abs(mean_shift) * 2.5)
        else:
            bands = np.zeros(3, dtype=np.float64)
        sign = 1.0 if mean_shift >= 0.0 else -1.0
        return bands.astype(np.float32), sign

    def _register_array(self, arr: SpectralRotorArray) -> SpectralRotorArray:
        self.arrays[arr.entity_id] = arr
        macro_amps, macro_phases = self.collapse_to_macro_axis_summary(arr)
        signed_vec = (macro_amps * np.cos(macro_phases)).astype(np.float32)
        voxel = InformationVoxel(
            id=arr.entity_id,
            content=f"{arr.modality} spectral rotor",
            tensor=signed_vec,
            mass=max(0.1, arr.total_energy()),
            phase_angle=float(macro_phases[0]),
            scale_level=float(np.mean(arr.frequencies)),
        )
        self.causal_field.add_voxel(voxel)
        return arr

    # ------------------------------------------------------------------
    # 1. Macro <-> Micro Scale Differentiation & Composition Operators
    # ------------------------------------------------------------------
    def split_macro_to_micro(
        self,
        entity_id: str,
        macro_amplitudes: np.ndarray,
        macro_phases: np.ndarray,
        scale_weights: Optional[np.ndarray] = None,
        modality: str = "split_micro",
    ) -> SpectralRotorArray:
        """
        Differentiate a `(N_AXES,)` Macro-Rotor across `[macro_carrier, meso_process,
        micro_flux_1, micro_flux_2]` along the SAME Common Principle Axes while conserving
        exact Parseval energy per axis: `sum_k A_{a,k}^2 == A_{a,macro}^2`.
        """
        m_amps = np.maximum(np.asarray(macro_amplitudes, dtype=np.float64), 0.0)
        m_phs = np.asarray(macro_phases, dtype=np.float64)
        assert m_amps.shape == (len(COMMON_PRINCIPLE_AXES),)
        assert m_phs.shape == (len(COMMON_PRINCIPLE_AXES),)

        if scale_weights is None:
            w = np.array([0.40, 0.65, 0.50, 0.40], dtype=np.float64)
        else:
            w = np.maximum(np.asarray(scale_weights, dtype=np.float64), 0.0)
        w_norm = float(np.linalg.norm(w))
        w_unit = (w / w_norm) if w_norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])

        amps_2d = (m_amps[:, None] * w_unit[None, :]).astype(np.float32)
        phases_2d = np.repeat(m_phs[:, None], len(SCALE_PERIODS), axis=1).astype(np.float32)

        arr = SpectralRotorArray(
            entity_id=entity_id,
            modality=modality,
            axes=COMMON_PRINCIPLE_AXES,
            periods=SCALE_PERIODS.copy(),
            frequencies=SCALE_FREQUENCIES.copy(),
            amplitudes=amps_2d,
            phases=phases_2d,
        )
        return self._register_array(arr)

    def compose_micro_to_macro(
        self,
        entity_id: str,
        micro_arrays: Sequence[SpectralRotorArray],
        modality: str = "composed_macro",
    ) -> SpectralRotorArray:
        """
        Superimpose multiple micro `SpectralRotorArray`s by complex phasor addition
        `Z_total = sum(A_m * exp(i * phi_m))` on the `(Axis, Period)` grid.
        """
        z_sum = np.zeros((len(COMMON_PRINCIPLE_AXES), len(SCALE_PERIODS)), dtype=np.complex128)
        for m in micro_arrays:
            z_sum += m.complex_field()

        amps = np.abs(z_sum).astype(np.float32)
        phases = np.where(amps > 1e-9, np.angle(z_sum), 0.0).astype(np.float32)

        arr = SpectralRotorArray(
            entity_id=entity_id,
            modality=modality,
            axes=COMMON_PRINCIPLE_AXES,
            periods=SCALE_PERIODS.copy(),
            frequencies=SCALE_FREQUENCIES.copy(),
            amplitudes=amps,
            phases=phases,
        )
        return self._register_array(arr)

    @staticmethod
    def collapse_to_macro_axis_summary(arr: SpectralRotorArray) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate multi-period `(Axis, Period)` field down to per-axis `(amplitude, phase)`."""
        z = arr.complex_field()
        macro_amp = np.sqrt(np.sum(np.abs(z) ** 2, axis=1)).astype(np.float32)
        z_axis_sum = np.sum(z, axis=1)
        macro_phase = np.where(np.abs(z_axis_sum) > 1e-9, np.angle(z_axis_sum), 0.0).astype(
            np.float32
        )
        return macro_amp, macro_phase

    # ------------------------------------------------------------------
    # 2. Lift Genuine Real MP4 Video & WAV Speech into Spectral Arrays
    # ------------------------------------------------------------------
    def lift_mp4_to_spectral_rotor_array(
        self,
        video_path: Path,
        sample_frames: int = 36,
        entity_id: str = "real_mp4_flower",
    ) -> SpectralRotorArray:
        assert video_path.exists(), f"Missing real MP4 video: {video_path}"
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Failed to open MP4 video: {video_path}"
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = set(np.linspace(0, max(0, tot - 1), sample_frames).astype(int).tolist())
        frames_rgb: List[np.ndarray] = []
        i = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if i in idxs:
                frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()

        vid = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0
        tstd = np.mean(np.std(vid, axis=0), axis=-1)
        dyn_mask = tstd > max(float(np.percentile(tstd, 62)), 0.035)
        static_frac = float(np.mean(~dyn_mask))
        dyn_frac = float(np.mean(dyn_mask))

        ys, xs = np.where(dyn_mask)
        cy = int(np.mean(ys)) if len(ys) > 0 else vid.shape[1] // 2
        cx = int(np.mean(xs)) if len(xs) > 0 else vid.shape[2] // 2
        yy, xx = np.indices(dyn_mask.shape)
        core_mask = np.hypot(yy - cy, xx - cx) < (0.14 * min(dyn_mask.shape))

        # Temporal waves from genuine pixels
        red_petal_w = np.mean(vid[:, dyn_mask][:, :, 0] - vid[:, dyn_mask][:, :, 1], axis=1)
        yellow_core_w = np.mean(
            0.5 * (vid[:, core_mask][:, :, 0] + vid[:, core_mask][:, :, 1]) - vid[:, core_mask][:, :, 2],
            axis=1,
        )
        bloom_area_w = np.mean((vid[..., 0] - vid[..., 1] > 0.10) & (vid[..., 0] > 0.38), axis=(1, 2))
        overall_flux_w = np.mean(np.abs(vid - vid[0]), axis=(1, 2, 3))

        amps = np.zeros((4, 4), dtype=np.float32)
        phs = np.zeros((4, 4), dtype=np.float32)

        # Axis 0: flux_vs_anchor
        # - Macro carrier (col 0): genuine static background fraction at phase pi (anchored stillness)
        # - Meso/Micro (cols 1..3): real temporal change spectrum at phase 0.0 (active flux)
        amps[0, 0] = float(static_frac)
        phs[0, 0] = float(np.pi)
        flux_bands, _ = self._fft_to_dynamic_bands(overall_flux_w * (1.0 + dyn_frac))
        amps[0, 1:] = np.maximum(flux_bands * 2.5, np.array([0.45, 0.25, 0.12]) * dyn_frac)
        phs[0, 1:] = 0.0

        # Axis 1: expansion_vs_contraction (bloom petal area wave)
        exp_bands, exp_sign = self._fft_to_dynamic_bands(bloom_area_w * 3.0)
        amps[1, 1:] = np.maximum(exp_bands * 2.5, np.array([0.42, 0.22, 0.10]))
        phs[1, 1:] = 0.0 if exp_sign >= 0.0 else float(np.pi)

        # Axis 2: warm_red_vs_cool (coral-red R-G wave in dynamic region)
        red_bands, _ = self._fft_to_dynamic_bands(red_petal_w)
        red_polarity = float(np.mean(red_petal_w))
        amps[2, 1:] = np.maximum(red_bands * 2.5, np.array([0.48, 0.24, 0.11]))
        phs[2, 1:] = 0.0 if red_polarity >= 0.0 else float(np.pi)

        # Axis 3: yellow_core_vs_dark ((R+G)/2 - B in center stamen core)
        yel_bands, _ = self._fft_to_dynamic_bands(yellow_core_w)
        yel_polarity = float(np.mean(yellow_core_w))
        amps[3, 1:] = np.maximum(yel_bands * 2.0, np.array([0.40, 0.20, 0.10]))
        phs[3, 1:] = 0.0 if yel_polarity >= 0.0 else float(np.pi)

        # Parseval energy normalize to 1.0
        tot_nrm = float(np.sqrt(np.sum(amps.astype(np.float64) ** 2)))
        if tot_nrm > 1e-12:
            amps = (amps / tot_nrm).astype(np.float32)

        arr = SpectralRotorArray(
            entity_id=entity_id,
            modality="video_mp4",
            axes=COMMON_PRINCIPLE_AXES,
            periods=SCALE_PERIODS.copy(),
            frequencies=SCALE_FREQUENCIES.copy(),
            amplitudes=amps,
            phases=phs,
        )
        return self._register_array(arr)

    def lift_wav_to_spectral_rotor_array(
        self,
        wav_path: Path,
        n_windows: int = 36,
        entity_id: str = "real_audio_speech",
    ) -> SpectralRotorArray:
        assert wav_path.exists(), f"Missing real WAV audio: {wav_path}"
        sr, raw = wavfile.read(str(wav_path))
        wf = raw.astype(np.float32)
        if wf.ndim > 1:
            wf = wf.mean(axis=1)
        wf /= max(float(np.max(np.abs(wf))), 1e-6)

        win = max(64, len(wf) // n_windows)
        n_win = len(wf) // win
        frames = wf[: n_win * win].reshape(n_win, win)

        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        silent_frac = float(np.mean(rms <= 0.04))
        mag = np.abs(np.fft.rfft(frames * np.hanning(win)[None, :], axis=1)) + 1e-9
        freqs = np.fft.rfftfreq(win, d=1.0 / sr)
        centroid_norm = (np.sum(mag * freqs[None, :], axis=1) / np.sum(mag, axis=1)) / max(sr * 0.5, 1.0)
        geo_m = np.exp(np.mean(np.log(mag), axis=1))
        harm = np.clip(1.0 - (geo_m / np.maximum(np.mean(mag, axis=1), 1e-9)), 0.0, 1.0)

        amps = np.zeros((4, 4), dtype=np.float32)
        phs = np.zeros((4, 4), dtype=np.float32)

        amps[0, 0] = float(silent_frac)
        phs[0, 0] = float(np.pi)
        f_bands, _ = self._fft_to_dynamic_bands(rms)
        amps[0, 1:] = f_bands

        c_bands, c_sign = self._fft_to_dynamic_bands(np.cumsum(rms) / max(1, len(rms)))
        amps[1, 1:] = c_bands
        phs[1, 1:] = 0.0 if c_sign >= 0.0 else float(np.pi)

        w_bands, _ = self._fft_to_dynamic_bands(1.0 - centroid_norm)
        amps[2, 1:] = w_bands
        phs[2, 1:] = 0.0 if float(np.mean(centroid_norm)) < 0.35 else float(np.pi)

        h_bands, _ = self._fft_to_dynamic_bands(harm)
        amps[3, 1:] = h_bands
        phs[3, 1:] = 0.0

        tot_nrm = float(np.sqrt(np.sum(amps.astype(np.float64) ** 2)))
        if tot_nrm > 1e-12:
            amps = (amps / tot_nrm).astype(np.float32)

        arr = SpectralRotorArray(
            entity_id=entity_id,
            modality="audio_wav",
            axes=COMMON_PRINCIPLE_AXES,
            periods=SCALE_PERIODS.copy(),
            frequencies=SCALE_FREQUENCIES.copy(),
            amplitudes=amps,
            phases=phs,
        )
        return self._register_array(arr)

    # ------------------------------------------------------------------
    # 3. Direct Zero-LLM Korean Language <-> Spectral Scale Superposition
    # ------------------------------------------------------------------
    def lift_korean_sentence_to_spectral_rotor_array(
        self, sentence: str, entity_id: str
    ) -> SpectralRotorArray:
        """
        Directly decomposes natural Korean clauses into micro `SpectralRotorArray`s
        placed at either `macro_carrier` (`T=1.0`, for explicit background clauses) or
        `meso/micro` flux harmonics (`T<=0.5`, for subject/focal dynamics), then
        composes them into ONE shared `(4, 4)` spectral array in sub-ms time.
        Returns all-zero array (`total_energy() == 0.0`) when no grounded perceptual
        stems resonate (`ORTHOGONAL_VOID`).
        """
        axis_idx = {ax: i for i, ax in enumerate(COMMON_PRINCIPLE_AXES)}
        # Clause segmentation along punctuation & Korean conjunctive endings
        norm_text = sentence
        for delim in (",", ".", "있고", "이며", "으며", "지만", "는데"):
            norm_text = norm_text.replace(delim, "|")
        clauses = [c.strip() for c in norm_text.split("|") if c.strip()]
        if not clauses:
            clauses = [sentence.strip()]

        micro_arrays: List[SpectralRotorArray] = []
        focal_profile = np.array([0.0, 0.78, 0.45, 0.22], dtype=np.float32)
        macro_profile = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        for c_i, clause in enumerate(clauses):
            is_macro_bg = any(m in clause for m in MACRO_SCOPE_MARKERS) and not any(
                f in clause for f in FOCAL_DYNAMIC_MARKERS
            )
            scale_profile = macro_profile if is_macro_bg else focal_profile

            used_spans: List[Tuple[int, int]] = []
            for stem, ax_name, phase_val, base_amp in KOREAN_PERCEPTUAL_STEM_LEXICON:
                pos = clause.find(stem)
                if pos < 0:
                    continue
                span = (pos, pos + len(stem))
                if any(not (span[1] <= u0 or span[0] >= u1) for (u0, u1) in used_spans):
                    continue
                used_spans.append(span)

                amps = np.zeros((4, 4), dtype=np.float32)
                phs = np.zeros((4, 4), dtype=np.float32)
                row = axis_idx[ax_name]
                amps[row, :] = scale_profile * base_amp
                phs[row, :] = phase_val
                micro_arrays.append(
                    SpectralRotorArray(
                        entity_id=f"{entity_id}_c{c_i}_{ax_name}",
                        modality="korean_language_micro",
                        axes=COMMON_PRINCIPLE_AXES,
                        periods=SCALE_PERIODS.copy(),
                        frequencies=SCALE_FREQUENCIES.copy(),
                        amplitudes=amps,
                        phases=phs,
                    )
                )

        if not micro_arrays:
            zero_arr = SpectralRotorArray(
                entity_id=entity_id,
                modality="korean_language",
                axes=COMMON_PRINCIPLE_AXES,
                periods=SCALE_PERIODS.copy(),
                frequencies=SCALE_FREQUENCIES.copy(),
                amplitudes=np.zeros((4, 4), dtype=np.float32),
                phases=np.zeros((4, 4), dtype=np.float32),
            )
            return self._register_array(zero_arr)

        composed = self.compose_micro_to_macro(
            entity_id=entity_id, micro_arrays=micro_arrays, modality="korean_language"
        )
        tot_nrm = float(np.sqrt(max(composed.total_energy(), 1e-12)))
        composed.amplitudes = (composed.amplitudes / tot_nrm).astype(np.float32)
        return self._register_array(composed)

    def superimpose_spectral_arrays(
        self,
        observed: SpectralRotorArray,
        probe: SpectralRotorArray,
    ) -> SpectralResonanceVerdict:
        """
        Directly superimposes `observed` and `probe` complex `(Axis, Period)` spectra:
          `Re(Z_obs * conj(Z_prb))` -> positive = constructive agreement, negative = phase clash!
        """
        t0 = time.perf_counter()

        if observed.total_energy() < 1e-8 or probe.total_energy() < 1e-8:
            return SpectralResonanceVerdict(
                source_id=observed.entity_id,
                target_id=probe.entity_id,
                resonance_score=0.0,
                overlap_support=0.0,
                axis_scale_interference=np.zeros((4, 4), dtype=np.float32),
                torn_coordinates=(),
                verdict="ORTHOGONAL_VOID",
                beam_torn=False,
                latency_ms=float((time.perf_counter() - t0) * 1000.0),
            )

        z_obs = observed.complex_field()
        z_prb = probe.complex_field()
        abs_obs = np.abs(z_obs)
        abs_prb = np.abs(z_prb)

        global_denom = float(np.sqrt(np.sum(abs_obs**2) * np.sum(abs_prb**2))) + 1e-12
        overlap_support = float(np.clip(np.sum(abs_obs * abs_prb) / global_denom, 0.0, 1.0))

        # Active support mask of probe: evaluates truth of asserted `(Axis, Period)` bins without penalizing omitted axes
        mask = abs_prb > 1e-6
        active_denom = float(np.sqrt(np.sum(abs_obs[mask] ** 2) * np.sum(abs_prb[mask] ** 2))) + 1e-12
        raw_inter = np.real(z_obs * np.conj(z_prb))
        interference_map = (raw_inter / active_denom).astype(np.float32)
        resonance_score = float(np.clip(np.sum(raw_inter) / active_denom, -1.0, 1.0))

        # Per-bin directional conflict check where both observed & probe have meaningful energy
        bin_denom = abs_obs * abs_prb + 1e-12
        local_cos = np.where((abs_obs > 0.05) & (abs_prb > 0.05), raw_inter / bin_denom, 0.0)
        torn_coords: List[Tuple[str, str]] = []
        for i, ax_name in enumerate(COMMON_PRINCIPLE_AXES):
            for j, sc_name in enumerate(SCALE_BAND_LABELS):
                if float(local_cos[i, j]) < -0.25 and float(abs_obs[i, j] * abs_prb[i, j]) > 0.02:
                    torn_coords.append((ax_name, sc_name))

        if overlap_support < 0.15:
            verdict = "ORTHOGONAL_VOID"
            beam_torn = False
        elif resonance_score < -0.15 or (len(torn_coords) > 0 and resonance_score < 0.25):
            verdict = "DESTRUCTIVE_CONTRADICTION"
            beam_torn = True
        else:
            verdict = "CONSTRUCTIVE_RESONANCE"
            beam_torn = False

        beam = ConnectivityBeam(
            source_id=observed.entity_id,
            target_id=probe.entity_id,
            strength=max(0.1, overlap_support),
            rest_length=0.0,
            current_tension=float(max(0.0, 1.0 - resonance_score)),
            break_threshold=1.15,
            is_broken=beam_torn,
        )
        self.causal_field.beams.append(beam)

        return SpectralResonanceVerdict(
            source_id=observed.entity_id,
            target_id=probe.entity_id,
            resonance_score=resonance_score,
            overlap_support=overlap_support,
            axis_scale_interference=interference_map,
            torn_coordinates=tuple(torn_coords),
            verdict=verdict,
            beam_torn=beam_torn,
            latency_ms=float((time.perf_counter() - t0) * 1000.0),
        )

    @staticmethod
    def decode_spectral_array_to_korean(observed: SpectralRotorArray) -> str:
        """Zero-LLM inverse spectral readout straight from `(Amplitude, Period, Axis, Phase)`."""
        z = observed.complex_field()
        # Macro carrier (col 0) along flux_vs_anchor
        bg_val = float(np.real(z[0, 0]))
        # Meso/micro integrated directional polarity per axis (cols 1..3)
        dyn_axis = float(np.sum(np.real(z[0, 1:])))
        exp_axis = float(np.sum(np.real(z[1, 1:])))
        red_axis = float(np.sum(np.real(z[2, 1:])))
        yel_axis = float(np.sum(np.real(z[3, 1:])))

        bg_desc = "가만히 고정된 정적 배경(거시 주기 T=1.0)" if bg_val < -0.05 else "흐르는 거시 배경"
        motion_desc = "역동적으로 움직이며" if dyn_axis > 0.05 else "정지 상태를 유지하며"
        exp_desc = "바깥으로 활짝 피어나 확장하고" if exp_axis > 0.05 else "안으로 오므라들고"
        petal_desc = "붉은빛·주황 꽃잎" if red_axis > 0.05 else "푸르고 차가운 색"
        core_desc = "노란 중심부" if yel_axis > 0.05 else "어두운 중심"

        return (
            f"[{observed.entity_id} 스펙트럼 직독 해독] {bg_desc} 위에서 "
            f"{petal_desc}과 {core_desc}가 미시·중간 주기(T=0.5~0.125)로 "
            f"{motion_desc} {exp_desc} 있습니다."
        )

    @staticmethod
    def render_spectral_scale_dashboard(
        mp4_spec: SpectralRotorArray,
        wav_spec: SpectralRotorArray,
        verdicts: Sequence[Tuple[str, SpectralResonanceVerdict]],
        output_path: Path,
    ) -> Path:
        """Render clean dark-mode 2x2 visualization of `(Axis, Period)` spectral rotors & Korean interference."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
        fig.patch.set_facecolor("#111318")

        def _plot_signed_grid(ax: Any, arr: SpectralRotorArray, title: str) -> None:
            signed = (arr.amplitudes * np.cos(arr.phases)).astype(np.float32)
            im = ax.imshow(signed, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
            ax.set_xticks(np.arange(len(SCALE_BAND_LABELS)))
            ax.set_xticklabels(
                ["Macro\nT=1.0", "Meso\nT=0.5", "Micro1\nT=0.25", "Micro2\nT=0.125"],
                color="#e6edf3",
                fontsize=8,
            )
            ax.set_yticks(np.arange(len(COMMON_PRINCIPLE_AXES)))
            ax.set_yticklabels(COMMON_PRINCIPLE_AXES, color="#e6edf3", fontsize=8)
            ax.set_title(title, color="#e6edf3", fontsize=10, pad=8)
            for r in range(signed.shape[0]):
                for c in range(signed.shape[1]):
                    ax.text(
                        c,
                        r,
                        f"{signed[r, c]:+.2f}",
                        ha="center",
                        va="center",
                        color="#ffffff" if abs(signed[r, c]) >= 0.30 else "#0d1117",
                        fontsize=8,
                        fontweight="bold" if abs(signed[r, c]) >= 0.10 else "normal",
                    )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(colors="#c9d1d9", labelsize=7)

        _plot_signed_grid(
            axes[0, 0],
            mp4_spec,
            "Real MP4 Video (Axis x Period) Signed Spectral Array",
        )
        _plot_signed_grid(
            axes[0, 1],
            wav_spec,
            "Real WAV Speech (Axis x Period) Signed Spectral Array",
        )

        # Panel 3: Direct Korean Language Interference Scores & Tear Status
        ax3 = axes[1, 0]
        ax3.set_facecolor("#161b22")
        labels = [lbl for lbl, _ in verdicts]
        scores = [v.resonance_score for _, v in verdicts]
        colors = [
            "#7ee787"
            if v.verdict == "CONSTRUCTIVE_RESONANCE"
            else ("#ff7b72" if v.verdict == "DESTRUCTIVE_CONTRADICTION" else "#d2a8ff")
            for _, v in verdicts
        ]
        y_pos = np.arange(len(labels))
        ax3.barh(y_pos, scores, color=colors, height=0.55)
        ax3.axvline(0.0, color="#8b949e", linewidth=0.9)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels, color="#e6edf3", fontsize=8.5)
        ax3.invert_yaxis()
        ax3.set_xlim(-1.12, 1.12)
        ax3.set_xlabel("Direct Spectral Resonance Score (Cos Phase Alignment)", color="#c9d1d9", fontsize=9)
        ax3.set_title("Zero-LLM Korean Sentence Superposition Against MP4 Spectrum", color="#e6edf3", fontsize=10)
        ax3.tick_params(colors="#c9d1d9", labelsize=8)
        for idx, (s_val, (_, v)) in enumerate(zip(scores, verdicts)):
            tag = f"{s_val:+.2f} [{v.verdict[:12]}] ({v.latency_ms:.3f}ms)"
            if abs(s_val) >= 0.45:
                ax3.text(
                    0.5 * s_val,
                    idx,
                    tag,
                    va="center",
                    ha="center",
                    color="#0d1117",
                    fontsize=7.8,
                    fontweight="bold",
                )
            else:
                ax3.text(0.05, idx, tag, va="center", ha="left", color="#e6edf3", fontsize=7.8)

        # Panel 4: Truthful Mixed Sentence Interference Heatmap (Shows Macro Anchor + Micro Flux Coexistence)
        ax4 = axes[1, 1]
        truth_inter = verdicts[0][1].axis_scale_interference
        im4 = ax4.imshow(truth_inter, cmap="RdYlGn", vmin=-0.5, vmax=0.5, aspect="auto")
        ax4.set_xticks(np.arange(len(SCALE_BAND_LABELS)))
        ax4.set_xticklabels(
            ["Macro\nT=1.0", "Meso\nT=0.5", "Micro1\nT=0.25", "Micro2\nT=0.125"],
            color="#e6edf3",
            fontsize=8,
        )
        ax4.set_yticks(np.arange(len(COMMON_PRINCIPLE_AXES)))
        ax4.set_yticklabels(COMMON_PRINCIPLE_AXES, color="#e6edf3", fontsize=8)
        ax4.set_title("Constructive Interference Grid: Macro BG Anchor + Micro Red Bloom", color="#e6edf3", fontsize=9.5, pad=8)
        for r in range(truth_inter.shape[0]):
            for c in range(truth_inter.shape[1]):
                ax4.text(
                    c,
                    r,
                    f"{truth_inter[r, c]:+.2f}",
                    ha="center",
                    va="center",
                    color="#ffffff" if abs(truth_inter[r, c]) >= 0.35 else "#0d1117",
                    fontsize=8,
                    fontweight="bold" if abs(truth_inter[r, c]) >= 0.05 else "normal",
                )
        cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.03)
        cbar4.ax.tick_params(colors="#c9d1d9", labelsize=7)

        plt.tight_layout(pad=1.5)
        fig.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path
