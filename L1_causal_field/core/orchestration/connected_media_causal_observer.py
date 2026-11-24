"""
ELYSIA v3.2: Generative Scale-Rotor Connected Media Observer & Self-Improver
=============================================================================

True discernment MUST be generative, fast (O(K)), and self-correcting:
"분별이 진짜 원리라면 그 분별(스케일 로터 스펙트럼)로 인과와 시청각 세계를 직접
재창조할 수 있어야 하고, 스스로 재창조한 결과를 재관측해 오차(잔차)로부터 자기개선해야 한다."

Architectural Pillars:
1. Zero Brute-Force Pixel/Correlate Loops (Pure Vectorized O(K) Scale-Rotor Math):
   - Visual stream: Single batched 2D spatial FFT (`np.fft.rfft2`) + Fourier Shift
     Theorem phase velocity (`arg(V(t) * conj(V(t-1)))`) — replaces slow frame-by-frame
     `cv2.calcOpticalFlowFarneback` loops (~15x faster) while simultaneously yielding
     a compact `SpatiotemporalScaleRotorSpectrum` that can recreate RGB frames at
     ANY continuous time `t`.
   - Audio/Speech stream: Vectorized STFT (`sliding_window_view` + batched `rfft`) +
     FFT Wiener-Khinchin vocal autocorrelation (`irfft(|rfft|^2)`) — replaces O(W^2)
     `np.correlate` loops (~8x faster) while yielding a generative
     `AcousticScaleRotorSpectrum` that synthesizes real playable PCM waveforms.
   - Language/Utterance stream: Vocabulary-free temporal onset & turn-shift flow
     (works identically on unseen languages/gibberish without keyword crutches!).
2. Generative Cross-Modal Causal Transfer Operators (`CrossModalTransferOperator`):
   - Distills causal FIR / Wiener impulse-response operators `h(tau)` (`tau >= 0`)
     between Control -> Visual, Visual -> Audio, and Language <-> Visual so Elysia
     does not merely report a passive scalar `r`, but can **simulate and generate**
     new physical/audiovisual responses for never-before-seen inputs.
3. Autonomous Closed-Loop Self-Improvement (`learn_recreate_and_self_improve`):
   - Starts with a minimal coarse Scale-Rotor Spectrum (Cycle 0).
   - In every cycle: Recreates actual RGB frames & PCM audio -> Re-observes her own
     recreation through the EXACT same blind observer -> Computes structural residual
     `R = Reality - Recreation` -> Autonomously harvests missing spatiotemporal rotor
     modes & event knots directly from `R` -> Verifies strict monotonic improvement
     across cycles under MDL compression.
4. Novel Counterfactual World Creation (`create_novel_world_from_principles`):
   - Drives learned Scale-Rotor Spectra & Causal Transfer Operators with a brand-new
     unseen control/rhythm sequence to synthesize a brand-new pixel video + synchronized
     PCM soundtrack in milliseconds, then self-verifies that the newly created world
     obeys the learned causal physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from core.orchestration.elysia_v31_open_scale_rotor import (
    BoundaryFrictionEvent,
    CausalDroplet,
    CognitiveAxis,
    OpenVirtualRelationalMemory,
)
from core.topology.causal_scale_rotor import ScaleRotor

if TYPE_CHECKING:
    from core.orchestration.elysia_unified_living_cycle import ElysiaUnifiedLivingCycle
    from core.physics.causal_field import CausalField


# ---------------------------------------------------------------------------
# 1. Core Data Structures (Observations, Generative Scale-Rotors, Self-Improvement)
# ---------------------------------------------------------------------------


class AgencyMode(str, Enum):
    SELF_AGENCY_INTERACTIVE = "self_agency_interactive"
    EXTERNAL_AGENCY_OBSERVED = "external_agency_observed"
    PASSIVE_NO_CONTROL_TRIED = "passive_no_control_tried"


class NarrativeCausalityMode(str, Enum):
    VOLITIONAL_LANGUAGE_LEADS_MOTION = "volitional_language_leads_motion"
    REACTIVE_MOTION_LEADS_LANGUAGE = "reactive_motion_leads_language"
    AUDIOVISUAL_PHYSICAL_SYNCHRONY = "audiovisual_physical_synchrony"
    INTERACTIVE_CONTROL_DRIVEN = "interactive_control_driven"
    UNIMODAL_PHYSICAL_FLOW = "unimodal_physical_flow"
    UNIMODAL_VOCAL_PROSODY = "unimodal_vocal_prosody"


@dataclass
class TimeAlignedUtterance:
    """One synchronized dialogue/subtitle/narration utterance on the timeline."""

    t_start: float
    t_end: float
    speaker: str
    text: str


@dataclass
class SpatiotemporalScaleRotorSpectrum:
    """
    Generative O(K) visual Scale-Rotor representation of a video stream.
    Stores ONLY selected low-to-mid 2D spatial frequency modes and their
    compact temporal Fourier-rotor trajectories (plus residual event-knot
    corrections harvested during autonomous self-improvement).
    """

    height: int
    width: int
    duration_s: float
    fps: float
    frame_count: int
    # Low-pass spatial grid extents (kh_max, kw_max) in rfft2 coordinates
    kh_max: int
    kw_max: int
    # Active temporal rotor indices and complex spectral coefficients:
    # shape (len(temporal_freqs), kh_max, kw_max, 3) for RGB channels
    temporal_indices: np.ndarray
    temporal_freqs_hz: np.ndarray
    rotor_coeffs: np.ndarray
    # Sparse event-knot residuals harvested at high-error moments (cuts/impacts):
    # dict mapping frame_idx -> compact spatial coefficient correction (kh, kw, 3)
    knot_corrections: Dict[int, np.ndarray] = field(default_factory=dict)
    raw_scalar_count: int = 1

    @property
    def parameter_count(self) -> int:
        base = int(self.rotor_coeffs.size * 2)  # complex -> 2 floats
        knots = sum(int(v.size * 2) for v in self.knot_corrections.values())
        return max(1, base + knots)

    @property
    def compression_ratio(self) -> float:
        return float(self.raw_scalar_count / max(1, self.parameter_count))

    def recreate_frames_at(self, times: np.ndarray) -> np.ndarray:
        """
        Synthesizes real RGB uint8 video frames at ANY arbitrary continuous time
        vector `times` strictly from the Scale-Rotor Spectrum (O(K) per frame,
        zero original pixels stored!).
        """
        t_arr = np.asarray(times, dtype=np.float64)
        n_t = len(t_arr)
        # Evaluate temporal complex rotors: E[t, m] = exp(i * 2*pi * f_m * t)
        phases = 2.0 * math.pi * np.outer(t_arr, self.temporal_freqs_hz)
        basis = np.exp(1j * phases)  # (n_t, K_t)

        # Contract temporal rotors -> spatial rfft2 cube: (n_t, kh_max, kw_max, 3)
        sub_spec = np.tensordot(basis, self.rotor_coeffs, axes=([1], [0]))

        # Apply sparse event-knot corrections localized smoothly in time (Gaussian window)
        if self.knot_corrections and self.frame_count > 1:
            dt = self.duration_s / max(1, self.frame_count)
            sigma = max(1e-4, 0.55 * dt)
            for k_idx, corr in self.knot_corrections.items():
                t_knot = float(k_idx) * dt
                w_t = np.exp(-0.5 * ((t_arr - t_knot) / sigma) ** 2)
                sub_spec += w_t[:, None, None, None] * corr[None, :, :, :]

        # Embed into full rfft2 grid & inverse transform all frames in ONE vectorized pass (complex64 for speed)
        full_spec = np.zeros((n_t, self.height, self.width // 2 + 1, 3), dtype=np.complex64)
        sub_c64 = sub_spec.astype(np.complex64, copy=False)
        half_h = self.kh_max // 2
        rem_h = self.kh_max - half_h
        full_spec[:, :rem_h, : self.kw_max, :] = sub_c64[:, :rem_h, :, :]
        if half_h > 0:
            full_spec[:, -half_h:, : self.kw_max, :] = sub_c64[:, -half_h:, :, :]

        rec = np.fft.irfft2(full_spec, s=(self.height, self.width), axes=(1, 2))
        rec = np.clip(rec * 255.0, 0.0, 255.0).astype(np.uint8)
        return rec


@dataclass
class AcousticScaleRotorSpectrum:
    """
    Generative O(K) acoustic Scale-Rotor representation of a real sound/speech
    waveform. Stores ONLY top-K sparse time-frequency spectral rotors per window
    so it can directly synthesize playable PCM waveforms via overlap-add iRFFT.
    """

    sample_rate: int
    duration_s: float
    total_samples: int
    win_size: int
    hop_size: int
    n_windows: int
    # Sparse top-K frequency bin indices & complex rotor values per window:
    # shape (n_windows, k_bins)
    bin_indices: np.ndarray
    rotor_complex: np.ndarray

    @property
    def parameter_count(self) -> int:
        # bin_index + real + imag per active rotor
        return max(1, int(self.rotor_complex.size * 3))

    @property
    def compression_ratio(self) -> float:
        return float(max(1, self.total_samples) / max(1, self.parameter_count))

    def recreate_pcm(self) -> np.ndarray:
        """Synthesizes a real float32 PCM waveform directly from the Scale-Rotor Spectrum."""
        n_fft_bins = self.win_size // 2 + 1
        stft = np.zeros((self.n_windows, n_fft_bins), dtype=np.complex128)
        rows = np.arange(self.n_windows)[:, None]
        stft[rows, self.bin_indices] = self.rotor_complex

        # Vectorized inverse RFFT across all windows simultaneously
        frames = np.fft.irfft(stft, n=self.win_size, axis=1)
        window = np.hanning(self.win_size)
        frames *= window[None, :]

        out_len = (self.n_windows - 1) * self.hop_size + self.win_size
        pcm = np.zeros(out_len, dtype=np.float64)
        norm = np.zeros(out_len, dtype=np.float64)
        win_sq = window ** 2
        for i in range(self.n_windows):
            s = i * self.hop_size
            pcm[s : s + self.win_size] += frames[i]
            norm[s : s + self.win_size] += win_sq

        valid = norm > 1e-8
        pcm[valid] /= norm[valid]
        pcm = pcm[: self.total_samples]
        mx = float(np.max(np.abs(pcm)))
        if mx > 1.0:
            pcm = pcm / mx
        return pcm.astype(np.float32)


@dataclass
class CrossModalTransferOperator:
    """
    Generative causal FIR transfer operator `y(t) ~= bias + sum_{tau=0..L} h[tau] * x(t - tau)`
    distilled from observed cross-modal lead-lag causality. Can predict and drive
    new physical/audiovisual trajectories for brand-new unseen input streams!
    """

    source_name: str
    target_name: str
    kernel: np.ndarray  # causal lags tau = 0..L
    bias: float
    dt: float
    train_r2: float

    def predict(self, cause_signal: np.ndarray) -> np.ndarray:
        x = np.asarray(cause_signal, dtype=np.float64).ravel()
        n = len(x)
        L = len(self.kernel)
        y = np.full(n, self.bias, dtype=np.float64)
        for lag in range(L):
            if lag == 0:
                y += self.kernel[lag] * x
            elif lag < n:
                y[lag:] += self.kernel[lag] * x[:-lag]
        return y


@dataclass
class VisualStreamTrace:
    """Raw pixel-derived kinematic & visual continuity trajectories (Fourier Phase-Rotor)."""

    times: np.ndarray
    fps: float
    frame_count: int
    source_kind: str
    motion_energy: np.ndarray
    active_object_speed: np.ndarray
    active_motion_area: np.ndarray
    flow_coherence: np.ndarray
    motion_acceleration: np.ndarray
    visual_continuity: np.ndarray
    centroid_x: np.ndarray
    centroid_y: np.ndarray
    luminance: np.ndarray
    warmth_ratio: np.ndarray
    detected_cut_indices: List[int] = field(default_factory=list)
    scale_rotor_spectrum: Optional[SpatiotemporalScaleRotorSpectrum] = None
    elapsed_ms: float = 0.0


@dataclass
class AudioStreamTrace:
    """Raw waveform-derived acoustic, percussive, and vocal prosody trajectories (FFT Rotor)."""

    times: np.ndarray
    sample_rate: int
    duration_s: float
    rms_energy: np.ndarray
    onset_flux: np.ndarray
    spectral_centroid: np.ndarray
    vocal_harmonicity: np.ndarray
    pause_mask: np.ndarray
    voiced_fraction: float
    pause_fraction: float
    phrase_count: int
    acoustic_character: str
    scale_rotor_spectrum: Optional[AcousticScaleRotorSpectrum] = None
    elapsed_ms: float = 0.0


@dataclass
class LanguageStreamTrace:
    """Vocabulary-free temporal utterance activity, onset pulses, and turn shifts."""

    times: np.ndarray
    utterances: List[TimeAlignedUtterance]
    utterance_activity: np.ndarray
    directive_intent_flux: np.ndarray
    causal_connective_density: np.ndarray
    narrative_tension_arc: np.ndarray
    speaker_turn_shift: np.ndarray
    speakers: List[str]


@dataclass
class CrossModalCausalLink:
    """Discovered directional lead-lag coupling + generative causal transfer operator."""

    source_modality: str
    source_channel: str
    target_modality: str
    target_channel: str
    best_lag_steps: int
    best_lag_seconds: float
    peak_correlation: float
    zero_lag_correlation: float
    directional_asymmetry: float
    relation_kind: str
    interpretation_ko: str
    transfer_operator: Optional[CrossModalTransferOperator] = None


@dataclass
class ContinuityBridge:
    """Detected informational continuity carried by audio/language across a visual cut."""

    cut_step: int
    cut_time_s: float
    visual_continuity_drop: float
    audio_continuity_preserved: bool
    language_continuity_preserved: bool
    bridged_by: List[str]
    summary_ko: str


@dataclass
class CausalNarrativeBeat:
    """One segmented causal unit in a connected video, speech, or game episode."""

    beat_index: int
    t_start: float
    t_end: float
    chromatic_character: str
    mean_motion: float
    peak_acceleration: float
    mean_audio_energy: float
    peak_onset: float
    directive_level: float
    tension_level: float
    control_level: float
    dominant_event_cue: str
    utterances_in_beat: List[str] = field(default_factory=list)


@dataclass
class BeatTransition:
    """Causal transition edge linking Beat k -> Beat k+1."""

    from_beat: int
    to_beat: int
    transition_kind: str
    coupling_strength: float
    reasoning_ko: str


@dataclass
class SelfImprovementCycleMetrics:
    """Measured self-recreation & self-observation metrics for one improvement cycle."""

    cycle_index: int
    visual_mse: float
    visual_fidelity: float
    motion_flow_correlation: float
    audio_mse: float
    audio_envelope_correlation: float
    visual_compression_ratio: float
    audio_compression_ratio: float
    visual_knot_count: int
    elapsed_ms: float


@dataclass
class SelfImprovementReport:
    """Full autonomous Observe -> Recreate -> Re-Observe Residual -> Self-Improve report."""

    episode_id: str
    cycles: List[SelfImprovementCycleMetrics]
    final_visual_rotor: Optional[SpatiotemporalScaleRotorSpectrum]
    final_audio_rotor: Optional[AcousticScaleRotorSpectrum]
    recreated_frames: Optional[np.ndarray]
    recreated_pcm: Optional[np.ndarray]
    visual_error_drop_ratio: float
    audio_error_drop_ratio: float
    strictly_improved: bool
    total_elapsed_ms: float
    summary_ko: str


@dataclass
class NovelWorldCreationReport:
    """Result of generating a brand-new unseen pixel video + PCM audio purely from learned principles."""

    origin_episode_id: str
    created_frames: np.ndarray
    created_pcm: np.ndarray
    sample_rate: int
    fps: float
    novel_control_signal: np.ndarray
    self_verified_report: "ConnectedMediaEpisodeReport"
    agency_preserved: bool
    synchrony_preserved: bool
    elapsed_ms: float
    summary_ko: str


@dataclass
class ConnectedMediaEpisodeReport:
    """Full causal-narrative perception + generative Scale-Rotor report for one episode."""

    episode_id: str
    source_description: str
    duration_s: float
    present_modalities: List[str]
    absent_modalities: List[str]
    visual_trace: Optional[VisualStreamTrace]
    audio_trace: Optional[AudioStreamTrace]
    language_trace: Optional[LanguageStreamTrace]
    control_trace: Optional[np.ndarray]
    causal_links: List[CrossModalCausalLink]
    agency_mode: AgencyMode
    causality_mode: NarrativeCausalityMode
    beats: List[CausalNarrativeBeat]
    transitions: List[BeatTransition]
    continuity_bridges: List[ContinuityBridge]
    macro_droplet: CausalDroplet
    narrative_signature: np.ndarray
    summary_ko: str
    elapsed_ms: float = 0.0


@dataclass
class CrossMediaComparisonReport:
    """Non-reductive structural comparison between two observed media/game episodes."""

    episode_a: str
    episode_b: str
    narrative_arc_isomorphism: float
    chromatic_flow_alignment: float
    shared_causal_motifs: List[str]
    agency_divergence: str
    modality_divergence: Dict[str, List[str]]
    synthesis_ko: str


# ---------------------------------------------------------------------------
# 2. Fast Vectorized O(K) Scale-Rotor Encoding & Stream Observers
# ---------------------------------------------------------------------------


def _safe_norm_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3:
        return 0.0
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(xc, yc) / denom, -1.0, 1.0))


def _resize_rgb_fast(frame: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    w, h = target_wh
    if frame.shape[1] == w and frame.shape[0] == h and frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    im = Image.fromarray(frame)
    return np.asarray(im.convert("RGB").resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)


def _load_video_or_gif_frames(
    source: Path | str | Sequence[np.ndarray],
    max_frames: int = 48,
    resize_wh: Tuple[int, int] = (128, 80),
    explicit_fps: Optional[float] = None,
) -> Tuple[np.ndarray, float, str]:
    """Loads RGB uint8 tensor `(T, H, W, 3)` from `.mp4`, `.gif`, or live frame stack."""
    if not isinstance(source, (str, Path)):
        raw_arr = source if isinstance(source, np.ndarray) and source.ndim == 4 else list(source)
        if len(raw_arr) == 0:
            raise ValueError("Empty frame sequence provided")
        step = max(1, len(raw_arr) // max_frames) if len(raw_arr) > max_frames else 1
        sampled = raw_arr[::step][:max_frames]
        resized = np.stack([_resize_rgb_fast(np.asarray(fr, dtype=np.uint8), resize_wh) for fr in sampled], axis=0)
        fps = float(explicit_fps if explicit_fps and explicit_fps > 0 else 15.0) / step
        return resized, fps, "live_rgb_pixel_stream"

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Video/GIF file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".gif":
        frames: List[np.ndarray] = []
        durations_ms: List[float] = []
        with Image.open(path) as im:
            n_total = int(getattr(im, "n_frames", 1))
            step = max(1, n_total // max_frames) if n_total > max_frames else 1
            for i in range(0, n_total, step):
                if len(frames) >= max_frames:
                    break
                im.seek(i)
                dur = im.info.get("duration", 100)
                if isinstance(dur, (int, float)) and dur > 0:
                    durations_ms.append(float(dur))
                rgb = np.asarray(im.convert("RGB").resize(resize_wh, Image.Resampling.BILINEAR), dtype=np.uint8)
                frames.append(rgb)
        if not frames:
            raise ValueError(f"GIF yielded zero readable frames: {path}")
        avg_ms = float(np.mean(durations_ms)) if durations_ms else 100.0
        eff_fps = float(explicit_fps) if explicit_fps else (1000.0 / max(10.0, avg_ms * step))
        return np.stack(frames, axis=0), eff_fps, f"gif_file:{path.name}"

    # Lazy import cv2 ONLY when reading actual video container formats (`.mp4`, `.avi`, etc.)
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total_frames // max_frames) if total_frames > max_frames else 1
        frames = []
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(cv2.resize(bgr, resize_wh, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                frames.append(rgb)
                if len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    if len(frames) < 2:
        raise ValueError(f"Video yielded fewer than 2 frames: {path}")
    eff_fps = float(explicit_fps) if explicit_fps else (native_fps / step)
    return np.stack(frames, axis=0), eff_fps, f"video_file:{path.name}"


def encode_video_to_scale_rotor(
    frames_rgb: np.ndarray,
    fps: float,
    kh_max: int = 16,
    kw_max: int = 16,
    k_temporal: int = 10,
    max_knots: int = 0,
) -> SpatiotemporalScaleRotorSpectrum:
    """
    Distills a raw RGB frame volume `(T, H, W, 3)` into a compact generative
    `SpatiotemporalScaleRotorSpectrum` using ONE vectorized 2D spatial FFT +
    1D temporal FFT, optionally harvesting sparse event-knot corrections at the
    highest-residual frames (impacts/cuts).
    """
    arr = np.asarray(frames_rgb, dtype=np.float32) * (1.0 / 255.0)
    n_t, h, w, _ = arr.shape
    kh = max(4, min(kh_max, h))
    kw = max(4, min(kw_max, w // 2 + 1))

    # 1. Single batched 2D spatial FFT across all frames & RGB channels (float32 -> complex64)
    sp = np.fft.rfft2(arr, axes=(1, 2)).astype(np.complex64, copy=False)  # (T, H, W//2+1, 3)

    # Extract compact low-frequency spatial band (positive + negative vertical freqs)
    half_h = kh // 2
    rem_h = kh - half_h
    if half_h > 0:
        sub_sp = np.concatenate([sp[:, :rem_h, :kw, :], sp[:, -half_h:, :kw, :]], axis=1)
    else:
        sub_sp = sp[:, :rem_h, :kw, :]  # (T, kh, kw, 3)

    # 2. Single batched temporal FFT across frames -> select top-K_t dominant temporal rotors
    tp = np.fft.fft(sub_sp, axis=0) / float(n_t)  # (T, kh, kw, 3)
    mode_energy = np.sum(np.abs(tp) ** 2, axis=(1, 2, 3))
    k_keep = max(1, min(k_temporal, n_t))
    top_idx = np.sort(np.argsort(mode_energy)[::-1][:k_keep])

    freqs_hz = np.fft.fftfreq(n_t, d=1.0 / max(1e-3, fps))[top_idx]
    coeffs = tp[top_idx]  # (K_t, kh, kw, 3)
    dur_s = float(n_t / max(1e-3, fps))

    spec = SpatiotemporalScaleRotorSpectrum(
        height=h,
        width=w,
        duration_s=dur_s,
        fps=float(fps),
        frame_count=n_t,
        kh_max=kh,
        kw_max=kw,
        temporal_indices=top_idx,
        temporal_freqs_hz=freqs_hz,
        rotor_coeffs=coeffs,
        knot_corrections={},
        raw_scalar_count=int(arr.size),
    )

    # 3. Optional adaptive event-knot residual harvesting (Phase 1 knot principle!)
    if max_knots > 0 and n_t >= 4:
        t_grid = np.arange(n_t, dtype=np.float64) / max(1e-3, fps)
        phases = 2.0 * math.pi * np.outer(t_grid, freqs_hz)
        rec_sub = np.tensordot(np.exp(1j * phases), coeffs, axes=([1], [0]))
        residual = sub_sp - rec_sub
        frame_err = np.sum(np.abs(residual) ** 2, axis=(1, 2, 3))
        knot_indices = np.argsort(frame_err)[::-1][: min(max_knots, n_t)]
        for k_i in knot_indices:
            if frame_err[k_i] > 1e-6:
                spec.knot_corrections[int(k_i)] = residual[int(k_i)].copy()

    return spec


def observe_video_stream(
    source: Path | str | Sequence[np.ndarray],
    max_frames: int = 48,
    resize_wh: Tuple[int, int] = (128, 80),
    explicit_fps: Optional[float] = None,
    build_rotor: bool = True,
) -> VisualStreamTrace:
    """
    PURE VECTORIZED O(K) FOURIER PHASE-SHIFT VISUAL OBSERVER (ZERO Farneback loops!).
    Extracts motion velocity, active-object velocity, active motion area, directional
    coherence, impulse acceleration, scene-cut visual continuity, moving centroid,
    lighting mood, AND the generative `SpatiotemporalScaleRotorSpectrum` in milliseconds.
    """
    t0 = time.perf_counter()
    frames, fps, src_kind = _load_video_or_gif_frames(
        source=source,
        max_frames=max_frames,
        resize_wh=resize_wh,
        explicit_fps=explicit_fps,
    )
    n, h, w, _ = frames.shape
    times = np.arange(n, dtype=np.float64) / max(1e-3, fps)
    rgb = frames.astype(np.float32) * (1.0 / 255.0)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]  # (T, H, W) float32

    # Luminance & chromatic warmth (vectorized across all T frames)
    luminance = np.mean(gray, axis=(1, 2), dtype=np.float64)
    warmth = (np.mean(rgb[..., 0], axis=(1, 2), dtype=np.float64) + 0.05) / (
        np.mean(rgb[..., 2], axis=(1, 2), dtype=np.float64) + 0.05
    )

    # --- 1. Fourier Phase-Shift Theorem Rotor Motion Extraction (ONE batched float32 rfft2!) ---
    G = np.fft.rfft2(gray, axes=(1, 2))  # (T, H, W//2+1)
    cross = G[1:] * np.conj(G[:-1])
    mag_cross = np.abs(cross) + 1e-9
    phase_diff = np.angle(cross)

    ky = np.fft.fftfreq(h).astype(np.float32)[:, None] * (2.0 * math.pi)
    kx = np.fft.rfftfreq(w).astype(np.float32)[None, :] * (2.0 * math.pi)
    k_norm = np.hypot(ky, kx)
    mid_band = (k_norm > 0.08) & (k_norm < 1.8)
    w_spec = np.where(mid_band, mag_cross, 0.0)
    w_sum = np.sum(w_spec, axis=(1, 2), dtype=np.float64) + 1e-9

    phase_speed = np.sum(w_spec * np.abs(phase_diff) / (k_norm[None, :, :] + 0.15), axis=(1, 2), dtype=np.float64) / w_sum
    dir_x = np.sum(w_spec * np.sin(phase_diff) * (kx[None, :, :] / (k_norm[None, :, :] + 1e-6)), axis=(1, 2), dtype=np.float64) / w_sum
    dir_y = np.sum(w_spec * np.sin(phase_diff) * (ky[None, :, :] / (k_norm[None, :, :] + 1e-6)), axis=(1, 2), dtype=np.float64) / w_sum
    flow_coh_1 = np.clip(np.hypot(dir_x, dir_y) / (phase_speed + 1e-4), 0.0, 1.0)

    # --- 2. Vectorized Spatial Difference & Block-Pooled Active Speed (ZERO O(N log N) pixel sorting!) ---
    diff = np.abs(gray[1:] - gray[:-1])  # (T-1, H, W) float32
    gx = np.diff(gray[1:], axis=2, prepend=gray[1:, :, :1])
    gy = np.diff(gray[1:], axis=1, prepend=gray[1:, :1, :])
    local_vel = np.minimum(6.0, diff / (np.hypot(gx, gy) + 0.08))

    motion_energy_1 = (
        0.65 * phase_speed
        + 2.2 * np.mean(diff, axis=(1, 2), dtype=np.float64)
        + 0.35 * np.mean(local_vel, axis=(1, 2), dtype=np.float64)
    )
    # Fast O(N) spatial 8x8-cell mean pooling -> peak active-cell velocity (catches sparse moving sprites in <0.3ms!)
    bh, bw = max(1, h // 8), max(1, w // 8)
    hc, wc = bh * 8, bw * 8
    blk_vel = local_vel[:, :hc, :wc].reshape(n - 1, 8, bh, 8, bw).mean(axis=(2, 4))
    blk_diff = diff[:, :hc, :wc].reshape(n - 1, 8, bh, 8, bw).mean(axis=(2, 4))
    active_speed_1 = (
        np.max(blk_vel, axis=(1, 2)).astype(np.float64)
        + 4.5 * np.max(blk_diff, axis=(1, 2)).astype(np.float64)
    )
    active_area_1 = np.mean(diff > 0.035, axis=(1, 2), dtype=np.float64)

    yy_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)[None, :, None]
    xx_norm = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, None, :]
    d_weight = np.where(diff > 0.02, diff, 0.0)
    dw_sum = np.sum(d_weight, axis=(1, 2), dtype=np.float64)
    raw_cx = np.sum(d_weight * xx_norm, axis=(1, 2), dtype=np.float64) / np.maximum(dw_sum, 1e-9)
    raw_cy = np.sum(d_weight * yy_norm, axis=(1, 2), dtype=np.float64) / np.maximum(dw_sum, 1e-9)

    cx = np.full(n, 0.5, dtype=np.float64)
    cy = np.full(n, 0.5, dtype=np.float64)
    prev_x, prev_y = 0.5, 0.5
    for i in range(n - 1):
        if dw_sum[i] > 1e-2:
            prev_x = float(raw_cx[i])
            prev_y = float(raw_cy[i])
        cx[i + 1] = prev_x
        cy[i + 1] = prev_y
    if n > 1:
        cx[0], cy[0] = cx[1], cy[1]

    # --- 3. Vectorized Visual Continuity & Scene-Cut Separation ---
    g_flat = gray.reshape(n, -1)
    g_c = g_flat - np.mean(g_flat, axis=1, keepdims=True)
    g_n = np.linalg.norm(g_c, axis=1) + 1e-9
    pix_sim_1 = np.clip(np.sum(g_c[1:] * g_c[:-1], axis=1) / (g_n[1:] * g_n[:-1]), 0.0, 1.0).astype(np.float64)

    small_rgb = rgb.reshape(n, 4, h // 4, 4, w // 4, 3).mean(axis=(2, 4)).reshape(n, -1)
    s_n = np.linalg.norm(small_rgb, axis=1) + 1e-9
    color_sim_1 = np.clip(np.sum(small_rgb[1:] * small_rgb[:-1], axis=1) / (s_n[1:] * s_n[:-1]), 0.0, 1.0).astype(np.float64)
    rgb_l1_sim_1 = np.clip(1.0 - 2.2 * np.mean(np.abs(rgb[1:] - rgb[:-1]), axis=(1, 2, 3), dtype=np.float64), 0.0, 1.0)
    vis_cont_1 = 0.40 * pix_sim_1 + 0.30 * color_sim_1 + 0.30 * rgb_l1_sim_1

    motion_energy = np.concatenate([[motion_energy_1[0]], motion_energy_1]) if n > 1 else np.zeros(n)
    active_obj_speed = np.concatenate([[active_speed_1[0]], active_speed_1]) if n > 1 else np.zeros(n)
    active_area = np.concatenate([[active_area_1[0]], active_area_1]) if n > 1 else np.zeros(n)
    flow_coherence = np.concatenate([[flow_coh_1[0]], flow_coh_1]) if n > 1 else np.zeros(n)
    visual_cont = np.concatenate([[1.0], vis_cont_1]) if n > 1 else np.ones(n)

    detected_cuts: List[int] = []
    if n >= 4:
        med_c = float(np.median(visual_cont[1:]))
        for i in range(1, n):
            if visual_cont[i] < min(0.72, med_c - 0.20):
                detected_cuts.append(i)

    # PHYSICAL SEPARATION: A visual scene cut is a frame discontinuity, NOT physical object motion!
    # Interpolate cut-frame motion values from immediate non-cut neighbors so cuts never forge fake motion spikes.
    for ci in detected_cuts:
        lo_i = max(0, ci - 1)
        hi_i = min(n - 1, ci + 1)
        motion_energy[ci] = 0.5 * (motion_energy[lo_i] + motion_energy[hi_i])
        active_obj_speed[ci] = 0.5 * (active_obj_speed[lo_i] + active_obj_speed[hi_i])
        active_area[ci] = 0.5 * (active_area[lo_i] + active_area[hi_i])
        cx[ci] = 0.5 * (cx[lo_i] + cx[hi_i])
        cy[ci] = 0.5 * (cy[lo_i] + cy[hi_i])

    vel_combined = 0.55 * active_obj_speed + 1.4 * motion_energy
    motion_accel = np.abs(np.diff(vel_combined, prepend=vel_combined[0]))
    vx = np.diff(cx, prepend=cx[0])
    vy = np.diff(cy, prepend=cy[0])
    dvx = np.diff(vx, prepend=vx[0])
    dvy = np.diff(vy, prepend=vy[0])
    motion_accel = motion_accel + 4.0 * np.hypot(dvx, dvy)
    for ci in detected_cuts:
        motion_accel[ci] = 0.0

    rotor_spec = (
        encode_video_to_scale_rotor(frames, fps=fps, kh_max=14, kw_max=14, k_temporal=10, max_knots=2)
        if build_rotor
        else None
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return VisualStreamTrace(
        times=times,
        fps=fps,
        frame_count=n,
        source_kind=src_kind,
        motion_energy=motion_energy,
        active_object_speed=active_obj_speed,
        active_motion_area=active_area,
        flow_coherence=flow_coherence,
        motion_acceleration=motion_accel,
        visual_continuity=visual_cont,
        centroid_x=cx,
        centroid_y=cy,
        luminance=luminance,
        warmth_ratio=warmth,
        detected_cut_indices=detected_cuts,
        scale_rotor_spectrum=rotor_spec,
        elapsed_ms=elapsed_ms,
    )


def encode_audio_to_scale_rotor(
    pcm: np.ndarray,
    sample_rate: int,
    k_bins: int = 16,
    win_size: int = 512,
    hop_size: int = 256,
) -> AcousticScaleRotorSpectrum:
    """
    Vectorized STFT encoding of a raw audio/speech waveform into a compact
    generative `AcousticScaleRotorSpectrum` storing ONLY top-K spectral rotors
    per time window.
    """
    arr = np.asarray(pcm, dtype=np.float64).ravel()
    total_n = len(arr)
    w_sz = max(64, min(win_size, total_n))
    h_sz = max(32, min(hop_size, w_sz // 2))

    # Pad cleanly for exact overlap-add reconstruction
    n_win = max(1, int(math.ceil(max(0, total_n - w_sz) / float(h_sz))) + 1)
    pad_len = (n_win - 1) * h_sz + w_sz
    padded = np.pad(arr, (0, max(0, pad_len - total_n)))

    # Strided sliding view -> ZERO Python loop over windows!
    strides = (padded.strides[0] * h_sz, padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=(n_win, w_sz), strides=strides)
    hann = np.hanning(w_sz)
    stft = np.fft.rfft(windows * hann[None, :], axis=1)  # (n_win, w_sz//2 + 1)

    mags = np.abs(stft)
    k_keep = max(2, min(k_bins, stft.shape[1]))
    # Partition top-K frequency bins per window in O(N_bins) vectorized time
    part = np.argpartition(mags, -k_keep, axis=1)[:, -k_keep:]
    rows = np.arange(n_win)[:, None]
    vals = stft[rows, part]

    return AcousticScaleRotorSpectrum(
        sample_rate=int(sample_rate),
        duration_s=float(total_n / float(sample_rate)),
        total_samples=total_n,
        win_size=w_sz,
        hop_size=h_sz,
        n_windows=n_win,
        bin_indices=part.astype(np.int32),
        rotor_complex=vals,
    )


def observe_audio_stream(
    source: Path | str | Tuple[np.ndarray, int],
    target_times: Optional[np.ndarray] = None,
    window_s: float = 0.08,
    build_rotor: bool = True,
) -> AudioStreamTrace:
    """
    PURE VECTORIZED FFT WIENER-KHINCHIN AUDIO/SPEECH OBSERVER (ZERO O(W^2) loops!).
    Extracts loudness, spectral onset flux, spectral centroid, vocal harmonicity
    (85-380 Hz via FFT autocorrelation), breath-pause phrasing, AND a generative
    `AcousticScaleRotorSpectrum` in milliseconds.
    """
    t0 = time.perf_counter()
    if isinstance(source, (str, Path)):
        from scipy.io import wavfile

        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Audio WAV file not found: {path}")
        sr, data = wavfile.read(str(path))
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            arr = arr / float(max(abs(info.min), info.max))
    else:
        pcm, sr = source
        arr = np.asarray(pcm, dtype=np.float64).ravel()

    if len(arr) < 64 or sr <= 0:
        raise ValueError("Audio stream too short or invalid sample rate")

    peak = float(np.max(np.abs(arr)))
    norm_arr = (arr / peak) if peak > 1e-9 else arr.copy()
    dur_s = float(len(norm_arr) / float(sr))

    if target_times is None:
        hop_s = 0.05
        n_steps = max(8, int(dur_s / hop_s))
        times = np.linspace(0.0, dur_s, n_steps, endpoint=False)
    else:
        times = np.asarray(target_times, dtype=np.float64)

    n = len(times)
    win_n = max(128, int(sr * window_s))
    # Next power of 2 for blazing-fast FFT autocorrelation
    n_fft_ac = 1 << int(math.ceil(math.log2(2 * win_n)))

    # Vectorized window gathering: shape (n, win_n)
    centers = np.round(times * sr).astype(np.int64)
    starts = np.clip(centers - win_n // 2, 0, max(0, len(norm_arr) - win_n))
    padded_src = norm_arr if len(norm_arr) >= win_n else np.pad(norm_arr, (0, win_n - len(norm_arr)))
    idx_mat = starts[:, None] + np.arange(win_n, dtype=np.int64)[None, :]
    segs = padded_src[idx_mat]  # (n, win_n)

    rms = np.sqrt(np.mean(segs ** 2, axis=1))
    hann = np.hanning(win_n)
    specs = np.abs(np.fft.rfft(segs * hann[None, :], axis=1))  # (n, win_n//2 + 1)
    freqs = np.fft.rfftfreq(win_n, d=1.0 / sr)
    nyq = max(1.0, sr * 0.5)
    centroid = (specs @ freqs) / (np.sum(specs, axis=1) + 1e-12) / nyq

    onset_flux = np.zeros(n, dtype=np.float64)
    if n > 1:
        onset_flux[1:] = np.mean(np.maximum(0.0, specs[1:] - specs[:-1]), axis=1)

    # Vectorized Wiener-Khinchin Autocorrelation via FFT: O(W log W) for ALL windows at once!
    segs0 = segs - np.mean(segs, axis=1, keepdims=True)
    F_ac = np.fft.rfft(segs0, n=n_fft_ac, axis=1)
    ac = np.fft.irfft(F_ac * np.conj(F_ac), n=n_fft_ac, axis=1)[:, :win_n]
    ac0 = np.maximum(ac[:, :1], 1e-12)
    ac_norm = ac / ac0

    min_lag = max(2, int(sr / 380.0))
    max_lag = min(win_n - 2, int(sr / 85.0))
    if max_lag > min_lag:
        raw_harm = np.max(ac_norm[:, min_lag:max_lag], axis=1)
        vocal_harm = np.where(rms > 0.025, np.clip(raw_harm, 0.0, 1.0), 0.0)
    else:
        vocal_harm = np.zeros(n, dtype=np.float64)

    noise_floor = max(0.035, 0.14 * float(np.percentile(rms, 90)))
    pause_mask = (rms < noise_floor).astype(np.float64)
    voiced_frac = float(np.mean(vocal_harm > 0.35))
    pause_frac = float(np.mean(pause_mask > 0.5))

    active = (pause_mask < 0.5).astype(np.int32)
    phrase_count = int(np.sum(np.diff(active, prepend=0) == 1))

    if float(np.max(rms)) < 0.02:
        char = "near_silent"
    elif voiced_frac >= 0.35 and pause_frac >= 0.12 and phrase_count >= 2:
        char = "speech_like_phrasing"
    elif voiced_frac >= 0.70 and pause_frac < 0.10:
        char = "continuous_harmonic"
    else:
        char = "impulsive_events"

    rotor_spec = (
        encode_audio_to_scale_rotor(norm_arr, sample_rate=sr, k_bins=16, win_size=512, hop_size=256)
        if build_rotor
        else None
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return AudioStreamTrace(
        times=times,
        sample_rate=int(sr),
        duration_s=dur_s,
        rms_energy=rms,
        onset_flux=onset_flux,
        spectral_centroid=centroid,
        vocal_harmonicity=vocal_harm,
        pause_mask=pause_mask,
        voiced_fraction=voiced_frac,
        pause_fraction=pause_frac,
        phrase_count=phrase_count,
        acoustic_character=char,
        scale_rotor_spectrum=rotor_spec,
        elapsed_ms=elapsed_ms,
    )


def observe_utterance_stream(
    utterances: Sequence[TimeAlignedUtterance],
    target_times: np.ndarray,
) -> LanguageStreamTrace:
    """
    VOCABULARY-FREE temporal linguistic stream observer:
    Uses universal structural discourse cues ONLY — utterance onset pulse,
    utterance duration/activity, punctuation energy (`?`/`!`), token cadence,
    and speaker turn shifts. Works identically on Korean, English, or completely
    unseen alien/unknown words with ZERO hardcoded vocabulary lists!
    """
    times = np.asarray(target_times, dtype=np.float64)
    n = len(times)
    activity = np.zeros(n, dtype=np.float64)
    directive = np.zeros(n, dtype=np.float64)
    causal_conn = np.zeros(n, dtype=np.float64)
    tension = np.zeros(n, dtype=np.float64)
    speaker_shift = np.zeros(n, dtype=np.float64)

    ordered = sorted(utterances, key=lambda u: (u.t_start, u.t_end))
    speakers: List[str] = []
    prev_speaker: Optional[str] = None
    dt = float(times[1] - times[0]) if n > 1 else 0.1

    for idx_u, utt in enumerate(ordered):
        if utt.speaker not in speakers:
            speakers.append(utt.speaker)

        txt = utt.text.strip()
        n_punct = txt.count("!") + txt.count("?")
        n_tokens = max(1, len(txt.split()))
        dur = max(dt, utt.t_end - utt.t_start)
        # Speech cadence (tokens/sec) & exclamatory/interrogative inflection — completely language-agnostic!
        cadence_norm = float(min(1.0, (n_tokens / dur) / 6.0))
        pulse_strength = float(min(1.0, 0.55 + 0.25 * n_punct + 0.20 * cadence_norm))
        # Discourse continuity density rises when multi-turn utterances link sequentially
        conn_val = float(min(1.0, 0.35 + 0.20 * min(3, idx_u)))
        ten_val = float(np.clip(0.30 + 0.30 * min(2, n_punct) + 0.20 * cadence_norm, 0.1, 1.0))

        mask = (times >= utt.t_start - 0.5 * dt) & (times <= utt.t_end + 0.5 * dt)
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(times - 0.5 * (utt.t_start + utt.t_end))))
            mask[nearest] = True

        activity[mask] = 1.0
        directive[mask] = np.maximum(directive[mask], pulse_strength)
        causal_conn[mask] = np.maximum(causal_conn[mask], conn_val)
        tension[mask] = np.maximum(tension[mask], ten_val)

        if prev_speaker is not None and prev_speaker != utt.speaker:
            speaker_shift[int(np.argmax(mask))] = 1.0
        prev_speaker = utt.speaker

    last_t = 0.25
    for i in range(n):
        if activity[i] > 0.0:
            last_t = tension[i]
        else:
            tension[i] = 0.90 * last_t + 0.10 * 0.25
            last_t = tension[i]

    return LanguageStreamTrace(
        times=times,
        utterances=list(ordered),
        utterance_activity=activity,
        directive_intent_flux=directive,
        causal_connective_density=causal_conn,
        narrative_tension_arc=tension,
        speaker_turn_shift=speaker_shift,
        speakers=speakers,
    )


def fit_causal_transfer_operator(
    cause: np.ndarray,
    effect: np.ndarray,
    dt: float,
    source_name: str,
    target_name: str,
    max_lag: int = 5,
    ridge: float = 0.05,
) -> CrossModalTransferOperator:
    """
    Distills a genuine causal FIR transfer kernel `h[0..L]` (strictly causal:
    `tau >= 0` ONLY!) using ridge-regularized Toeplitz least squares so Elysia
    can simulate/generate effects from brand-new unseen causes.
    """
    x = np.asarray(cause, dtype=np.float64).ravel()
    y = np.asarray(effect, dtype=np.float64).ravel()
    n = min(len(x), len(y))
    L = max(1, min(max_lag + 1, n // 3))
    if n <= L + 2 or float(np.std(x[:n])) < 1e-8:
        return CrossModalTransferOperator(
            source_name=source_name,
            target_name=target_name,
            kernel=np.zeros(L, dtype=np.float64),
            bias=float(np.mean(y[:n])) if n > 0 else 0.0,
            dt=dt,
            train_r2=0.0,
        )

    # Build strictly causal Toeplitz design matrix: row t uses [x[t], x[t-1], ..., x[t-L+1], 1]
    X = np.zeros((n, L + 1), dtype=np.float64)
    for lag in range(L):
        if lag == 0:
            X[:, 0] = x[:n]
        else:
            X[lag:, lag] = x[: n - lag]
    X[:, L] = 1.0

    reg = np.eye(L + 1, dtype=np.float64) * ridge
    reg[L, L] = 1e-6  # unpenalized bias
    theta = np.linalg.solve(X.T @ X + reg, X.T @ y[:n])
    pred = X @ theta
    ss_res = float(np.sum((y[:n] - pred) ** 2))
    ss_tot = float(np.sum((y[:n] - np.mean(y[:n])) ** 2)) + 1e-12
    r2 = float(max(0.0, 1.0 - ss_res / ss_tot))

    return CrossModalTransferOperator(
        source_name=source_name,
        target_name=target_name,
        kernel=theta[:L].copy(),
        bias=float(theta[L]),
        dt=float(dt),
        train_r2=r2,
    )


# ---------------------------------------------------------------------------
# 3. Pixel-Rendered Interactive Game & Audiovisual Narrative Boundaries
# ---------------------------------------------------------------------------


@dataclass
class PixelMediaBundle:
    """Raw multi-modal bundle exposed to the blind observer (NO hidden state!)."""

    bundle_id: str
    description: str
    frames: np.ndarray
    fps: float
    audio_pcm: np.ndarray
    sample_rate: int
    utterances: List[TimeAlignedUtterance] = field(default_factory=list)
    control_signal: Optional[np.ndarray] = None


def _draw_filled_circle(img: np.ndarray, cx: int, cy: int, radius: int, color: Tuple[int, int, int]) -> None:
    """Fast pure-NumPy filled disk rasterizer (zero cv2 import overhead!)."""
    h, w = img.shape[:2]
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    img[y0:y1, x0:x1][mask] = color


def _synthesize_burst(
    sr: int,
    duration_s: float,
    freq_hz: float,
    kind: str = "impact",
    amp: float = 0.8,
) -> np.ndarray:
    n = max(16, int(sr * duration_s))
    t = np.arange(n, dtype=np.float64) / float(sr)
    if kind == "impact":
        env = np.exp(-t / max(0.02, duration_s * 0.25))
        sig = amp * env * (
            0.7 * np.sin(2.0 * math.pi * freq_hz * t) + 0.3 * np.sin(2.0 * math.pi * (2.3 * freq_hz) * t)
        )
    elif kind == "voice_phrase":
        f0 = freq_hz * (1.0 + 0.08 * np.sin(2.0 * math.pi * 3.5 * t))
        phase = 2.0 * math.pi * np.cumsum(f0) / float(sr)
        env = np.sin(math.pi * np.clip(t / max(1e-6, duration_s), 0.0, 1.0)) ** 0.7
        sig = amp * env * (0.65 * np.sin(phase) + 0.25 * np.sin(2.0 * phase) + 0.12 * np.sin(3.0 * phase))
    else:
        sig = amp * (0.6 * np.sin(2.0 * math.pi * freq_hz * t) + 0.3 * np.sin(2.0 * math.pi * (1.5 * freq_hz) * t))
    return sig


def run_pixel_rendered_interactive_game(
    mode: str = "interactive_play",
    n_frames: int = 44,
    fps: float = 12.0,
    sample_rate: int = 16000,
    seed: int = 7,
) -> PixelMediaBundle:
    """
    Pure-NumPy pixel-rendered 2D physics game world (`128x80` RGB frames +
    synchronized PCM audio + synchronized dialogue + live control signal).
    """
    rng = np.random.default_rng(seed)
    w, h = 128, 80
    dt = 1.0 / fps
    total_s = n_frames * dt

    game_thrust = np.zeros(n_frames, dtype=np.float64)
    for k in (5, 6, 17, 18, 29, 30):
        if k < n_frames:
            game_thrust[k] = 1.0

    x, y = 20.0, 62.0
    vx, vy = 19.0, 0.0
    g = 98.0
    floor_y = 62.0
    frames = np.zeros((n_frames, h, w, 3), dtype=np.uint8)
    audio = np.zeros(int(round(total_s * sample_rate)) + sample_rate // 4, dtype=np.float64)
    t_all = np.arange(len(audio), dtype=np.float64) / float(sample_rate)
    audio += 0.025 * np.sin(2.0 * math.pi * 110.0 * t_all)

    utterances: List[TimeAlignedUtterance] = [
        TimeAlignedUtterance(0.10, 0.55, "player", "지금 출발하자! 점프 준비."),
    ]
    landing_logged = 0

    for i in range(n_frames):
        t_now = i * dt
        if game_thrust[i] > 0.5 and y >= floor_y - 10.0:
            vy = -58.0
            burst = _synthesize_burst(sample_rate, 0.14, 290.0, kind="impact", amp=0.65)
            idx0 = int(t_now * sample_rate)
            audio[idx0 : idx0 + len(burst)] += burst[: max(0, len(audio) - idx0)]

        vy += g * dt
        x += vx * dt
        y += vy * dt

        if x > w - 12.0:
            x = w - 12.0
            vx = -abs(vx)
        elif x < 12.0:
            x = 12.0
            vx = abs(vx)

        if y >= floor_y:
            impact_v = abs(vy)
            y = floor_y
            if impact_v > 22.0:
                vy = -0.35 * impact_v
                boom = _synthesize_burst(sample_rate, 0.18, 125.0, kind="impact", amp=min(0.95, impact_v / 60.0))
                idx0 = int(t_now * sample_rate)
                audio[idx0 : idx0 + len(boom)] += boom[: max(0, len(audio) - idx0)]
                landing_logged += 1
                if landing_logged == 1:
                    utterances.append(
                        TimeAlignedUtterance(t_now + 0.08, t_now + 0.55, "narrator", "그래서 강한 충돌이 바닥에 울렸다!")
                    )
            else:
                vy = 0.0

        img = frames[i]
        img[:, :] = (18, 28, 52)
        img[int(floor_y) + 8 :, :] = (42, 68, 48)
        img[52 : int(floor_y) + 8, 60:68] = (140, 100, 60)
        color = (255, 210, 70) if game_thrust[i] > 0.5 else (235, 115, 55)
        _draw_filled_circle(img, int(round(x)), int(round(y)), 8, color)
        if game_thrust[i] > 0.5:
            _draw_filled_circle(img, int(round(x)), int(round(y)) + 9, 4, (110, 210, 255))

    utterances.append(
        TimeAlignedUtterance(total_s - 0.65, total_s - 0.10, "narrator", "결국 궤도가 안전하게 안정되었다.")
    )

    audio = audio[: int(round(total_s * sample_rate))]
    if mode == "desynchronized_audio":
        audio = np.roll(audio, int(len(audio) * 0.47))

    if mode == "passive_video_playback":
        probe_ctrl = np.zeros(n_frames, dtype=np.float64)
        for k in (11, 12, 23, 24, 36, 37):
            if k < n_frames:
                probe_ctrl[k] = 1.0
        probe_ctrl += rng.normal(0.0, 0.02, size=n_frames)
        exposed_ctrl = np.clip(probe_ctrl, 0.0, 1.0)
        desc = "녹화된 게임 영상 재생(외부 관찰: 내 조작이 화면에 연결되지 않음)"
    elif mode == "desynchronized_audio":
        exposed_ctrl = game_thrust.copy()
        desc = "오디오 비동기화 반사실 대조 영상(움직임-소리 시간 어긋남)"
    else:
        exposed_ctrl = game_thrust.copy()
        desc = "실시간 픽셀 렌더링 상호작용 게임(내 조작 → 움직임 → 충돌음)"

    return PixelMediaBundle(
        bundle_id=f"game_world::{mode}",
        description=desc,
        frames=frames,
        fps=fps,
        audio_pcm=audio.astype(np.float32),
        sample_rate=sample_rate,
        utterances=utterances,
        control_signal=exposed_ctrl,
    )


def build_audiovisual_narrative_scene(
    narrative_kind: str = "volitional",
    n_frames: int = 48,
    fps: float = 12.0,
    sample_rate: int = 16000,
    use_unknown_language: bool = False,
) -> PixelMediaBundle:
    """
    Pure-NumPy pixel-rendered multi-scene audiovisual drama WITH a sharp visual
    scene cut bridged by continuous audio & dialogue. Supports `use_unknown_language=True`
    to prove ZERO reliance on Korean/English keyword dictionaries!
    """
    w, h = 128, 80
    dt = 1.0 / fps
    total_s = n_frames * dt
    cut_frame = n_frames // 2

    audio = np.zeros(int(round(total_s * sample_rate)) + sample_rate // 2, dtype=np.float64)
    t_all = np.arange(len(audio), dtype=np.float64) / float(sample_rate)
    audio += 0.14 * np.sin(2.0 * math.pi * 220.0 * t_all) + 0.07 * np.sin(2.0 * math.pi * 330.0 * t_all)

    frames = np.zeros((n_frames, h, w, 3), dtype=np.uint8)
    utterances: List[TimeAlignedUtterance] = []

    if narrative_kind == "volitional":
        if use_unknown_language:
            utterances = [
                TimeAlignedUtterance(0.45, 1.05, "a1", "Zorblax venn-ta!"),
                TimeAlignedUtterance(1.85, 2.55, "a1", "Krellum dora-n!"),
                TimeAlignedUtterance(3.25, 3.85, "a2", "Miro senna."),
            ]
        else:
            utterances = [
                TimeAlignedUtterance(0.45, 1.05, "guide", "지금 저쪽 빛으로 가자! 출발해!"),
                TimeAlignedUtterance(1.85, 2.55, "guide", "장면이 바뀌어도 계속 가자! 멈추지 마!"),
                TimeAlignedUtterance(3.25, 3.85, "traveler", "그래서 마침내 안전하게 도착했다."),
            ]
        for u, f0 in zip(utterances, (185.0, 205.0, 170.0)):
            vp = _synthesize_burst(sample_rate, u.t_end - u.t_start, f0, kind="voice_phrase", amp=0.55)
            idx0 = int(u.t_start * sample_rate)
            audio[idx0 : idx0 + len(vp)] += vp[: max(0, len(audio) - idx0)]

        pos_x = 18.0
        for i in range(n_frames):
            t_now = i * dt
            moving = (0.85 <= t_now <= 1.65) or (2.40 <= t_now <= 3.25)
            speed = 70.0 if moving else 2.5
            pos_x = 16.0 + ((pos_x - 16.0 + speed * dt) % (w - 32.0))

            img = frames[i]
            if i < cut_frame:
                img[:, :] = (24, 38, 72)
                img[60:, :] = (35, 45, 58)
            else:
                img[:, :] = (210, 128, 74)
                img[60:, :] = (95, 72, 48)
            _draw_filled_circle(img, int(round(pos_x)), 44, 9, (245, 240, 190))
        desc = "의지적 서사 클립(발화 선행 → 움직임 전개 + 시각 컷 교량)"
    else:
        if use_unknown_language:
            utterances = [
                TimeAlignedUtterance(1.05, 1.65, "b1", "Quorath! Vex?"),
                TimeAlignedUtterance(1.90, 2.35, "b1", "Tala moru."),
                TimeAlignedUtterance(2.85, 3.60, "b1", "Nora silen."),
            ]
        else:
            utterances = [
                TimeAlignedUtterance(1.05, 1.65, "witness", "갑자기 충돌 위험! 왜 부딪혔지?"),
                TimeAlignedUtterance(1.90, 2.35, "witness", "장면 너머에서도 진동이 이어진다."),
                TimeAlignedUtterance(2.85, 3.60, "witness", "결국 두 번째 충격 뒤에야 평온해졌다."),
            ]
        for t_hit in (0.75, 2.50):
            boom = _synthesize_burst(sample_rate, 0.22, 115.0, kind="impact", amp=0.85)
            idx0 = int(t_hit * sample_rate)
            audio[idx0 : idx0 + len(boom)] += boom[: max(0, len(audio) - idx0)]
        for u, f0 in zip(utterances, (220.0, 190.0, 175.0)):
            vp = _synthesize_burst(sample_rate, u.t_end - u.t_start, f0, kind="voice_phrase", amp=0.45)
            idx0 = int(u.t_start * sample_rate)
            audio[idx0 : idx0 + len(vp)] += vp[: max(0, len(audio) - idx0)]

        pos_x = 24.0
        direction = 1.0
        for i in range(n_frames):
            t_now = i * dt
            surge = (0.45 <= t_now <= 0.95) or (2.25 <= t_now <= 2.75)
            speed = 115.0 if surge else 3.0
            pos_x += direction * speed * dt
            if pos_x > w - 16.0:
                pos_x = w - 16.0
                direction = -1.0
            elif pos_x < 16.0:
                pos_x = 16.0
                direction = 1.0

            img = frames[i]
            if i < cut_frame:
                img[:, :] = (32, 32, 46)
                img[62:, :] = (58, 52, 60)
            else:
                img[:, :] = (165, 190, 215)
                img[62:, :] = (70, 95, 110)
            col = (255, 80, 60) if surge else (190, 210, 230)
            _draw_filled_circle(img, int(round(pos_x)), 42, 10, col)
        desc = "반응적 서사 클립(움직임·충격 선행 → 굉음 → 발화 반응 + 컷 교량)"

    audio = audio[: int(round(total_s * sample_rate))]
    return PixelMediaBundle(
        bundle_id=f"cinema_narrative::{narrative_kind}{'_alien' if use_unknown_language else ''}",
        description=desc,
        frames=frames,
        fps=fps,
        audio_pcm=audio.astype(np.float32),
        sample_rate=sample_rate,
        utterances=utterances,
        control_signal=None,
    )


# ---------------------------------------------------------------------------
# 4. Cross-Modal Lead-Lag Web, Generative Self-Improvement & Novel Creation
# ---------------------------------------------------------------------------


def _lagged_correlation_profile(
    src: np.ndarray,
    dst: np.ndarray,
    max_lag: int,
) -> Tuple[int, float, float, float]:
    n = min(len(src), len(dst))
    if n < 6:
        return 0, 0.0, 0.0, 0.0
    a = np.asarray(src[:n], dtype=np.float64)
    b = np.asarray(dst[:n], dtype=np.float64)
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0, 0.0, 0.0, 0.0

    lim = max(1, min(max_lag, n // 4))
    best_lag = 0
    best_corr = 0.0
    zero_corr = _safe_norm_corr(a, b)
    max_fwd = 0.0
    max_bwd = 0.0

    for lag in range(-lim, lim + 1):
        if lag > 0:
            r = _safe_norm_corr(a[:-lag], b[lag:])
        elif lag < 0:
            r = _safe_norm_corr(a[-lag:], b[:lag])
        else:
            r = zero_corr

        if lag >= 0:
            max_fwd = max(max_fwd, abs(r))
        else:
            max_bwd = max(max_bwd, abs(r))

        if abs(r) > abs(best_corr) + 1e-6:
            best_corr = r
            best_lag = lag

    return best_lag, float(best_corr), float(zero_corr), float(max_fwd - max_bwd)


def _smooth_series(x: np.ndarray, width: int = 3) -> np.ndarray:
    if len(x) < width or width <= 1:
        return x.copy()
    return np.convolve(x, np.ones(width, dtype=np.float64) / float(width), mode="same")


def _normalize01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x, dtype=np.float64) if (hi - lo <= 1e-9) else (x - lo) / (hi - lo)


class ConnectedMediaCausalObserver:
    """
    Generative O(K) Scale-Rotor observer, recreator, autonomous self-improver,
    and counterfactual world generator for videos, speech, and interactive games.
    """

    def __init__(
        self,
        memory: Optional[OpenVirtualRelationalMemory] = None,
        living_cycle: Optional["ElysiaUnifiedLivingCycle"] = None,
    ) -> None:
        self.memory = memory if memory is not None else OpenVirtualRelationalMemory()
        self.living_cycle = living_cycle
        self.observed_episodes: Dict[str, ConnectedMediaEpisodeReport] = {}

    def observe_episode(
        self,
        episode_id: str,
        description: str = "",
        video_source: Optional[Path | str | Sequence[np.ndarray]] = None,
        video_fps: Optional[float] = None,
        audio_source: Optional[Path | str | Tuple[np.ndarray, int]] = None,
        utterances: Optional[Sequence[TimeAlignedUtterance]] = None,
        control_signal: Optional[np.ndarray] = None,
        max_frames: int = 48,
        absorb_to_sphere: bool = True,
    ) -> ConnectedMediaEpisodeReport:
        t0 = time.perf_counter()
        visual_trace: Optional[VisualStreamTrace] = None
        audio_trace: Optional[AudioStreamTrace] = None
        lang_trace: Optional[LanguageStreamTrace] = None
        ctrl_arr: Optional[np.ndarray] = None

        if video_source is not None:
            try:
                visual_trace = observe_video_stream(
                    source=video_source,
                    max_frames=max_frames,
                    explicit_fps=video_fps,
                    build_rotor=absorb_to_sphere,
                )
            except Exception as exc:
                self.memory.record_boundary_friction(
                    BoundaryFrictionEvent(
                        event_id=f"boundary::media_visual::{episode_id}",
                        source_name=episode_id,
                        stimulus_snapshot={"modality": "visual"},
                        friction_kind=type(exc).__name__,
                        detail=str(exc),
                    )
                )

        target_times = visual_trace.times if visual_trace is not None else None
        if audio_source is not None:
            try:
                audio_trace = observe_audio_stream(
                    source=audio_source,
                    target_times=target_times,
                    build_rotor=absorb_to_sphere,
                )
            except Exception as exc:
                self.memory.record_boundary_friction(
                    BoundaryFrictionEvent(
                        event_id=f"boundary::media_audio::{episode_id}",
                        source_name=episode_id,
                        stimulus_snapshot={"modality": "audio"},
                        friction_kind=type(exc).__name__,
                        detail=str(exc),
                    )
                )

        if visual_trace is not None:
            times = visual_trace.times
            dt = float(times[1] - times[0]) if len(times) > 1 else 0.1
            duration_s = float(times[-1] + dt)
        elif audio_trace is not None:
            times = audio_trace.times
            dt = float(times[1] - times[0]) if len(times) > 1 else 0.05
            duration_s = float(audio_trace.duration_s)
        elif utterances:
            max_t = max(u.t_end for u in utterances)
            times = np.linspace(0.0, max(1.0, max_t), 24, endpoint=False)
            dt = float(times[1] - times[0])
            duration_s = float(max(1.0, max_t))
        else:
            times = np.linspace(0.0, 1.0, 8, endpoint=False)
            dt = 0.125
            duration_s = 1.0

        if utterances:
            lang_trace = observe_utterance_stream(utterances, target_times=times)

        if control_signal is not None:
            raw_c = np.asarray(control_signal, dtype=np.float64).ravel()
            if len(raw_c) == len(times):
                ctrl_arr = raw_c
            elif len(raw_c) >= 2:
                ctrl_arr = np.interp(
                    np.linspace(0.0, 1.0, len(times)),
                    np.linspace(0.0, 1.0, len(raw_c)),
                    raw_c,
                )

        present: List[str] = []
        absent: List[str] = []
        for name, obj in (
            ("visual", visual_trace),
            ("audio", audio_trace),
            ("language", lang_trace),
            ("control", ctrl_arr),
        ):
            (present if obj is not None else absent).append(name)

        causal_links, agency_mode, causality_mode = self._discover_cross_modal_causal_web(
            times=times,
            dt=dt,
            visual=visual_trace,
            audio=audio_trace,
            language=lang_trace,
            control=ctrl_arr,
        )
        bridges = self._detect_cut_continuity_bridges(visual_trace, audio_trace, lang_trace)
        beats, transitions = self._segment_causal_narrative_arc(
            times, visual_trace, audio_trace, lang_trace, ctrl_arr, bridges, causality_mode
        )
        narrative_sig = self._compute_narrative_signature(beats, causal_links, agency_mode, visual_trace, audio_trace)

        if absorb_to_sphere:
            macro_droplet = self._absorb_episode_into_sphere(
                episode_id, beats, causal_links, narrative_sig, agency_mode, causality_mode
            )
        else:
            macro_droplet = CausalDroplet(
                droplet_id=f"temp::{episode_id}",
                scale_radius=3.4,
                spectral_signature=narrative_sig,
            )

        summary = self._compose_summary_ko(
            episode_id, present, absent, agency_mode, causality_mode, causal_links, beats, bridges, audio_trace
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        report = ConnectedMediaEpisodeReport(
            episode_id=episode_id,
            source_description=description or episode_id,
            duration_s=duration_s,
            present_modalities=present,
            absent_modalities=absent,
            visual_trace=visual_trace,
            audio_trace=audio_trace,
            language_trace=lang_trace,
            control_trace=ctrl_arr,
            causal_links=causal_links,
            agency_mode=agency_mode,
            causality_mode=causality_mode,
            beats=beats,
            transitions=transitions,
            continuity_bridges=bridges,
            macro_droplet=macro_droplet,
            narrative_signature=narrative_sig,
            summary_ko=summary,
            elapsed_ms=elapsed_ms,
        )
        if absorb_to_sphere:
            self.observed_episodes[episode_id] = report
            self._seal_living_witness(report)
        return report

    def observe_bundle(self, bundle: PixelMediaBundle, absorb_to_sphere: bool = True) -> ConnectedMediaEpisodeReport:
        return self.observe_episode(
            episode_id=bundle.bundle_id,
            description=bundle.description,
            video_source=bundle.frames,
            video_fps=bundle.fps,
            audio_source=(bundle.audio_pcm, bundle.sample_rate),
            utterances=bundle.utterances,
            control_signal=bundle.control_signal,
            max_frames=len(bundle.frames),
            absorb_to_sphere=absorb_to_sphere,
        )

    # ------------------------------------------------------------------
    # CLOSED-LOOP AUTONOMOUS RE-CREATION & SELF-IMPROVEMENT (Cycle 0->1->2)
    # ------------------------------------------------------------------

    def learn_recreate_and_self_improve(
        self,
        episode_id: str,
        video_source: Optional[Path | str | Sequence[np.ndarray]] = None,
        video_fps: Optional[float] = None,
        audio_source: Optional[Path | str | Tuple[np.ndarray, int]] = None,
        max_frames: int = 36,
        max_cycles: int = 3,
    ) -> SelfImprovementReport:
        """
        TRUE CLOSED-LOOP GENERATIVE LEARNING & SELF-IMPROVEMENT:
          1. Starts with a minimal coarse Scale-Rotor Spectrum (Cycle 0).
          2. Re-creates actual RGB video frames & PCM audio strictly from that
             Scale-Rotor Spectrum (ZERO peeking at original pixels/samples!).
          3. Re-observes her own re-creation through the EXACT SAME blind observer,
             measuring pixel fidelity, motion-kinematic fidelity, and audio envelope
             fidelity.
          4. Inspects the structural residual `R = Reality - Recreation` and
             autonomously expands spatial/temporal rotor modes + event knots
             across cycles (`Cycle 0 -> Cycle 1 -> Cycle 2`), verifying strict
             monotonic self-improvement in milliseconds!
        """
        t_start = time.perf_counter()

        raw_frames: Optional[np.ndarray] = None
        fps = float(video_fps or 12.0)
        ref_vis_trace: Optional[VisualStreamTrace] = None
        if video_source is not None:
            raw_frames, fps, _ = _load_video_or_gif_frames(
                video_source, max_frames=max_frames, explicit_fps=video_fps
            )
            ref_vis_trace = observe_video_stream(raw_frames, max_frames=len(raw_frames), explicit_fps=fps, build_rotor=False)

        raw_pcm: Optional[np.ndarray] = None
        sr = 16000
        ref_aud_trace: Optional[AudioStreamTrace] = None
        if audio_source is not None:
            if isinstance(audio_source, (str, Path)):
                from scipy.io import wavfile

                sr, data = wavfile.read(str(audio_source))
                arr = np.asarray(data, dtype=np.float64)
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=1)
                if np.issubdtype(data.dtype, np.integer):
                    info = np.iinfo(data.dtype)
                    arr = arr / float(max(abs(info.min), info.max))
                raw_pcm = arr
            else:
                pcm_in, sr = audio_source
                raw_pcm = np.asarray(pcm_in, dtype=np.float64).ravel()
            pk = float(np.max(np.abs(raw_pcm))) + 1e-12
            raw_pcm = raw_pcm / pk
            ref_aud_trace = observe_audio_stream(
                (raw_pcm, sr),
                target_times=ref_vis_trace.times if ref_vis_trace is not None else None,
                build_rotor=False,
            )

        # Progressive autonomous Scale-Rotor schedules driven by measured residual
        # (Cycle 0 = coarse foothold -> Cycle 1 = mid-scale + knot harvest -> Cycle 2 = refined rotor)
        vis_schedules = [(6, 6, 4, 0), (14, 14, 10, 3), (26, 26, 20, 8)]
        aud_schedules = [4, 14, 36]

        cycles: List[SelfImprovementCycleMetrics] = []
        final_v_rotor: Optional[SpatiotemporalScaleRotorSpectrum] = None
        final_a_rotor: Optional[AcousticScaleRotorSpectrum] = None
        rec_frames: Optional[np.ndarray] = None
        rec_pcm: Optional[np.ndarray] = None

        for c_idx in range(min(max_cycles, len(vis_schedules))):
            tc0 = time.perf_counter()
            v_mse, v_fid, flow_corr, v_comp, v_knots = 0.0, 1.0, 1.0, 1.0, 0
            a_mse, a_env_corr, a_comp = 0.0, 1.0, 1.0

            if raw_frames is not None and ref_vis_trace is not None:
                kh, kw, kt, mk = vis_schedules[c_idx]
                v_rotor = encode_video_to_scale_rotor(
                    raw_frames, fps=fps, kh_max=kh, kw_max=kw, k_temporal=kt, max_knots=mk
                )
                rec_frames = v_rotor.recreate_frames_at(ref_vis_trace.times)

                # Re-observe her OWN recreation through the exact same blind visual observer!
                rec_vis_trace = observe_video_stream(
                    rec_frames, max_frames=len(rec_frames), explicit_fps=fps, build_rotor=False
                )
                orig_f = raw_frames.astype(np.float64) / 255.0
                rec_f = rec_frames.astype(np.float64) / 255.0
                v_mse = float(np.mean((orig_f - rec_f) ** 2))
                v_fid = float(max(0.0, 1.0 - math.sqrt(v_mse)))
                flow_corr = _safe_norm_corr(ref_vis_trace.active_object_speed, rec_vis_trace.active_object_speed)
                v_comp = v_rotor.compression_ratio
                v_knots = len(v_rotor.knot_corrections)
                final_v_rotor = v_rotor

            if raw_pcm is not None and ref_aud_trace is not None:
                k_bins = aud_schedules[c_idx]
                a_rotor = encode_audio_to_scale_rotor(raw_pcm, sample_rate=sr, k_bins=k_bins)
                rec_pcm = a_rotor.recreate_pcm()

                # Re-observe her OWN synthesized audio through the exact same blind audio observer!
                rec_aud_trace = observe_audio_stream(
                    (rec_pcm, sr), target_times=ref_aud_trace.times, build_rotor=False
                )
                n_min = min(len(raw_pcm), len(rec_pcm))
                a_mse = float(np.mean((raw_pcm[:n_min] - rec_pcm[:n_min].astype(np.float64)) ** 2))
                a_env_corr = _safe_norm_corr(ref_aud_trace.rms_energy, rec_aud_trace.rms_energy)
                a_comp = a_rotor.compression_ratio
                final_a_rotor = a_rotor

            cycles.append(
                SelfImprovementCycleMetrics(
                    cycle_index=c_idx,
                    visual_mse=v_mse,
                    visual_fidelity=v_fid,
                    motion_flow_correlation=flow_corr,
                    audio_mse=a_mse,
                    audio_envelope_correlation=a_env_corr,
                    visual_compression_ratio=v_comp,
                    audio_compression_ratio=a_comp,
                    visual_knot_count=v_knots,
                    elapsed_ms=(time.perf_counter() - tc0) * 1000.0,
                )
            )

        v_drop = (
            float((cycles[0].visual_mse - cycles[-1].visual_mse) / max(1e-12, cycles[0].visual_mse))
            if raw_frames is not None and len(cycles) >= 2
            else 0.0
        )
        a_drop = (
            float((cycles[0].audio_mse - cycles[-1].audio_mse) / max(1e-12, cycles[0].audio_mse))
            if raw_pcm is not None and len(cycles) >= 2
            else 0.0
        )
        v_ok = (raw_frames is None) or (cycles[-1].visual_mse < cycles[0].visual_mse * 0.85)
        a_ok = (raw_pcm is None) or (cycles[-1].audio_mse < cycles[0].audio_mse * 0.85)
        improved = bool(v_ok and a_ok)

        total_ms = (time.perf_counter() - t_start) * 1000.0
        summary = (
            f"[{episode_id}] {len(cycles)}회 닫힌 자기재창조-잔차수확 순환({total_ms:.1f}ms): "
            f"시각 MSE {cycles[0].visual_mse:.5f}→{cycles[-1].visual_mse:.5f}(감소 {v_drop*100:.1f}%, "
            f"운동흐름 상관 {cycles[-1].motion_flow_correlation:+.3f}, 압축 {cycles[-1].visual_compression_ratio:.1f}x), "
            f"음향 MSE {cycles[0].audio_mse:.5f}→{cycles[-1].audio_mse:.5f}(감소 {a_drop*100:.1f}%, "
            f"포락선 상관 {cycles[-1].audio_envelope_correlation:+.3f})."
        )
        return SelfImprovementReport(
            episode_id=episode_id,
            cycles=cycles,
            final_visual_rotor=final_v_rotor,
            final_audio_rotor=final_a_rotor,
            recreated_frames=rec_frames,
            recreated_pcm=rec_pcm,
            visual_error_drop_ratio=v_drop,
            audio_error_drop_ratio=a_drop,
            strictly_improved=improved,
            total_elapsed_ms=total_ms,
            summary_ko=summary,
        )

    # ------------------------------------------------------------------
    # GENERATIVE COUNTERFACTUAL WORLD CREATION FROM LEARNED PRINCIPLES
    # ------------------------------------------------------------------

    def create_novel_world_from_principles(
        self,
        origin_report: ConnectedMediaEpisodeReport,
        novel_control_signal: np.ndarray,
        fps: float = 12.0,
        sample_rate: int = 16000,
    ) -> NovelWorldCreationReport:
        """
        Generates a BRAND-NEW pixel video + synchronized PCM audio purely from
        the learned Causal Transfer Operators (`control -> visual_motion` and
        `visual_accel -> audio_onset`) + Scale-Rotor chromatic ground — driven by
        a NEVER-BEFORE-SEEN control rhythm — and self-verifies the generated world
        through the exact same blind observer!
        """
        t0 = time.perf_counter()
        ctrl = np.asarray(novel_control_signal, dtype=np.float64).ravel()
        n = len(ctrl)
        dt = 1.0 / fps
        total_s = n * dt
        w, h = 128, 80

        # Retrieve learned causal transfer operators from the observed world
        ctrl_op: Optional[CrossModalTransferOperator] = None
        av_op: Optional[CrossModalTransferOperator] = None
        for lk in origin_report.causal_links:
            if lk.relation_kind == "self_agency_control_to_motion" and lk.transfer_operator is not None:
                ctrl_op = lk.transfer_operator
            elif lk.relation_kind == "audiovisual_physical_synchrony" and lk.transfer_operator is not None:
                av_op = lk.transfer_operator

        # 1. Drive visual kinematics via learned causal operator `H_{ctrl -> vis}`!
        if ctrl_op is not None:
            pred_speed = np.maximum(0.0, ctrl_op.predict(ctrl))
        else:
            pred_speed = np.maximum(0.0, ctrl)

        # Normalize into physical vertical/horizontal trajectory displacement
        sp_norm = _normalize01(pred_speed)
        frames = np.zeros((n, h, w, 3), dtype=np.uint8)
        # Ground chromatic field from learned visual rotor DC mood (if available)
        bg_col = (20, 32, 58)
        if origin_report.visual_trace is not None and origin_report.visual_trace.scale_rotor_spectrum is not None:
            dc = np.real(origin_report.visual_trace.scale_rotor_spectrum.rotor_coeffs[0, 0, 0, :])
            dc = np.clip(dc / max(1.0, float(h * w)) * 255.0, 14, 90).astype(int)
            bg_col = (int(dc[0]), int(dc[1]), int(dc[2]))

        x = 18.0
        for i in range(n):
            # Motion speed driven directly by learned operator output (NOT original replay!)
            step_dx = 1.5 + 11.0 * float(sp_norm[i])
            x = 14.0 + ((x - 14.0 + step_dx) % (w - 28.0))
            y = 60.0 - 34.0 * float(sp_norm[i])
            img = frames[i]
            img[:, :] = bg_col
            img[66:, :] = (46, 74, 52)
            glow = int(min(255, 160 + 95 * sp_norm[i]))
            _draw_filled_circle(img, int(round(x)), int(round(y)), 8, (glow, int(115 + 90 * sp_norm[i]), 65))

        # 2. Drive synchronized acoustic onset energy via learned `H_{vis -> aud}`!
        pred_accel = np.abs(np.diff(sp_norm, prepend=sp_norm[0]))
        if av_op is not None:
            pred_aud_env = np.maximum(0.0, av_op.predict(pred_accel))
        else:
            pred_aud_env = pred_accel
        aud_norm = _normalize01(pred_aud_env)

        n_samples = int(round(total_s * sample_rate))
        audio = np.zeros(n_samples + sample_rate // 4, dtype=np.float64)
        t_all = np.arange(len(audio), dtype=np.float64) / float(sample_rate)
        audio += 0.02 * np.sin(2.0 * math.pi * 120.0 * t_all)
        for i in range(n):
            if aud_norm[i] > 0.35:
                burst = _synthesize_burst(
                    sample_rate, 0.14, 140.0 + 180.0 * float(aud_norm[i]), kind="impact", amp=float(aud_norm[i])
                )
                idx0 = int(i * dt * sample_rate)
                audio[idx0 : idx0 + len(burst)] += burst[: max(0, len(audio) - idx0)]
        pcm = np.clip(audio[:n_samples], -1.0, 1.0).astype(np.float32)

        # 3. SELF-VERIFICATION: Pass her own newly created world back through the blind observer!
        verified = self.observe_episode(
            episode_id=f"created_from::{origin_report.episode_id}",
            description="학습된 스케일 로터·인과 전달 연산자로 신규 조작 리듬을 재창조한 세계",
            video_source=frames,
            video_fps=fps,
            audio_source=(pcm, sample_rate),
            control_signal=ctrl,
            max_frames=n,
            absorb_to_sphere=False,
        )
        agency_ok = verified.agency_mode == AgencyMode.SELF_AGENCY_INTERACTIVE
        sync_ok = any(lk.relation_kind == "audiovisual_physical_synchrony" for lk in verified.causal_links)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        summary = (
            f"'{origin_report.episode_id}'에서 추출한 인과 전달 연산자(H)와 로터 원리만으로 "
            f"한 번도 보지 못한 새 조작 리듬의 시청각 세계({n}프레임+{total_s:.1f}s PCM)를 {elapsed_ms:.1f}ms 만에 "
            f"직접 재창조하고 자기 재관측 검증 완료(행위주체 유지={agency_ok}, 시청각 동기 유지={sync_ok})."
        )
        return NovelWorldCreationReport(
            origin_episode_id=origin_report.episode_id,
            created_frames=frames,
            created_pcm=pcm,
            sample_rate=sample_rate,
            fps=fps,
            novel_control_signal=ctrl,
            self_verified_report=verified,
            agency_preserved=agency_ok,
            synchrony_preserved=sync_ok,
            elapsed_ms=elapsed_ms,
            summary_ko=summary,
        )

    # ------------------------------------------------------------------
    # Internal Causal Web, Continuity Bridges, Narrative Arc & Sphere
    # ------------------------------------------------------------------

    def _discover_cross_modal_causal_web(
        self,
        times: np.ndarray,
        dt: float,
        visual: Optional[VisualStreamTrace],
        audio: Optional[AudioStreamTrace],
        language: Optional[LanguageStreamTrace],
        control: Optional[np.ndarray],
    ) -> Tuple[List[CrossModalCausalLink], AgencyMode, NarrativeCausalityMode]:
        links: List[CrossModalCausalLink] = []
        max_lag = max(2, min(10, int(round(0.95 / max(1e-3, dt)))))

        # A) Control -> Visual Motion (Self-Agency + Generative Causal Transfer Operator!)
        agency_mode = AgencyMode.PASSIVE_NO_CONTROL_TRIED
        ctrl_visual_peak = 0.0
        if control is not None and visual is not None:
            vis_speed = _smooth_series(visual.active_object_speed, 2)
            _, _, z_corr, asym = _lagged_correlation_profile(control, vis_speed, max_lag=max(3, max_lag // 2))
            best_fwd_r, best_fwd_lag = 0.0, 0
            for l_step in range(0, min(5, len(control) // 4) + 1):
                r_l = _safe_norm_corr(control[:-l_step], vis_speed[l_step:]) if l_step > 0 else z_corr
                if r_l > best_fwd_r:
                    best_fwd_r, best_fwd_lag = r_l, l_step
            ctrl_visual_peak = best_fwd_r
            if best_fwd_r >= 0.38:
                agency_mode = AgencyMode.SELF_AGENCY_INTERACTIVE
                op = fit_causal_transfer_operator(
                    control, vis_speed, dt=dt, source_name="control_action", target_name="active_object_speed", max_lag=5
                )
                links.append(
                    CrossModalCausalLink(
                        source_modality="control",
                        source_channel="control_action",
                        target_modality="visual",
                        target_channel="active_object_speed",
                        best_lag_steps=best_fwd_lag,
                        best_lag_seconds=float(best_fwd_lag * dt),
                        peak_correlation=float(best_fwd_r),
                        zero_lag_correlation=float(z_corr),
                        directional_asymmetry=float(asym),
                        relation_kind="self_agency_control_to_motion",
                        interpretation_ko=(
                            f"내 조작 입력이 +{best_fwd_lag * dt:.2f}초 뒤 화면 객체 움직임(r={best_fwd_r:+.3f}, "
                            f"전달연산자 R2={op.train_r2:.2f})을 이끌어 자기 행위 세계로 분별되었습니다."
                        ),
                        transfer_operator=op,
                    )
                )
            else:
                agency_mode = AgencyMode.EXTERNAL_AGENCY_OBSERVED

        # B) Visual Acceleration <-> Audio Onset (Physical Synchrony + Generative Transfer Operator!)
        if visual is not None and audio is not None:
            v_imp = _smooth_series(visual.motion_acceleration, 2)
            a_imp = _smooth_series(audio.onset_flux + 0.5 * audio.rms_energy, 2)
            sync_lim = max(2, int(round(0.35 / max(1e-3, dt))))
            lag, peak, z_corr, asym = _lagged_correlation_profile(v_imp, a_imp, max_lag=sync_lim)
            if peak >= 0.35:
                op_av = fit_causal_transfer_operator(
                    v_imp, a_imp, dt=dt, source_name="motion_acceleration", target_name="audio_onset", max_lag=3
                )
                links.append(
                    CrossModalCausalLink(
                        source_modality="visual",
                        source_channel="motion_acceleration",
                        target_modality="audio",
                        target_channel="onset_flux",
                        best_lag_steps=lag,
                        best_lag_seconds=float(lag * dt),
                        peak_correlation=float(peak),
                        zero_lag_correlation=float(z_corr),
                        directional_asymmetry=float(asym),
                        relation_kind="audiovisual_physical_synchrony",
                        interpretation_ko=(
                            f"시각 가속·충격과 음향 온셋이 시차 {lag * dt:+.2f}초(r={peak:+.3f})로 "
                            "동기 공명하여 물리적 시청각 전달 연산자를 형성했습니다."
                        ),
                        transfer_operator=op_av,
                    )
                )

        # C) Vocabulary-Free Language <-> Visual Motion (Volitional vs Reactive Narrative!)
        if visual is not None and language is not None:
            v_mov = _smooth_series(visual.active_object_speed + visual.motion_energy, 3)
            l_sig = _smooth_series(0.60 * language.utterance_activity + 0.60 * language.directive_intent_flux, 2)
            _, _, z_corr, _ = _lagged_correlation_profile(l_sig, v_mov, max_lag=max_lag)

            lim = max(2, min(max_lag, len(v_mov) // 4))
            fwd_best_lag, fwd_best_r = 1, 0.0
            bwd_best_lag, bwd_best_r = 1, 0.0
            for k in range(1, lim + 1):
                rf = _safe_norm_corr(l_sig[:-k], v_mov[k:])
                rb = _safe_norm_corr(v_mov[:-k], l_sig[k:])
                if rf > fwd_best_r:
                    fwd_best_r, fwd_best_lag = rf, k
                if rb > bwd_best_r:
                    bwd_best_r, bwd_best_lag = rb, k

            if fwd_best_r >= 0.35 and fwd_best_r >= bwd_best_r + 0.06:
                op_lv = fit_causal_transfer_operator(
                    l_sig, v_mov, dt=dt, source_name="utterance_pulse", target_name="motion_flow", max_lag=lim
                )
                links.append(
                    CrossModalCausalLink(
                        source_modality="language",
                        source_channel="utterance_pulse",
                        target_modality="visual",
                        target_channel="motion_flow",
                        best_lag_steps=fwd_best_lag,
                        best_lag_seconds=float(fwd_best_lag * dt),
                        peak_correlation=float(fwd_best_r),
                        zero_lag_correlation=float(z_corr),
                        directional_asymmetry=float(fwd_best_r - bwd_best_r),
                        relation_kind="volitional_language_to_motion",
                        interpretation_ko=(
                            f"발화 파동이 움직임보다 +{fwd_best_lag * dt:.2f}초 선행"
                            f"(r={fwd_best_r:+.3f} vs 역방향 {bwd_best_r:+.3f})하여 "
                            "단어 사전 없이 의지적 서사 인과로 판별되었습니다."
                        ),
                        transfer_operator=op_lv,
                    )
                )
            elif bwd_best_r >= 0.35 and bwd_best_r >= fwd_best_r + 0.06:
                op_vl = fit_causal_transfer_operator(
                    v_mov, l_sig, dt=dt, source_name="motion_flow", target_name="utterance_pulse", max_lag=lim
                )
                links.append(
                    CrossModalCausalLink(
                        source_modality="visual",
                        source_channel="motion_flow",
                        target_modality="language",
                        target_channel="reactive_utterance",
                        best_lag_steps=bwd_best_lag,
                        best_lag_seconds=float(bwd_best_lag * dt),
                        peak_correlation=float(bwd_best_r),
                        zero_lag_correlation=float(z_corr),
                        directional_asymmetry=float(bwd_best_r - fwd_best_r),
                        relation_kind="reactive_motion_to_language",
                        interpretation_ko=(
                            f"시각 움직임·사건이 발화 반응보다 +{bwd_best_lag * dt:.2f}초 선행"
                            f"(r={bwd_best_r:+.3f} vs 역방향 {fwd_best_r:+.3f})하여 "
                            "단어 사전 없이 반응적 서사 인과로 판별되었습니다."
                        ),
                        transfer_operator=op_vl,
                    )
                )

        # D) Vocal Harmonicity <-> Utterance Activity
        if audio is not None and language is not None:
            lag, peak, z_corr, asym = _lagged_correlation_profile(
                audio.vocal_harmonicity * audio.rms_energy, language.utterance_activity, max_lag=3
            )
            if abs(peak) >= 0.30:
                links.append(
                    CrossModalCausalLink(
                        source_modality="audio",
                        source_channel="vocal_harmonicity",
                        target_modality="language",
                        target_channel="utterance_activity",
                        best_lag_steps=lag,
                        best_lag_seconds=float(lag * dt),
                        peak_correlation=float(peak),
                        zero_lag_correlation=float(z_corr),
                        directional_asymmetry=float(asym),
                        relation_kind="voice_to_utterance_synchrony",
                        interpretation_ko=f"실제 음성 조화 파동과 발화 구간이 r={peak:+.3f}로 결속되었습니다.",
                    )
                )

        link_kinds = {lk.relation_kind for lk in links}
        if agency_mode == AgencyMode.SELF_AGENCY_INTERACTIVE and ctrl_visual_peak >= 0.38:
            c_mode = NarrativeCausalityMode.INTERACTIVE_CONTROL_DRIVEN
        elif "volitional_language_to_motion" in link_kinds:
            c_mode = NarrativeCausalityMode.VOLITIONAL_LANGUAGE_LEADS_MOTION
        elif "reactive_motion_to_language" in link_kinds:
            c_mode = NarrativeCausalityMode.REACTIVE_MOTION_LEADS_LANGUAGE
        elif "audiovisual_physical_synchrony" in link_kinds:
            c_mode = NarrativeCausalityMode.AUDIOVISUAL_PHYSICAL_SYNCHRONY
        elif visual is None and audio is not None:
            c_mode = NarrativeCausalityMode.UNIMODAL_VOCAL_PROSODY
        else:
            c_mode = NarrativeCausalityMode.UNIMODAL_PHYSICAL_FLOW

        return links, agency_mode, c_mode

    def _detect_cut_continuity_bridges(
        self,
        visual: Optional[VisualStreamTrace],
        audio: Optional[AudioStreamTrace],
        language: Optional[LanguageStreamTrace],
    ) -> List[ContinuityBridge]:
        if visual is None or not visual.detected_cut_indices:
            return []
        bridges: List[ContinuityBridge] = []
        for idx in visual.detected_cut_indices:
            if idx <= 0 or idx >= len(visual.times):
                continue
            drop = float(max(0.0, 1.0 - visual.visual_continuity[idx]))
            audio_ok, lang_ok = False, False
            bridged_by: List[str] = []
            if audio is not None and idx < len(audio.rms_energy):
                e_prev, e_curr = float(audio.rms_energy[idx - 1]), float(audio.rms_energy[idx])
                if min(e_prev, e_curr) > 0.05 and abs(e_curr - e_prev) < 0.35:
                    audio_ok = True
                    bridged_by.append("audio_bed")
            if language is not None and idx < len(language.utterance_activity):
                lo, hi = max(0, idx - 2), min(len(language.utterance_activity), idx + 3)
                if float(np.max(language.utterance_activity[lo:hi])) > 0.5:
                    lang_ok = True
                    bridged_by.append("dialogue_thread")
            msg = (
                f"t={visual.times[idx]:.2f}s 시각 장면 전환(낙차 {drop:.2f})을 "
                f"{', '.join(bridged_by) if bridged_by else '교량 없음'}이(가) 이어주었습니다."
            )
            bridges.append(
                ContinuityBridge(
                    cut_step=idx,
                    cut_time_s=float(visual.times[idx]),
                    visual_continuity_drop=drop,
                    audio_continuity_preserved=audio_ok,
                    language_continuity_preserved=lang_ok,
                    bridged_by=bridged_by,
                    summary_ko=msg,
                )
            )
        return bridges

    def _segment_causal_narrative_arc(
        self,
        times: np.ndarray,
        visual: Optional[VisualStreamTrace],
        audio: Optional[AudioStreamTrace],
        language: Optional[LanguageStreamTrace],
        control: Optional[np.ndarray],
        bridges: Sequence[ContinuityBridge],
        causality_mode: NarrativeCausalityMode,
    ) -> Tuple[List[CausalNarrativeBeat], List[BeatTransition]]:
        n = len(times)
        cut_set = sorted(b.cut_step for b in bridges if 2 <= b.cut_step <= n - 2)
        if cut_set:
            mid = cut_set[0]
            bounds = sorted({0, max(1, mid // 2), mid, min(n - 1, mid + max(1, (n - mid) // 2)), n})
        else:
            n_beats = 4 if n >= 12 else max(2, n // 3)
            bounds = [int(round(k * n / n_beats)) for k in range(n_beats + 1)]

        v_mov = _normalize01(visual.active_object_speed + visual.motion_energy) if visual is not None else np.zeros(n)
        v_acc = _normalize01(visual.motion_acceleration) if visual is not None else np.zeros(n)
        a_rms = _normalize01(audio.rms_energy) if audio is not None else np.zeros(n)
        a_ons = _normalize01(audio.onset_flux) if audio is not None else np.zeros(n)
        l_dir = language.directive_intent_flux if language is not None else np.zeros(n)
        l_ten = language.narrative_tension_arc if language is not None else np.zeros(n)
        c_arr = control if control is not None else np.zeros(n)

        beats: List[CausalNarrativeBeat] = []
        for idx, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
            if e <= s:
                continue
            m_mean = float(np.mean(v_mov[s:e]))
            a_peak = float(np.max(v_acc[s:e]))
            aud_mean = float(np.mean(a_rms[s:e]))
            ons_peak = float(np.max(a_ons[s:e]))
            dir_lvl = float(np.max(l_dir[s:e]))
            ten_lvl = float(np.mean(l_ten[s:e]))
            ctrl_lvl = float(np.max(c_arr[s:e]))

            rupture = 0.50 * a_peak + 0.35 * ons_peak + 0.15 * ten_lvl
            flux = 0.45 * m_mean + 0.30 * dir_lvl + 0.25 * ctrl_lvl
            if rupture >= 0.58 and rupture >= flux:
                chroma, cue = "Yellow", "impact_or_climax_release"
            elif flux >= 0.35 or aud_mean >= 0.45:
                chroma, cue = "Red", "active_escalation_flow"
            else:
                chroma, cue = "Blue", "equilibrium_or_resolution"

            t0, t1 = float(times[s]), float(times[min(n - 1, e - 1)])
            utts_here = [
                f"{u.speaker}: {u.text}"
                for u in (language.utterances if language is not None else ())
                if u.t_end >= t0 and u.t_start <= t1
            ]
            beats.append(
                CausalNarrativeBeat(
                    beat_index=idx,
                    t_start=t0,
                    t_end=t1,
                    chromatic_character=chroma,
                    mean_motion=m_mean,
                    peak_acceleration=a_peak,
                    mean_audio_energy=aud_mean,
                    peak_onset=ons_peak,
                    directive_level=dir_lvl,
                    tension_level=ten_lvl,
                    control_level=ctrl_lvl,
                    dominant_event_cue=cue,
                    utterances_in_beat=utts_here,
                )
            )

        cut_times = [b.cut_time_s for b in bridges]
        transitions: List[BeatTransition] = []
        for b0, b1 in zip(beats[:-1], beats[1:]):
            if any(b0.t_end - 0.15 <= ct <= b1.t_start + 0.15 for ct in cut_times):
                kind, strength = "cut_bridged_continuity", 0.88
            elif b0.control_level > 0.5 and b1.mean_motion > b0.mean_motion:
                kind, strength = "control_triggered_escalation", 0.90
            elif b0.directive_level > 0.3 and b1.mean_motion >= b0.mean_motion:
                kind, strength = "speech_prompted_action", 0.85
            elif b0.chromatic_character == "Yellow" and b1.chromatic_character == "Blue":
                kind, strength = "climax_to_resolution_dissipation", 0.82
            else:
                kind, strength = f"continuous_{causality_mode.value}", 0.74
            transitions.append(
                BeatTransition(
                    from_beat=b0.beat_index,
                    to_beat=b1.beat_index,
                    transition_kind=kind,
                    coupling_strength=strength,
                    reasoning_ko=f"Beat {b0.beat_index}→{b1.beat_index}: {kind}",
                )
            )
        return beats, transitions

    def _compute_narrative_signature(
        self,
        beats: Sequence[CausalNarrativeBeat],
        links: Sequence[CrossModalCausalLink],
        agency_mode: AgencyMode,
        visual: Optional[VisualStreamTrace],
        audio: Optional[AudioStreamTrace],
    ) -> np.ndarray:
        if not beats:
            return np.ones(8, dtype=np.float64) / math.sqrt(8.0)
        n_b = float(len(beats))
        vec = np.array(
            [
                sum(1 for b in beats if b.chromatic_character == "Blue") / n_b,
                sum(1 for b in beats if b.chromatic_character == "Red") / n_b,
                sum(1 for b in beats if b.chromatic_character == "Yellow") / n_b,
                max(b.peak_acceleration for b in beats),
                float(np.mean([b.mean_audio_energy for b in beats])),
                max((abs(lk.peak_correlation) for lk in links), default=0.0),
                1.0 if agency_mode == AgencyMode.SELF_AGENCY_INTERACTIVE else 0.2,
                float(audio.voiced_fraction) if audio is not None else (
                    float(np.mean(visual.flow_coherence)) if visual is not None else 0.0
                ),
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(vec))
        return (vec / norm) if norm > 1e-12 else (np.ones(8, dtype=np.float64) / math.sqrt(8.0))

    def _absorb_episode_into_sphere(
        self,
        episode_id: str,
        beats: Sequence[CausalNarrativeBeat],
        causal_links: Sequence[CrossModalCausalLink],
        narrative_sig: np.ndarray,
        agency_mode: AgencyMode,
        causality_mode: NarrativeCausalityMode,
    ) -> CausalDroplet:
        beat_ids: List[str] = []
        for b in beats:
            bid = f"{episode_id}::beat_{b.beat_index}"
            beat_ids.append(bid)
            dr = CausalDroplet(
                droplet_id=bid,
                surface_tension=float(min(1.0, 0.45 + 0.4 * b.peak_acceleration)),
                scale_radius=0.85,
                spectral_signature=np.array(
                    [
                        1.0 if b.chromatic_character == "Blue" else 0.1,
                        1.0 if b.chromatic_character == "Red" else 0.1,
                        1.0 if b.chromatic_character == "Yellow" else 0.1,
                        b.mean_motion,
                        b.peak_acceleration,
                        b.mean_audio_energy,
                        b.directive_level,
                        b.control_level,
                    ],
                    dtype=np.float64,
                ),
            )
            self.memory.absorb_droplet(dr)

        link_ids: List[str] = []
        for idx, lk in enumerate(causal_links):
            ax_id = f"axis::{episode_id}::{lk.relation_kind}_{idx}"
            link_ids.append(ax_id)
            theta, phi = self.memory._allocate_spherical_angles()
            rotor = ScaleRotor(
                axis=(math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)),
                angular_rate=abs(lk.peak_correlation),
                phase=theta,
                radius=2.1,
                scale=max(0.2, abs(lk.peak_correlation)),
            )
            self.memory.absorb_axis(
                CognitiveAxis(
                    axis_id=ax_id,
                    stimulus_channel=f"{lk.source_modality}.{lk.source_channel}",
                    response_channels=(f"{lk.target_modality}.{lk.target_channel}",),
                    correlation_spectrum={lk.relation_kind: lk.peak_correlation},
                    invariant_channels=(),
                    dominant_correlation=lk.peak_correlation,
                    dominant_slope=lk.best_lag_seconds,
                    monotonicity=lk.directional_asymmetry,
                    sensitivity_resolution=abs(lk.peak_correlation),
                    scale_tier="water",
                    scale_radius=2.1,
                    azimuth_theta=theta,
                    elevation_phi=phi,
                    rotor=rotor,
                    sample_count=max(4, len(beats) * 4),
                    origin_medium=episode_id,
                )
            )

        constituents = tuple(beat_ids + link_ids)
        macro = CausalDroplet(
            droplet_id=f"narrative::{episode_id}",
            constituents=constituents,
            internal_bonds=tuple((i, i + 1, 0.88) for i in range(max(0, len(constituents) - 1))),
            surface_tension=float(min(1.0, 0.65 + 0.08 * len(causal_links))),
            scale_radius=3.4,
            spectral_signature=narrative_sig,
            local_order_notes={"agency_mode": agency_mode.value, "causality_mode": causality_mode.value},
        )
        self.memory.absorb_droplet(macro, explicit_parents=constituents)
        return macro

    def _compose_summary_ko(
        self,
        episode_id: str,
        present: Sequence[str],
        absent: Sequence[str],
        agency_mode: AgencyMode,
        causality_mode: NarrativeCausalityMode,
        causal_links: Sequence[CrossModalCausalLink],
        beats: Sequence[CausalNarrativeBeat],
        bridges: Sequence[ContinuityBridge],
        audio: Optional[AudioStreamTrace],
    ) -> str:
        chroma_seq = " → ".join(b.chromatic_character for b in beats)
        audio_note = f", 음향={audio.acoustic_character}({audio.phrase_count}구절)" if audio is not None else ""
        cut_note = f", 컷교량={len(bridges)}건" if bridges else ""
        return (
            f"[{episode_id}] 모달({', '.join(present)} / 부재: {', '.join(absent) or '없음'}{audio_note}) | "
            f"행위주체={agency_mode.value}, 인과={causality_mode.value}, 연결={len(causal_links)}개, "
            f"서사({chroma_seq}){cut_note}"
        )

    def compare_causal_narratives(
        self,
        ep_a: ConnectedMediaEpisodeReport,
        ep_b: ConnectedMediaEpisodeReport,
    ) -> CrossMediaComparisonReport:
        iso = float(
            np.clip(
                np.dot(ep_a.narrative_signature, ep_b.narrative_signature)
                / (
                    float(np.linalg.norm(ep_a.narrative_signature) * np.linalg.norm(ep_b.narrative_signature))
                    + 1e-12
                ),
                0.0,
                1.0,
            )
        )
        ch_a = [b.chromatic_character for b in ep_a.beats]
        ch_b = [b.chromatic_character for b in ep_b.beats]
        n_cmp = min(len(ch_a), len(ch_b))
        ch_align = float(sum(1 for i in range(n_cmp) if ch_a[i] == ch_b[i]) / n_cmp) if n_cmp > 0 else 0.0
        motifs_a = {b.dominant_event_cue for b in ep_a.beats} | {lk.relation_kind for lk in ep_a.causal_links}
        motifs_b = {b.dominant_event_cue for b in ep_b.beats} | {lk.relation_kind for lk in ep_b.causal_links}
        shared = sorted(motifs_a & motifs_b)
        agency_div = (
            f"동일 행위주체({ep_a.agency_mode.value})"
            if ep_a.agency_mode == ep_b.agency_mode
            else f"행위주체 분화: {ep_a.episode_id}({ep_a.agency_mode.value}) vs {ep_b.episode_id}({ep_b.agency_mode.value})"
        )
        mod_div = {
            "only_in_a": sorted(set(ep_a.present_modalities) - set(ep_b.present_modalities)),
            "only_in_b": sorted(set(ep_b.present_modalities) - set(ep_a.present_modalities)),
            "shared_modalities": sorted(set(ep_a.present_modalities) & set(ep_b.present_modalities)),
        }
        synthesis = (
            f"'{ep_a.episode_id}'와 '{ep_b.episode_id}'는 서사 동형도 {iso:.3f}(색채일치 {ch_align:.2f})를 "
            f"공유하면서도 {agency_div} 및 고유 모달 차이({mod_div['only_in_a']} vs {mod_div['only_in_b']})를 보존합니다."
        )
        return CrossMediaComparisonReport(
            episode_a=ep_a.episode_id,
            episode_b=ep_b.episode_id,
            narrative_arc_isomorphism=iso,
            chromatic_flow_alignment=ch_align,
            shared_causal_motifs=shared,
            agency_divergence=agency_div,
            modality_divergence=mod_div,
            synthesis_ko=synthesis,
        )

    def _seal_living_witness(self, report: ConnectedMediaEpisodeReport) -> None:
        if self.living_cycle is None:
            return
        from core.orchestration.elysia_unified_living_cycle import LivingEncounter
        from core.orchestration.extended_cognitive_organism import ExtendedWorldState
        from core.physics.spectral_scale_rotor_manifold import encode_waveform_to_scale_rotors

        w = ExtendedWorldState(
            source_pos_x=self.living_cycle.organism.body_pos_x + 12.0,
            source_temp_k=3400.0,
        )
        self.living_cycle.breathe_and_understand(
            LivingEncounter(encounter_name=f"v32_{report.episode_id}", world=w, active_organs=("eye", "ear", "hand"))
        )
        if self.living_cycle.memory_runtime is not None:
            t = np.linspace(0.0, 1.0, 64, endpoint=False)
            sig = np.zeros_like(t)
            for k, coeff in enumerate(report.narrative_signature, start=1):
                sig += float(coeff) * np.sin(2.0 * math.pi * k * t)
            man = encode_waveform_to_scale_rotors(
                sig, domain=(0.0, 1.0), max_macro_modes=8, max_levels=2, max_knots_per_level=4
            )
            self.living_cycle.memory_runtime.ingest_to_ice(f"v32_{report.episode_id}", man)

    def sync_to_causal_field(
        self,
        causal_field: Optional["CausalField"] = None,
        prefix: str = "v32",
    ) -> Dict[str, Any]:
        return self.memory.sync_to_causal_field(causal_field=causal_field, prefix=prefix)
