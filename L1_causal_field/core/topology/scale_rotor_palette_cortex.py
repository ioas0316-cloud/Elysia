"""
Scale-Rotor Palette Cortex: Micro-Meso-Macro Continuous Causal Knob Architecture
================================================================================

Instead of chopping reality into flat pixel arrays or opaque FFT bins:
1. MICRO LAYER (`RotorWave1D` & `PaletteLayerRotor`):
   Discovers the dominant chromatic/luminance ratios of a scene as continuous
   palette knobs `(R(t), G(t), B(t))` rotating smoothly over time — e.g. a
   color-state evolving `(15,10,15) -> (10,15,20) -> (10,25,20)` is held as ONE
   continuous phase-wave of RGB ratios, NOT discrete dot stamps.
2. MESO LAYER (`PaletteLayerRotor` trajectory waves):
   Tracks HOW each palette layer moves, scales, and pulses across space-time
   `(cx(t), cy(t), rx(t), ry(t), presence(t))` as continuous causal rotor waves,
   so any region's spatial trajectory is directly inspectable, time-scalable,
   reversible, and phase-couplable.
3. MACRO LAYER (`MacroScaleRotorCortex`):
   A persistent, continuously accumulating cognitive field (NEVER reset per clip!)
   that ingests a stream of real videos (`.mp4`), animations (`.gif`), and real
   human speech / music (`.wav`) into shared **Chromatic**, **Trajectory**, and
   **Acoustic Cadence** Rotor Axes.
   Cognition & Direction ("연출") = redistributing & phase-coupling these learned
   axes (e.g., turning RGB warmth/tempo/scale knobs, or driving World A's Color
   Wave with World B's Motion Trajectory locked to World C's Human Voice Cadence)
   so frames converge naturally from continuous knob fields in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Fundamental Continuous Rotor Wave (Micro & Meso Building Block)
# ---------------------------------------------------------------------------


@dataclass
class RotorWave1D:
    """
    A single continuous causal knob trajectory represented as a compact sum of
    rotating phase components: `v(t) = dc + sum_k amp_k * cos(2*pi*f_k*t + phi_k)`.
    Evaluates at ANY arbitrary continuous time `t` in O(K) with zero stepping.
    """

    dc: float
    freqs_hz: np.ndarray
    amps: np.ndarray
    phases: np.ndarray
    duration_s: float

    @classmethod
    def fit(cls, values: np.ndarray, duration_s: float, max_modes: int = 5) -> "RotorWave1D":
        y = np.asarray(values, dtype=np.float64).ravel()
        n = len(y)
        dur = max(1e-4, float(duration_s))
        if n == 0:
            return cls(0.0, np.zeros(0), np.zeros(0), np.zeros(0), dur)
        dc = float(np.mean(y))
        if n < 3 or float(np.std(y)) < 1e-9:
            return cls(dc, np.zeros(0), np.zeros(0), np.zeros(0), dur)

        spec = np.fft.rfft(y - dc)
        mags = np.abs(spec[1:])
        if len(mags) == 0:
            return cls(dc, np.zeros(0), np.zeros(0), np.zeros(0), dur)
        k_keep = max(1, min(max_modes, len(mags)))
        top = np.argsort(mags)[::-1][:k_keep]
        bins = top + 1

        freqs = bins.astype(np.float64) / dur
        scale = np.full(len(bins), 2.0 / float(n), dtype=np.float64)
        if n % 2 == 0:
            scale = np.where(bins == n // 2, 1.0 / float(n), scale)
        amps = np.abs(spec[bins]) * scale
        phases = np.angle(spec[bins])
        return cls(dc=dc, freqs_hz=freqs, amps=amps, phases=phases, duration_s=dur)

    def evaluate(
        self,
        times: np.ndarray,
        tempo_scale: float = 1.0,
        phase_shift_rad: float = 0.0,
        amp_scale: float = 1.0,
        dc_offset: float = 0.0,
    ) -> np.ndarray:
        t = np.asarray(times, dtype=np.float64)
        out = np.full(t.shape, self.dc + dc_offset, dtype=np.float64)
        if len(self.freqs_hz) > 0:
            arg = (
                2.0 * math.pi * np.outer(t.ravel() * tempo_scale, self.freqs_hz)
                + self.phases[None, :]
                + phase_shift_rad
            )
            out = out + (amp_scale * (np.cos(arg) @ self.amps)).reshape(t.shape)
        return out

    @property
    def parameter_count(self) -> int:
        return 1 + 3 * len(self.freqs_hz)

    def dynamic_energy(self) -> float:
        return float(np.sqrt(np.sum(0.5 * (self.amps ** 2)))) if len(self.amps) > 0 else 0.0


# ---------------------------------------------------------------------------
# 2. Micro (RGB Ratio Palette) + Meso (Spatial Trajectory) + Acoustic Cadence
# ---------------------------------------------------------------------------


@dataclass
class PaletteLayerRotor:
    """
    One discerned visual causal layer combining:
    - MICRO: continuous RGB-ratio knob waves `(r_wave, g_wave, b_wave)`
    - MESO : continuous spatial trajectory waves `(cx_wave, cy_wave, rx_wave, ry_wave, presence_wave)`
    """

    layer_id: str
    is_backdrop: bool
    r_wave: RotorWave1D
    g_wave: RotorWave1D
    b_wave: RotorWave1D
    cx_wave: RotorWave1D
    cy_wave: RotorWave1D
    rx_wave: RotorWave1D
    ry_wave: RotorWave1D
    presence_wave: RotorWave1D

    @property
    def parameter_count(self) -> int:
        return (
            self.r_wave.parameter_count
            + self.g_wave.parameter_count
            + self.b_wave.parameter_count
            + self.cx_wave.parameter_count
            + self.cy_wave.parameter_count
            + self.rx_wave.parameter_count
            + self.ry_wave.parameter_count
            + self.presence_wave.parameter_count
        )

    def sample_rgb(
        self,
        times: np.ndarray,
        rgb_gain: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        rgb_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        tempo_scale: float = 1.0,
    ) -> np.ndarray:
        """Evaluates continuous RGB knobs `(T, 3)` in [0, 255] via ratio convergence."""
        r = self.r_wave.evaluate(times, tempo_scale=tempo_scale) * rgb_gain[0] + rgb_bias[0]
        g = self.g_wave.evaluate(times, tempo_scale=tempo_scale) * rgb_gain[1] + rgb_bias[1]
        b = self.b_wave.evaluate(times, tempo_scale=tempo_scale) * rgb_gain[2] + rgb_bias[2]
        return np.clip(np.stack([r, g, b], axis=-1), 0.0, 255.0)

    def sample_trajectory(
        self,
        times: np.ndarray,
        tempo_scale: float = 1.0,
        spatial_scale: float = 1.0,
        phase_shift_rad: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        cx = np.clip(
            self.cx_wave.evaluate(times, tempo_scale=tempo_scale, phase_shift_rad=phase_shift_rad),
            0.0,
            1.0,
        )
        cy = np.clip(
            self.cy_wave.evaluate(times, tempo_scale=tempo_scale, phase_shift_rad=phase_shift_rad),
            0.0,
            1.0,
        )
        rx = np.clip(self.rx_wave.evaluate(times, tempo_scale=tempo_scale) * spatial_scale, 0.03, 0.95)
        ry = np.clip(self.ry_wave.evaluate(times, tempo_scale=tempo_scale) * spatial_scale, 0.03, 0.95)
        pr = np.clip(self.presence_wave.evaluate(times, tempo_scale=tempo_scale), 0.0, 1.0)
        return {"cx": cx, "cy": cy, "rx": rx, "ry": ry, "presence": pr}


@dataclass
class VisualPaletteRotorField:
    """Complete Micro-Meso layered causal palette representation of a visual stream."""

    world_id: str
    source_kind: str
    duration_s: float
    fps: float
    frame_count: int
    raw_scalar_count: int
    layers: List[PaletteLayerRotor]
    extract_ms: float

    @property
    def parameter_count(self) -> int:
        return max(1, sum(l.parameter_count for l in self.layers))

    @property
    def compression_ratio(self) -> float:
        return float(self.raw_scalar_count / max(1, self.parameter_count))

    def render_frames(
        self,
        times: np.ndarray,
        height: int = 64,
        width: int = 96,
        rgb_gain: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        rgb_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        tempo_scale: float = 1.0,
        spatial_scale: float = 1.0,
        phase_shift_rad: float = 0.0,
        cadence_modulation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Converges continuous Micro RGB-ratio knobs and Meso spatial-trajectory
        rotors into an RGB video `(T, H, W, 3)` — ZERO pixel-table stamping!
        """
        t_arr = np.asarray(times, dtype=np.float64)
        n_t = len(t_arr)
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[None, :, None]
        xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, None, :]

        canvas = np.zeros((n_t, height, width, 3), dtype=np.float32)
        mod = (
            np.asarray(cadence_modulation, dtype=np.float32).reshape(n_t, 1, 1)
            if cadence_modulation is not None and len(cadence_modulation) == n_t
            else None
        )

        for layer in self.layers:
            rgb = layer.sample_rgb(
                t_arr, rgb_gain=rgb_gain, rgb_bias=rgb_bias, tempo_scale=tempo_scale
            ).astype(np.float32)
            if layer.is_backdrop:
                grad = (0.85 + 0.30 * yy).astype(np.float32)
                if mod is not None:
                    # Subtle luminous breath modulation coupled to vocal/acoustic cadence
                    lum_breath = (0.88 + 0.24 * mod)[..., None]
                    canvas = np.clip(rgb[:, None, None, :] * grad[..., None] * lum_breath, 0.0, 255.0)
                else:
                    canvas = np.clip(rgb[:, None, None, :] * grad[..., None], 0.0, 255.0)
            else:
                traj = layer.sample_trajectory(
                    t_arr,
                    tempo_scale=tempo_scale,
                    spatial_scale=spatial_scale,
                    phase_shift_rad=phase_shift_rad,
                )
                cx = traj["cx"].astype(np.float32)[:, None, None]
                cy = traj["cy"].astype(np.float32)[:, None, None]
                rx = traj["rx"].astype(np.float32)[:, None, None]
                ry = traj["ry"].astype(np.float32)[:, None, None]
                pr = traj["presence"].astype(np.float32)[:, None, None]
                if mod is not None:
                    rx = np.clip(rx * (0.78 + 0.50 * mod), 0.03, 0.95)
                    ry = np.clip(ry * (0.78 + 0.50 * mod), 0.03, 0.95)
                    pr = np.clip(pr * (0.60 + 0.65 * mod), 0.0, 1.0)

                dist_sq = ((xx - cx) / np.maximum(rx, 0.03)) ** 2 + ((yy - cy) / np.maximum(ry, 0.03)) ** 2
                alpha = pr * np.exp(-0.5 * (dist_sq ** 1.35))
                a4 = alpha[..., None]
                canvas = (1.0 - a4) * canvas + a4 * rgb[:, None, None, :]

        return np.clip(canvas, 0.0, 255.0).astype(np.uint8)


@dataclass
class AcousticCadenceRotorField:
    """Continuous acoustic/vocal knob waves (Energy/Breath, Pitch/Brightness, Voicing)."""

    world_id: str
    duration_s: float
    sample_rate: int
    raw_sample_count: int
    energy_wave: RotorWave1D
    pitch_hz_wave: RotorWave1D
    voicing_wave: RotorWave1D
    pause_fraction: float
    extract_ms: float

    @property
    def parameter_count(self) -> int:
        return (
            self.energy_wave.parameter_count
            + self.pitch_hz_wave.parameter_count
            + self.voicing_wave.parameter_count
        )

    @property
    def compression_ratio(self) -> float:
        return float(max(1, self.raw_sample_count) / max(1, self.parameter_count))

    def synthesize_pcm(
        self,
        duration_s: Optional[float] = None,
        sample_rate: int = 16000,
        tempo_scale: float = 1.0,
        pitch_scale: float = 1.0,
    ) -> np.ndarray:
        """Synthesizes continuous audio directly from Energy, Pitch, and Voicing rotor knobs."""
        dur = float(duration_s if duration_s is not None else self.duration_s)
        n = max(128, int(round(dur * sample_rate)))
        n_ctrl = max(16, int(round(dur * 100.0)))
        t_ctrl = np.linspace(0.0, dur, n_ctrl, endpoint=False)
        env_c = np.clip(self.energy_wave.evaluate(t_ctrl, tempo_scale=tempo_scale), 0.0, 1.0)
        f0_c = np.clip(self.pitch_hz_wave.evaluate(t_ctrl, tempo_scale=tempo_scale) * pitch_scale, 75.0, 1200.0)
        voi_c = np.clip(self.voicing_wave.evaluate(t_ctrl, tempo_scale=tempo_scale), 0.0, 1.0)

        t_s = np.linspace(0.0, dur, n, endpoint=False)
        env = np.interp(t_s, t_ctrl, env_c)
        f0 = np.interp(t_s, t_ctrl, f0_c)
        voi = np.interp(t_s, t_ctrl, voi_c)

        phase = 2.0 * math.pi * np.cumsum(f0) / float(sample_rate)
        harmonic = 0.60 * np.sin(phase) + 0.28 * np.sin(2.0 * phase) + 0.12 * np.sin(3.0 * phase)
        breath = np.sin(phase * 3.73 + 0.5 * np.sin(phase * 1.618))
        pcm = env * (voi * harmonic + (1.0 - voi) * 0.35 * breath)
        mx = float(np.max(np.abs(pcm)))
        if mx > 1.0:
            pcm = pcm / mx
        return pcm.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Fast Vectorized Palette-Rotor Distillation from Real Video / GIF / WAV
# ---------------------------------------------------------------------------


def _load_rgb_tensor(
    source: Path | str | Sequence[np.ndarray],
    max_frames: int = 36,
    resize_wh: Tuple[int, int] = (96, 64),
    explicit_fps: Optional[float] = None,
) -> Tuple[np.ndarray, float, str]:
    if not isinstance(source, (str, Path)):
        raw = source if isinstance(source, np.ndarray) and source.ndim == 4 else list(source)
        step = max(1, len(raw) // max_frames) if len(raw) > max_frames else 1
        sampled = raw[::step][:max_frames]
        w, h = resize_wh
        out = []
        for fr in sampled:
            a = np.asarray(fr, dtype=np.uint8)
            if a.shape[0] != h or a.shape[1] != w:
                a = np.asarray(
                    Image.fromarray(a).convert("RGB").resize((w, h), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
            out.append(a)
        fps = float(explicit_fps or 12.0) / step
        return np.stack(out, axis=0), fps, "live_frames"

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {path}")
    if path.suffix.lower() == ".gif":
        frames: List[np.ndarray] = []
        durs: List[float] = []
        with Image.open(path) as im:
            n_tot = int(getattr(im, "n_frames", 1))
            step = max(1, n_tot // max_frames) if n_tot > max_frames else 1
            for i in range(0, n_tot, step):
                if len(frames) >= max_frames:
                    break
                im.seek(i)
                d = im.info.get("duration", 100)
                if isinstance(d, (int, float)) and d > 0:
                    durs.append(float(d))
                frames.append(
                    np.asarray(im.convert("RGB").resize(resize_wh, Image.Resampling.BILINEAR), dtype=np.uint8)
                )
        avg_ms = float(np.mean(durs)) if durs else 100.0
        fps = float(explicit_fps) if explicit_fps else (1000.0 / max(10.0, avg_ms * step))
        return np.stack(frames, axis=0), fps, f"gif:{path.name}"

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total // max_frames) if total > max_frames else 1
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
    if not frames:
        raise ValueError(f"Video yielded zero frames: {path}")
    return np.stack(frames, axis=0), float(explicit_fps or (native_fps / step)), f"video:{path.name}"


def distill_visual_palette_rotors(
    world_id: str,
    source: Path | str | Sequence[np.ndarray],
    max_frames: int = 36,
    explicit_fps: Optional[float] = None,
    max_modes: int = 5,
) -> VisualPaletteRotorField:
    """
    Distills a real video/GIF/frame-stack directly into a Micro-Meso
    Palette-Rotor Field (ZERO opaque pixel-table storage):
      - Layer 0 (Backdrop): Scene median ambient RGB-ratio wave over time.
      - Layer 1 (Warm Salient Foreground): High-salience pixels where R >= B.
      - Layer 2 (Cool Salient Foreground): High-salience pixels where B > R.
      - Layer 3 (Kinetic Focus): Regions of highest frame-to-frame motion impulse.
    """
    t0 = time.perf_counter()
    frames, fps, src_kind = _load_rgb_tensor(source, max_frames=max_frames, explicit_fps=explicit_fps)
    n_t, h, w, _ = frames.shape
    dur_s = float(n_t / max(1e-3, fps))
    rgb = frames.astype(np.float32)

    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[None, :, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, None, :]

    bg_rgb = np.median(rgb.reshape(n_t, -1, 3), axis=1)
    bg_layer = PaletteLayerRotor(
        layer_id=f"{world_id}::backdrop",
        is_backdrop=True,
        r_wave=RotorWave1D.fit(bg_rgb[:, 0], dur_s, max_modes=max_modes),
        g_wave=RotorWave1D.fit(bg_rgb[:, 1], dur_s, max_modes=max_modes),
        b_wave=RotorWave1D.fit(bg_rgb[:, 2], dur_s, max_modes=max_modes),
        cx_wave=RotorWave1D(0.5, np.zeros(0), np.zeros(0), np.zeros(0), dur_s),
        cy_wave=RotorWave1D(0.5, np.zeros(0), np.zeros(0), np.zeros(0), dur_s),
        rx_wave=RotorWave1D(1.0, np.zeros(0), np.zeros(0), np.zeros(0), dur_s),
        ry_wave=RotorWave1D(1.0, np.zeros(0), np.zeros(0), np.zeros(0), dur_s),
        presence_wave=RotorWave1D(1.0, np.zeros(0), np.zeros(0), np.zeros(0), dur_s),
    )

    sal = np.linalg.norm(rgb - bg_rgb[:, None, None, :], axis=-1)
    sal_thresh = np.maximum(18.0, np.percentile(sal.reshape(n_t, -1), 78, axis=1))[:, None, None]
    warm_score = rgb[..., 0] - rgb[..., 2]

    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    mdiff = np.zeros_like(gray)
    if n_t > 1:
        mdiff[1:] = np.abs(gray[1:] - gray[:-1])
        mdiff[0] = mdiff[1]
    m_thresh = np.maximum(6.0, np.percentile(mdiff.reshape(n_t, -1), 85, axis=1))[:, None, None]

    masks = [
        ("warm_salient", np.where((sal >= sal_thresh) & (warm_score >= 0.0), sal, 0.0)),
        ("cool_salient", np.where((sal >= sal_thresh) & (warm_score < 0.0), sal, 0.0)),
        ("kinetic_focus", np.where(mdiff >= m_thresh, mdiff, 0.0)),
    ]

    layers: List[PaletteLayerRotor] = [bg_layer]
    for name, w_map in masks:
        w_sum = np.sum(w_map, axis=(1, 2))
        active_t = w_sum > 1e-3
        if not np.any(active_t):
            continue

        safe_w = np.maximum(w_sum, 1e-6)
        r_t = np.sum(w_map * rgb[..., 0], axis=(1, 2)) / safe_w
        g_t = np.sum(w_map * rgb[..., 1], axis=(1, 2)) / safe_w
        b_t = np.sum(w_map * rgb[..., 2], axis=(1, 2)) / safe_w
        cx_t = np.sum(w_map * xx, axis=(1, 2)) / safe_w
        cy_t = np.sum(w_map * yy, axis=(1, 2)) / safe_w
        var_x = np.sum(w_map * ((xx - cx_t[:, None, None]) ** 2), axis=(1, 2)) / safe_w
        var_y = np.sum(w_map * ((yy - cy_t[:, None, None]) ** 2), axis=(1, 2)) / safe_w
        rx_t = np.clip(np.sqrt(np.maximum(var_x, 1e-4)) * 1.45, 0.04, 0.65)
        ry_t = np.clip(np.sqrt(np.maximum(var_y, 1e-4)) * 1.45, 0.04, 0.65)
        area_frac = np.mean(w_map > 0.0, axis=(1, 2))
        pres_t = np.clip(np.where(active_t, 0.35 + 2.2 * np.sqrt(area_frac), 0.0), 0.0, 0.95)

        first_ok = int(np.argmax(active_t))
        for arr in (r_t, g_t, b_t, cx_t, cy_t, rx_t, ry_t):
            last_v = float(arr[first_ok])
            for i in range(n_t):
                if active_t[i]:
                    last_v = float(arr[i])
                else:
                    arr[i] = last_v

        layers.append(
            PaletteLayerRotor(
                layer_id=f"{world_id}::{name}",
                is_backdrop=False,
                r_wave=RotorWave1D.fit(r_t, dur_s, max_modes=max_modes),
                g_wave=RotorWave1D.fit(g_t, dur_s, max_modes=max_modes),
                b_wave=RotorWave1D.fit(b_t, dur_s, max_modes=max_modes),
                cx_wave=RotorWave1D.fit(cx_t, dur_s, max_modes=max_modes),
                cy_wave=RotorWave1D.fit(cy_t, dur_s, max_modes=max_modes),
                rx_wave=RotorWave1D.fit(rx_t, dur_s, max_modes=max_modes),
                ry_wave=RotorWave1D.fit(ry_t, dur_s, max_modes=max_modes),
                presence_wave=RotorWave1D.fit(pres_t, dur_s, max_modes=max_modes),
            )
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return VisualPaletteRotorField(
        world_id=world_id,
        source_kind=src_kind,
        duration_s=dur_s,
        fps=float(fps),
        frame_count=n_t,
        raw_scalar_count=int(frames.size),
        layers=layers,
        extract_ms=elapsed_ms,
    )


def distill_acoustic_cadence_rotors(
    world_id: str,
    source: Path | str | Tuple[np.ndarray, int],
    max_modes: int = 8,
) -> AcousticCadenceRotorField:
    """Distills a real WAV or PCM waveform into continuous Energy, Pitch, and Voicing rotor waves."""
    t0 = time.perf_counter()
    if isinstance(source, (str, Path)):
        from scipy.io import wavfile

        sr, data = wavfile.read(str(source))
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            arr = arr / float(max(abs(info.min), info.max))
    else:
        arr, sr = source
        arr = np.asarray(arr, dtype=np.float64).ravel()

    pk = float(np.max(np.abs(arr))) + 1e-12
    x = arr / pk
    dur_s = float(len(x) / float(sr))

    n_steps = max(16, min(160, int(dur_s * 20.0)))
    win_n = max(128, int(sr * 0.06))
    n_fft_ac = 1 << int(math.ceil(math.log2(2 * win_n)))
    times = np.linspace(0.0, dur_s, n_steps, endpoint=False)
    centers = np.round(times * sr).astype(np.int64)
    starts = np.clip(centers - win_n // 2, 0, max(0, len(x) - win_n))
    pad_x = x if len(x) >= win_n else np.pad(x, (0, win_n - len(x)))
    segs = pad_x[starts[:, None] + np.arange(win_n, dtype=np.int64)[None, :]]

    rms = np.sqrt(np.mean(segs ** 2, axis=1))
    rms_norm = rms / (float(np.max(rms)) + 1e-9)

    s0 = segs - np.mean(segs, axis=1, keepdims=True)
    F = np.fft.rfft(s0, n=n_fft_ac, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=n_fft_ac, axis=1)[:, :win_n]
    ac_norm = ac / np.maximum(ac[:, :1], 1e-12)
    min_lag = max(2, int(sr / 420.0))
    max_lag = min(win_n - 2, int(sr / 85.0))
    if max_lag > min_lag:
        sub_ac = ac_norm[:, min_lag:max_lag]
        best_rel = np.argmax(sub_ac, axis=1)
        voicing = np.where(rms_norm > 0.08, np.clip(sub_ac[np.arange(n_steps), best_rel], 0.0, 1.0), 0.0)
        pitch_hz = float(sr) / np.maximum(1.0, (best_rel + min_lag).astype(np.float64))
    else:
        voicing = np.zeros(n_steps)
        pitch_hz = np.full(n_steps, 180.0)

    pause_frac = float(np.mean(rms_norm < 0.14))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return AcousticCadenceRotorField(
        world_id=world_id,
        duration_s=dur_s,
        sample_rate=int(sr),
        raw_sample_count=len(x),
        energy_wave=RotorWave1D.fit(rms_norm, dur_s, max_modes=max_modes),
        pitch_hz_wave=RotorWave1D.fit(pitch_hz, dur_s, max_modes=max_modes),
        voicing_wave=RotorWave1D.fit(voicing, dur_s, max_modes=max_modes),
        pause_fraction=pause_frac,
        extract_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# 4. Persistent Macro Scale-Rotor Palette Cortex (Continuous Accumulation & Direction)
# ---------------------------------------------------------------------------


@dataclass
class DirectedCompositionResult:
    """Result of cross-world axis redistribution ("색채 축 + 궤적 축 + 음성 리듬 축 재분배 연출")."""

    composition_id: str
    palette_source_id: str
    motion_source_id: str
    cadence_source_id: Optional[str]
    frames: np.ndarray
    pcm: Optional[np.ndarray]
    fps: float
    sample_rate: int
    render_ms: float
    mean_rgb: Tuple[float, float, float]
    motion_energy_mean: float
    cadence_visual_coupling: float
    summary_ko: str


class MacroScaleRotorCortex:
    """
    Persistent living cortex that continuously accumulates real visual & acoustic
    worlds into shared Micro-Meso-Macro Palette & Trajectory Rotor Axes — NEVER
    resetting between files — and directs/creates new worlds strictly by
    redistributing and phase-coupling those axes!
    """

    def __init__(self) -> None:
        self.visual_worlds: Dict[str, VisualPaletteRotorField] = {}
        self.acoustic_worlds: Dict[str, AcousticCadenceRotorField] = {}
        self.ingestion_history: List[str] = []

    def absorb_visual_stream(
        self,
        world_id: str,
        source: Path | str | Sequence[np.ndarray],
        max_frames: int = 36,
        explicit_fps: Optional[float] = None,
        max_modes: int = 5,
    ) -> VisualPaletteRotorField:
        vf = distill_visual_palette_rotors(
            world_id=world_id,
            source=source,
            max_frames=max_frames,
            explicit_fps=explicit_fps,
            max_modes=max_modes,
        )
        self.visual_worlds[world_id] = vf
        self.ingestion_history.append(f"visual:{world_id}")
        return vf

    def absorb_acoustic_stream(
        self,
        world_id: str,
        source: Path | str | Tuple[np.ndarray, int],
        max_modes: int = 8,
    ) -> AcousticCadenceRotorField:
        af = distill_acoustic_cadence_rotors(world_id=world_id, source=source, max_modes=max_modes)
        self.acoustic_worlds[world_id] = af
        self.ingestion_history.append(f"acoustic:{world_id}")
        return af

    def absorb_workspace_media_curriculum(self, media_specs: Sequence[Tuple[str, str, Path]]) -> Dict[str, Any]:
        """Continuously ingests a whole sequence of real videos, GIFs, and WAVs into ONE living cortex."""
        t0 = time.perf_counter()
        absorbed_v, absorbed_a, skipped = 0, 0, 0
        for kind, wid, path in media_specs:
            if not path.is_file():
                skipped += 1
                continue
            if kind == "visual":
                self.absorb_visual_stream(wid, path)
                absorbed_v += 1
            elif kind == "acoustic":
                self.absorb_acoustic_stream(wid, path)
                absorbed_a += 1
        return {
            "visual_worlds_absorbed": absorbed_v,
            "acoustic_worlds_absorbed": absorbed_a,
            "skipped": skipped,
            "total_visual_worlds": len(self.visual_worlds),
            "total_acoustic_worlds": len(self.acoustic_worlds),
            "total_elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    def direct_cross_world_composition(
        self,
        composition_id: str,
        palette_world_id: str,
        motion_world_id: str,
        cadence_world_id: Optional[str] = None,
        n_frames: int = 36,
        fps: float = 12.0,
        height: int = 64,
        width: int = 96,
        rgb_gain: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        rgb_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        tempo_scale: float = 1.0,
    ) -> DirectedCompositionResult:
        """
        TRUE AXIS REDISTRIBUTION DIRECTION ("축의 재분배 연출"):
        Takes:
          - Micro RGB-ratio color waves from `palette_world_id` (World A),
          - Meso spatial-trajectory waves from `motion_world_id` (World B),
          - Acoustic/vocal breath-cadence wave from `cadence_world_id` (World C),
        binds them into a unified hybrid Palette-Rotor Field, and renders a brand-new
        continuous audiovisual stream purely by knob convergence!
        """
        t0 = time.perf_counter()
        p_world = self.visual_worlds[palette_world_id]
        m_world = self.visual_worlds[motion_world_id]
        c_world = self.acoustic_worlds.get(cadence_world_id) if cadence_world_id else None

        dur_s = float(n_frames / max(1e-3, fps))
        times = np.arange(n_frames, dtype=np.float64) / max(1e-3, fps)

        hybrid_layers: List[PaletteLayerRotor] = []
        n_layers = min(len(p_world.layers), len(m_world.layers))
        for i in range(n_layers):
            pl = p_world.layers[i]
            ml = m_world.layers[i]
            hybrid_layers.append(
                PaletteLayerRotor(
                    layer_id=f"{composition_id}::layer_{i}",
                    is_backdrop=pl.is_backdrop,
                    r_wave=pl.r_wave,
                    g_wave=pl.g_wave,
                    b_wave=pl.b_wave,
                    cx_wave=ml.cx_wave,
                    cy_wave=ml.cy_wave,
                    rx_wave=ml.rx_wave,
                    ry_wave=ml.ry_wave,
                    presence_wave=ml.presence_wave,
                )
            )

        hybrid_field = VisualPaletteRotorField(
            world_id=composition_id,
            source_kind=f"directed({palette_world_id}+{motion_world_id}+{cadence_world_id})",
            duration_s=dur_s,
            fps=fps,
            frame_count=n_frames,
            raw_scalar_count=n_frames * height * width * 3,
            layers=hybrid_layers,
            extract_ms=0.0,
        )

        cadence_mod: Optional[np.ndarray] = None
        pcm: Optional[np.ndarray] = None
        if c_world is not None:
            cadence_mod = np.clip(c_world.energy_wave.evaluate(times, tempo_scale=tempo_scale), 0.0, 1.0)
            pcm = c_world.synthesize_pcm(duration_s=dur_s, sample_rate=16000, tempo_scale=tempo_scale)

        frames = hybrid_field.render_frames(
            times=times,
            height=height,
            width=width,
            rgb_gain=rgb_gain,
            rgb_bias=rgb_bias,
            tempo_scale=tempo_scale,
            cadence_modulation=cadence_mod,
        )

        f_float = frames.astype(np.float64)
        mean_rgb = (
            float(np.mean(f_float[..., 0])),
            float(np.mean(f_float[..., 1])),
            float(np.mean(f_float[..., 2])),
        )
        gray = 0.299 * f_float[..., 0] + 0.587 * f_float[..., 1] + 0.114 * f_float[..., 2]
        frame_motion = np.mean(np.abs(gray[1:] - gray[:-1]), axis=(1, 2)) if n_frames > 1 else np.zeros(1)
        motion_mean = float(np.mean(frame_motion))

        cad_coupling = 0.0
        if cadence_mod is not None and len(cadence_mod) == n_frames:
            lum_t = np.mean(gray, axis=(1, 2))
            c0 = cadence_mod - float(np.mean(cadence_mod))
            l0 = lum_t - float(np.mean(lum_t))
            den = float(np.linalg.norm(c0) * np.linalg.norm(l0))
            if den > 1e-9:
                cad_coupling = float(abs(np.dot(c0, l0)) / den)

        render_ms = (time.perf_counter() - t0) * 1000.0
        summary = (
            f"[{composition_id}] '{palette_world_id}'의 색채 비율 파형 + '{motion_world_id}'의 공간 궤적 로터"
            + (f" + '{cadence_world_id}'의 음성/음향 리듬" if cadence_world_id else "")
            + f"을 축 재분배 결합해 {n_frames}프레임({render_ms:.1f}ms)을 연속 노브 수렴으로 연출했습니다 "
            f"(평균 RGB=({mean_rgb[0]:.1f},{mean_rgb[1]:.1f},{mean_rgb[2]:.1f}), 동적변화={motion_mean:.2f}, 리듬결합={cad_coupling:.3f})."
        )
        return DirectedCompositionResult(
            composition_id=composition_id,
            palette_source_id=palette_world_id,
            motion_source_id=motion_world_id,
            cadence_source_id=cadence_world_id,
            frames=frames,
            pcm=pcm,
            fps=fps,
            sample_rate=16000,
            render_ms=render_ms,
            mean_rgb=mean_rgb,
            motion_energy_mean=motion_mean,
            cadence_visual_coupling=cad_coupling,
            summary_ko=summary,
        )
