"""
Universal Knob Cortex: Domain-Agnostic Self-Growing Cognitive Knob Architecture
==============================================================================

Instead of hardcoding domain-specific extractors ("color", "pitch", "motion"),
Elysia starts with **ZERO knobs** (`knobs = {}`) and applies ONE identical
recursive cognitive law across every continuous stream `X` of shape `(T, D)`:

1. DISCERN & REUSE (분별과 재사용):
   Measure how much of the current world stream is already explained by
   already-internalized knobs `B` in space `D`. Reuse matching internalized knobs
   and expose the unexplained causal residual `R`.
2. CREATE & INTERNALIZE (자생과 내적화):
   When the unexplained residual exceeds tolerance, distill the dominant coherent
   axis `u` from `R` via SVD + Modified Gram-Schmidt orthogonalization, fit a
   continuous temporal `RotorWave1D` to how that axis evolves over time, and
   internalize it as a new `SelfDiscoveredKnob`.
3. CONNECT ACROSS WORLDS & MODALITIES (교차 연결):
   Compute lead-lag phase resonance across any co-occurring knob trajectories
   (e.g., Vision <-> Audio <-> Language <-> Action) to form `KnobConnection` links.
4. THINK & STEER IN KNOB SPACE (노브 공간 사고 및 재창조):
   Given any target horizon/state or associative cross-modal impulse, compute
   or propagate knob adjustments directly inside internalized knob space and
   persist the grown cortex to disk without forgetting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from core.topology.scale_rotor_palette_cortex import RotorWave1D


# ---------------------------------------------------------------------------
# 1. Core Data Structures
# ---------------------------------------------------------------------------


@dataclass
class SelfDiscoveredKnob:
    """One autonomous causal knob discovered and internalized from lived residuals."""

    knob_id: str
    origin_stream_id: str
    space_dim: int
    basis_vec: np.ndarray  # unit vector of shape (D,)
    mean_anchor: np.ndarray  # space anchor of shape (D,)
    temporal_rotor: RotorWave1D
    value_min: float
    value_max: float
    explained_variance: float
    reuse_count: int = 0

    def effect_of_turning(self, delta: float) -> np.ndarray:
        """Vector delta in `D`-dimensional sensory/structural space per knob turn."""
        return float(delta) * self.basis_vec


@dataclass
class KnobConnection:
    """Directed lead-lag resonance connection between two internalized knobs."""

    src_knob_id: str
    dst_knob_id: str
    best_lag_steps: int
    resonance_corr: float
    transfer_gain: float


@dataclass
class KnobGrowthReport:
    """Measured discernment, reuse, growth, and reconstruction of one experience."""

    stream_id: str
    space_dim: int
    explained_before: float
    explained_after: float
    reused_knob_ids: List[str]
    created_knob_ids: List[str]
    knob_trajectories: Dict[str, np.ndarray]
    reconstructed_stream: np.ndarray
    elapsed_ms: float


# ---------------------------------------------------------------------------
# 2. Thin Outer Sensory-Membrane Adapters (Raw Bytes -> Normalized (T, D))
# ---------------------------------------------------------------------------


def adapt_video_to_raw_stream(
    source: Path | str | Sequence[np.ndarray] | np.ndarray,
    max_frames: int = 36,
    grid_hw: Tuple[int, int] = (16, 24),
    explicit_fps: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Thin sensory membrane converting `.mp4`, `.gif`, or raw RGB frames into
    a flat normalized `(T, H*W*3)` float64 stream in `[0.0, 1.0]`.
    """
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    target_wh = (gw, gh)

    frames: List[np.ndarray] = []
    fps = float(explicit_fps or 12.0)

    if not isinstance(source, (str, Path)):
        raw = source if isinstance(source, np.ndarray) and source.ndim == 4 else list(source)
        if len(raw) == 0:
            raise ValueError("Empty video frame sequence")
        step = max(1, len(raw) // max_frames) if len(raw) > max_frames else 1
        sampled = raw[::step][:max_frames]
        for fr in sampled:
            a = np.asarray(fr, dtype=np.uint8)
            if a.shape[0] != gh or a.shape[1] != gw:
                a = np.asarray(
                    Image.fromarray(a).convert("RGB").resize(target_wh, Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
            frames.append(a)
        fps = float(explicit_fps or 12.0) / step
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Video/GIF file not found: {path}")
        if path.suffix.lower() == ".gif":
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
                        np.asarray(im.convert("RGB").resize(target_wh, Image.Resampling.BILINEAR), dtype=np.uint8)
                    )
            avg_ms = float(np.mean(durs)) if durs else 100.0
            fps = float(explicit_fps) if explicit_fps else (1000.0 / max(10.0, avg_ms * step))
        else:
            import cv2

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV failed to open video: {path}")
            try:
                native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                step = max(1, total // max_frames) if total > max_frames else 1
                idx = 0
                while True:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    if idx % step == 0:
                        rgb = cv2.cvtColor(cv2.resize(bgr, target_wh, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                        frames.append(rgb)
                        if len(frames) >= max_frames:
                            break
                    idx += 1
            finally:
                cap.release()
            fps = float(explicit_fps) if explicit_fps else (native_fps / step)

    if not frames:
        raise ValueError("Video source produced zero frames")
    arr = np.stack(frames, axis=0).astype(np.float64) / 255.0
    n_t = arr.shape[0]
    dur_s = float(n_t / max(1e-3, fps))
    return arr.reshape(n_t, gh * gw * 3), dur_s


def adapt_audio_to_raw_stream(
    source: Path | str | Tuple[np.ndarray, int],
    n_steps: int = 48,
    n_bands: int = 32,
) -> Tuple[np.ndarray, float]:
    """
    Thin acoustic membrane converting `.wav` or PCM samples into a `(T, n_bands)`
    log-magnitude spectral band stream.
    """
    if isinstance(source, (str, Path)):
        from scipy.io import wavfile

        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
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
    peak = float(np.max(np.abs(arr))) + 1e-12
    x = arr / peak
    dur_s = float(len(x) / float(sr))

    win_n = max(128, int(sr * 0.05))
    hann = np.hanning(win_n)
    times = np.linspace(0.0, dur_s, n_steps, endpoint=False)
    centers = np.round(times * sr).astype(np.int64)
    starts = np.clip(centers - win_n // 2, 0, max(0, len(x) - win_n))
    pad_x = x if len(x) >= win_n else np.pad(x, (0, win_n - len(x)))
    segs = pad_x[starts[:, None] + np.arange(win_n, dtype=np.int64)[None, :]]

    mag = np.abs(np.fft.rfft(segs * hann[None, :], axis=1))  # (T, F)
    # Pool FFT bins evenly into `n_bands` continuous spectral log-amplitude channels
    usable = (mag.shape[1] // n_bands) * n_bands
    if usable >= n_bands:
        bands = mag[:, :usable].reshape(n_steps, n_bands, usable // n_bands).mean(axis=2)
    else:
        bands = np.pad(mag, ((0, 0), (0, n_bands - mag.shape[1])))[:, :n_bands]
    log_bands = np.log1p(bands * 4.0)
    return log_bands.astype(np.float64), dur_s


def adapt_text_to_raw_stream(
    utterances_or_chunks: Sequence[str],
    n_buckets: int = 32,
    seconds_per_chunk: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    Vocabulary-free hashed character 2-gram density stream `(T, n_buckets)`.
    Zero hardcoded dictionaries — handles any natural or synthetic text uniformly.
    """
    chunks = list(utterances_or_chunks)
    if not chunks:
        raise ValueError("Empty text sequence provided")
    n_t = len(chunks)
    mat = np.zeros((n_t, n_buckets), dtype=np.float64)
    for i, text in enumerate(chunks):
        s = text.strip().lower()
        if not s:
            continue
        if len(s) == 1:
            idx = int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little") % n_buckets
            mat[i, idx] += 1.0
        else:
            for k in range(len(s) - 1):
                bg = s[k : k + 2].encode("utf-8")
                idx = int.from_bytes(hashlib.blake2b(bg, digest_size=4).digest(), "little") % n_buckets
                mat[i, idx] += 1.0
        row_norm = float(np.linalg.norm(mat[i]))
        if row_norm > 1e-12:
            mat[i] /= row_norm
    dur_s = float(max(0.5, n_t * seconds_per_chunk))
    return mat, dur_s


# ---------------------------------------------------------------------------
# 3. Universal Self-Growing Knob Cortex
# ---------------------------------------------------------------------------


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ac = a - float(np.mean(a))
    bc = b - float(np.mean(b))
    den = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if den <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(ac, bc) / den, -1.0, 1.0))


class UniversalKnobCortex:
    """
    Domain-agnostic cognitive architecture starting with ZERO knobs (`{}`).
    Perceives any continuous `(T, D)` stream, reuses internalized knobs where they
    fit, spontaneously distills new orthogonal causal knobs from unexplained
    residuals, links knobs across spaces via lead-lag resonance, thinks/steers
    inside its own knob space, and persists grown knobs & connections to disk.
    """

    def __init__(self) -> None:
        self.knobs: Dict[str, SelfDiscoveredKnob] = {}
        self.connections: List[KnobConnection] = []
        self.space_anchors: Dict[int, np.ndarray] = {}
        self._knob_counter: int = 0
        self._conn_keys: set[Tuple[str, str]] = set()

    def _knobs_for_dim(self, space_dim: int) -> List[SelfDiscoveredKnob]:
        return [k for k in self.knobs.values() if k.space_dim == space_dim]

    def perceive_and_grow(
        self,
        stream_id: str,
        raw_stream: np.ndarray,
        duration_s: float,
        target_explained_ratio: float = 0.88,
        max_new_knobs: int = 6,
        max_modes: int = 5,
    ) -> KnobGrowthReport:
        t0 = time.perf_counter()
        X = np.asarray(raw_stream, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
            raise ValueError(f"Expected 2D stream (T>=2, D>=1), got {X.shape}")

        n_t, dim = X.shape
        dur = max(1e-4, float(duration_s))

        if dim not in self.space_anchors:
            # Natural physical zero origin — never lock all future worlds to Stream 1's static backdrop!
            self.space_anchors[dim] = np.zeros(dim, dtype=np.float64)
        anchor = self.space_anchors[dim]
        Xc = X - anchor[None, :]
        total_var = float(np.sum(Xc ** 2))
        if total_var <= 1e-12:
            return KnobGrowthReport(
                stream_id=stream_id,
                space_dim=dim,
                explained_before=1.0,
                explained_after=1.0,
                reused_knob_ids=[],
                created_knob_ids=[],
                knob_trajectories={},
                reconstructed_stream=X.copy(),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        X_dyn = Xc - np.mean(Xc, axis=0, keepdims=True)
        dyn_var = float(np.sum(X_dyn ** 2))
        has_dynamics = dyn_var > 1e-8 * max(1.0, total_var)

        # 1. Measure how much existing internalized knobs explain BEFORE growing new ones
        existing = self._knobs_for_dim(dim)
        R = Xc.copy()
        reused_ids: List[str] = []

        if existing:
            B = np.stack([k.basis_vec for k in existing], axis=0)  # (K, D)
            coeffs, *_ = np.linalg.lstsq(B.T, Xc.T, rcond=1e-8)
            coeffs = coeffs.T  # (T, K)
            X_hat_prev = coeffs @ B
            R = Xc - X_hat_prev
            raw_explained = float(np.clip(1.0 - float(np.sum(R ** 2)) / total_var, 0.0, 1.0))

            for idx, k in enumerate(existing):
                k_energy = float(np.sum(coeffs[:, idx] ** 2) * float(np.dot(k.basis_vec, k.basis_vec)))
                if k_energy / total_var > 0.01:
                    k.reuse_count += 1
                    reused_ids.append(k.knob_id)
        else:
            raw_explained = 0.0
        explained_before = raw_explained

        R_dyn = R - np.mean(R, axis=0, keepdims=True)
        dyn_explained = (
            float(np.clip(1.0 - float(np.sum(R_dyn ** 2)) / dyn_var, 0.0, 1.0)) if has_dynamics else 1.0
        )

        # 2. Spontaneously birth & internalize new orthogonal knobs covering BOTH scene structure & temporal dynamics
        created_ids: List[str] = []
        basis_pool: List[np.ndarray] = [k.basis_vec.copy() for k in existing]

        while (
            raw_explained < target_explained_ratio
            or (has_dynamics and dyn_explained < target_explained_ratio)
        ) and len(created_ids) < max_new_knobs:
            svd_target = R if raw_explained < target_explained_ratio else R_dyn
            _u_left, _s_vals, vh = np.linalg.svd(svd_target, full_matrices=False)
            if vh.shape[0] == 0:
                break
            u = vh[0].astype(np.float64)

            # Modified Gram-Schmidt orthogonalization against all existing D-dim knobs
            for prev_u in basis_pool:
                u = u - float(np.dot(u, prev_u)) * prev_u
            u_norm = float(np.linalg.norm(u))
            if u_norm < 1e-4:
                break
            u = u / u_norm

            proj = R @ u  # temporal activation along knob `u`
            proj_rms = float(np.sqrt(np.mean(proj ** 2)))
            if proj_rms < 1e-4:
                break

            knob_var = float(np.sum(proj ** 2))
            self._knob_counter += 1
            kid = f"knob_d{dim}_{self._knob_counter:03d}::{stream_id}"
            rotor = RotorWave1D.fit(proj, dur, max_modes=max_modes)
            new_knob = SelfDiscoveredKnob(
                knob_id=kid,
                origin_stream_id=stream_id,
                space_dim=dim,
                basis_vec=u.copy(),
                mean_anchor=anchor.copy(),
                temporal_rotor=rotor,
                value_min=float(np.min(proj)),
                value_max=float(np.max(proj)),
                explained_variance=knob_var,
                reuse_count=0,
            )
            self.knobs[kid] = new_knob
            basis_pool.append(u.copy())
            created_ids.append(kid)

            R = R - np.outer(proj, u)
            raw_explained = float(np.clip(1.0 - float(np.sum(R ** 2)) / total_var, 0.0, 1.0))
            R_dyn = R - np.mean(R, axis=0, keepdims=True)
            dyn_explained = (
                float(np.clip(1.0 - float(np.sum(R_dyn ** 2)) / dyn_var, 0.0, 1.0)) if has_dynamics else 1.0
            )

        # 3. Compute final trajectories for all internalized D-dim knobs & rebuild stream
        final_knobs = self._knobs_for_dim(dim)
        B_final = np.stack([k.basis_vec for k in final_knobs], axis=0)
        coeffs_final, *_ = np.linalg.lstsq(B_final.T, Xc.T, rcond=1e-8)
        coeffs_final = coeffs_final.T  # (T, K_final)
        reconstructed = anchor[None, :] + (coeffs_final @ B_final)
        explained_after = float(np.clip(1.0 - float(np.sum((X - reconstructed) ** 2)) / total_var, 0.0, 1.0))

        traj_map: Dict[str, np.ndarray] = {}
        for idx, k in enumerate(final_knobs):
            # Keep trajectories that either were newly created, reused, or carry meaningful variance
            tr = coeffs_final[:, idx]
            if k.knob_id in created_ids or k.knob_id in reused_ids or float(np.std(tr)) > 1e-5:
                traj_map[k.knob_id] = tr.copy()

        return KnobGrowthReport(
            stream_id=stream_id,
            space_dim=dim,
            explained_before=explained_before,
            explained_after=explained_after,
            reused_knob_ids=reused_ids,
            created_knob_ids=created_ids,
            knob_trajectories=traj_map,
            reconstructed_stream=reconstructed,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def connect_cooccurring_streams(
        self,
        traj_a: Dict[str, np.ndarray],
        traj_b: Dict[str, np.ndarray],
        max_lag: int = 4,
        min_corr: float = 0.45,
    ) -> List[KnobConnection]:
        """Discovers lead-lag resonance links across any two co-occurring knob sets."""
        new_conns: List[KnobConnection] = []
        for ka, va_raw in traj_a.items():
            for kb, vb_raw in traj_b.items():
                if ka == kb:
                    continue
                va = np.asarray(va_raw, dtype=np.float64).ravel()
                vb = np.asarray(vb_raw, dtype=np.float64).ravel()
                if len(va) != len(vb):
                    n_common = min(len(va), len(vb))
                    if n_common < 4:
                        continue
                    va = np.interp(np.linspace(0.0, 1.0, n_common), np.linspace(0.0, 1.0, len(va)), va)
                    vb = np.interp(np.linspace(0.0, 1.0, n_common), np.linspace(0.0, 1.0, len(vb)), vb)
                n = len(va)
                if n < 6 or float(np.std(va)) < 1e-8 or float(np.std(vb)) < 1e-8:
                    continue

                lim = max(1, min(max_lag, n // 4))
                best_lag, best_r = 0, 0.0
                for lag in range(-lim, lim + 1):
                    if lag > 0:
                        r = _safe_corr(va[:-lag], vb[lag:])
                    elif lag < 0:
                        r = _safe_corr(va[-lag:], vb[:lag])
                    else:
                        r = _safe_corr(va, vb)
                    if abs(r) > abs(best_r) + 1e-6:
                        best_r = r
                        best_lag = lag

                if abs(best_r) < min_corr:
                    continue

                if best_lag >= 0:
                    src_id, dst_id, lag_out = ka, kb, best_lag
                    x_s, y_s = (va[:-best_lag], vb[best_lag:]) if best_lag > 0 else (va, vb)
                else:
                    src_id, dst_id, lag_out = kb, ka, -best_lag
                    x_s, y_s = vb[:-lag_out], va[lag_out:]

                gain = float(np.cov(x_s, y_s, bias=True)[0, 1] / max(1e-12, float(np.var(x_s))))
                key = (src_id, dst_id)
                conn = KnobConnection(
                    src_knob_id=src_id,
                    dst_knob_id=dst_id,
                    best_lag_steps=int(lag_out),
                    resonance_corr=float(best_r),
                    transfer_gain=gain,
                )
                if key not in self._conn_keys:
                    self._conn_keys.add(key)
                    self.connections.append(conn)
                new_conns.append(conn)

        return new_conns

    def think_and_steer_to_target(
        self,
        space_dim: int,
        target_state: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Autonomously thinks INSIDE internalized knob space to find exact knob
        settings `a` such that `anchor + sum_k a_k * basis_k` converges toward
        `target_state` — with ZERO human-supplied gain!
        """
        knobs = self._knobs_for_dim(space_dim)
        target = np.asarray(target_state, dtype=np.float64).ravel()
        if not knobs or space_dim not in self.space_anchors or len(target) != space_dim:
            return {
                "knob_settings": {},
                "recreated_state": np.zeros(space_dim, dtype=np.float64),
                "relative_error": 1.0,
                "cosine_similarity": 0.0,
            }

        anchor = self.space_anchors[space_dim]
        delta_target = target - anchor
        B = np.stack([k.basis_vec for k in knobs], axis=0)  # (K, D)
        settings, *_ = np.linalg.lstsq(B.T, delta_target, rcond=1e-8)
        recreated = anchor + settings @ B

        t_norm = float(np.linalg.norm(target))
        r_norm = float(np.linalg.norm(recreated))
        rel_err = float(np.linalg.norm(target - recreated) / max(1e-12, t_norm))
        cos_sim = float(np.dot(target, recreated) / max(1e-12, t_norm * r_norm))
        return {
            "knob_settings": {k.knob_id: float(settings[i]) for i, k in enumerate(knobs)},
            "recreated_state": recreated,
            "relative_error": rel_err,
            "cosine_similarity": cos_sim,
        }

    def think_across_connected_knobs(
        self,
        source_activations: Dict[str, float],
    ) -> Dict[str, float]:
        """Propagates source knob turns across internalized `KnobConnection` web."""
        emergent: Dict[str, float] = {}
        for conn in self.connections:
            if conn.src_knob_id in source_activations:
                s_val = float(source_activations[conn.src_knob_id])
                delta = s_val * conn.transfer_gain * conn.resonance_corr
                emergent[conn.dst_knob_id] = emergent.get(conn.dst_knob_id, 0.0) + delta
        return emergent

    def save_state(self, path: Path | str) -> Path:
        """Persists all self-discovered knobs, rotor waves, anchors, and connections."""
        out_path = Path(path)
        arrays: Dict[str, np.ndarray] = {}
        knob_meta: List[Dict[str, Any]] = []

        for idx, k in enumerate(self.knobs.values()):
            p = f"k_{idx}"
            arrays[f"{p}_basis"] = k.basis_vec
            arrays[f"{p}_anchor"] = k.mean_anchor
            arrays[f"{p}_freqs"] = k.temporal_rotor.freqs_hz
            arrays[f"{p}_amps"] = k.temporal_rotor.amps
            arrays[f"{p}_phases"] = k.temporal_rotor.phases
            knob_meta.append(
                {
                    "idx": idx,
                    "knob_id": k.knob_id,
                    "origin_stream_id": k.origin_stream_id,
                    "space_dim": int(k.space_dim),
                    "dc": float(k.temporal_rotor.dc),
                    "duration_s": float(k.temporal_rotor.duration_s),
                    "value_min": float(k.value_min),
                    "value_max": float(k.value_max),
                    "explained_variance": float(k.explained_variance),
                    "reuse_count": int(k.reuse_count),
                }
            )

        for dim, anch in self.space_anchors.items():
            arrays[f"anchor_d{dim}"] = anch

        meta = {
            "knob_counter": int(self._knob_counter),
            "anchor_dims": [int(d) for d in self.space_anchors.keys()],
            "knobs": knob_meta,
            "connections": [
                {
                    "src_knob_id": c.src_knob_id,
                    "dst_knob_id": c.dst_knob_id,
                    "best_lag_steps": int(c.best_lag_steps),
                    "resonance_corr": float(c.resonance_corr),
                    "transfer_gain": float(c.transfer_gain),
                }
                for c in self.connections
            ],
        }
        arrays["_meta_json"] = np.array(json.dumps(meta))
        np.savez(str(out_path), **arrays)
        return out_path

    @classmethod
    def load_state(cls, path: Path | str) -> "UniversalKnobCortex":
        in_path = Path(path)
        cortex = cls()
        with np.load(str(in_path), allow_pickle=False) as z:
            meta = json.loads(str(z["_meta_json"].item()))
            cortex._knob_counter = int(meta["knob_counter"])
            for dim in meta["anchor_dims"]:
                cortex.space_anchors[int(dim)] = z[f"anchor_d{int(dim)}"].astype(np.float64)
            for km in meta["knobs"]:
                idx = int(km["idx"])
                p = f"k_{idx}"
                rotor = RotorWave1D(
                    dc=float(km["dc"]),
                    freqs_hz=z[f"{p}_freqs"].astype(np.float64),
                    amps=z[f"{p}_amps"].astype(np.float64),
                    phases=z[f"{p}_phases"].astype(np.float64),
                    duration_s=float(km["duration_s"]),
                )
                k = SelfDiscoveredKnob(
                    knob_id=str(km["knob_id"]),
                    origin_stream_id=str(km["origin_stream_id"]),
                    space_dim=int(km["space_dim"]),
                    basis_vec=z[f"{p}_basis"].astype(np.float64),
                    mean_anchor=z[f"{p}_anchor"].astype(np.float64),
                    temporal_rotor=rotor,
                    value_min=float(km["value_min"]),
                    value_max=float(km["value_max"]),
                    explained_variance=float(km["explained_variance"]),
                    reuse_count=int(km["reuse_count"]),
                )
                cortex.knobs[k.knob_id] = k
            for cm in meta["connections"]:
                conn = KnobConnection(
                    src_knob_id=str(cm["src_knob_id"]),
                    dst_knob_id=str(cm["dst_knob_id"]),
                    best_lag_steps=int(cm["best_lag_steps"]),
                    resonance_corr=float(cm["resonance_corr"]),
                    transfer_gain=float(cm["transfer_gain"]),
                )
                cortex.connections.append(conn)
                cortex._conn_keys.add((conn.src_knob_id, conn.dst_knob_id))
        return cortex
