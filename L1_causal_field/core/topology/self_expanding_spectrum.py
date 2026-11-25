r"""
ELYSIA Self-Expanding Spectrum Axis & Causal Mechanism Discovery Engine
========================================================================

Realizes cognitive expansion outward from innate sensory footholds (AGENTS.md §5-6):
  - Starts ONLY with the 3 physical sensory foothold axes (wavelength, kinetic, harmony).
  - Observes open-ended physical/media/software/body observables `Dict[str, float]`.
  - Detects when existing axes fail to explain structural variation (residual energy).
  - Extracts new orthogonal bipolar spectrum axes (`- <-> 0 <-> +`) via SVD over
    unexplained residuals, grounding `-` and `+` poles directly in extreme observed tags.
  - Learns `CausalMechanismLink` deltas along EVERY axis (what action/tool/process moves
    phenomena toward `-` vs `+`) and supports arbitrary-resolution spectrum slicing
    (`0.5 / 0.2 / 0.1 / 0.05`) plus dynamic `Cl(3,0)` context rotor projection.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.topology.geometric_rotor_docking import MV_DIM, normalize_multivector


@dataclass
class EmergentSpectrumAxis:
    """Single bipolar cognitive spectrum axis (`-1.0 <-> 0.0 <-> +1.0`)."""

    axis_id: str
    origin: str  # "innate_sensory" | "self_discovered"
    weights: Dict[str, float]
    center: Dict[str, float]
    scale: float
    neg_pole_tags: List[str] = field(default_factory=list)
    pos_pole_tags: List[str] = field(default_factory=list)
    explained_variance_gain: float = 0.0
    observations_count: int = 0

    def project(self, observables: Dict[str, float]) -> float:
        """Projects raw observables onto this bipolar spectrum (`-1.0 .. +1.0`)."""
        if not observables or not self.weights:
            return 0.0
        raw = 0.0
        matched = 0
        for feat, w in self.weights.items():
            if feat in observables:
                raw += (float(observables[feat]) - float(self.center.get(feat, 0.0))) * float(w)
                matched += 1
        if matched == 0:
            return 0.0
        return float(math.tanh(raw / max(1e-4, 1.4 * float(self.scale))))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EmergentSpectrumAxis":
        return cls(
            axis_id=str(raw.get("axis_id", "")),
            origin=str(raw.get("origin", "self_discovered")),
            weights={str(k): float(v) for k, v in (raw.get("weights") or {}).items()},
            center={str(k): float(v) for k, v in (raw.get("center") or {}).items()},
            scale=float(raw.get("scale", 1.0)),
            neg_pole_tags=[str(t) for t in raw.get("neg_pole_tags", [])],
            pos_pole_tags=[str(t) for t in raw.get("pos_pole_tags", [])],
            explained_variance_gain=float(raw.get("explained_variance_gain", 0.0)),
            observations_count=int(raw.get("observations_count", 0)),
        )


@dataclass
class CausalMechanismLink:
    """Learned causal relationship: how `cause_tag` shifts state along `axis_id`."""

    cause_tag: str
    axis_id: str
    mean_delta: float
    consistency: float
    samples: int
    _m2: float = 0.0  # running Welford sum-of-squares for delta variance

    def update(self, delta: float) -> None:
        self.samples += 1
        d1 = float(delta) - self.mean_delta
        self.mean_delta += d1 / float(self.samples)
        d2 = float(delta) - self.mean_delta
        self._m2 += d1 * d2
        std = math.sqrt(max(0.0, self._m2 / max(1, self.samples)))
        self.consistency = round(min(1.0, abs(self.mean_delta) / (std + 0.05)), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause_tag": self.cause_tag,
            "axis_id": self.axis_id,
            "mean_delta": round(self.mean_delta, 5),
            "consistency": round(self.consistency, 4),
            "samples": int(self.samples),
            "m2": round(self._m2, 6),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CausalMechanismLink":
        return cls(
            cause_tag=str(raw.get("cause_tag", "")),
            axis_id=str(raw.get("axis_id", "")),
            mean_delta=float(raw.get("mean_delta", 0.0)),
            consistency=float(raw.get("consistency", 0.0)),
            samples=int(raw.get("samples", 0)),
            _m2=float(raw.get("m2", 0.0)),
        )


def _default_innate_axes() -> List[EmergentSpectrumAxis]:
    return [
        EmergentSpectrumAxis(
            axis_id="axis::innate_wavelength",
            origin="innate_sensory",
            weights={"warm_cool_ratio": 0.92, "brightness": 0.39},
            center={"warm_cool_ratio": 1.0, "brightness": 0.5},
            scale=0.55,
            neg_pole_tags=["cool", "short_wave"],
            pos_pole_tags=["warm", "long_wave"],
        ),
        EmergentSpectrumAxis(
            axis_id="axis::innate_kinetic",
            origin="innate_sensory",
            weights={"motion_flux": 0.78, "rms_energy": 0.62},
            center={"motion_flux": 0.14, "rms_energy": 0.11},
            scale=0.18,
            neg_pole_tags=["still", "quiet"],
            pos_pole_tags=["kinetic", "loud"],
        ),
        EmergentSpectrumAxis(
            axis_id="axis::innate_harmony",
            origin="innate_sensory",
            weights={"valence": 0.85, "contrast_std": -0.35, "zcr": -0.39},
            center={"valence": 0.0, "contrast_std": 0.16, "zcr": 0.14},
            scale=0.55,
            neg_pole_tags=["tense", "friction"],
            pos_pole_tags=["resonant", "calm"],
        ),
    ]


class SelfExpandingSpectrumCortex:
    """
    Autonomous cognitive spectrum & causal mechanism expander.
    Starts from 3 innate physical axes and spawns new axes whenever external world
    observations exhibit coherent structural variance unexplained by existing axes.
    """

    MAX_BUFFER = 256

    def __init__(self) -> None:
        self.axes: List[EmergentSpectrumAxis] = _default_innate_axes()
        self.feature_vocab: List[str] = []
        self._obs_buffer: List[Dict[str, Any]] = []
        self.mechanisms: Dict[Tuple[str, str], CausalMechanismLink] = {}
        for ax in self.axes:
            for fk in ax.weights:
                if fk not in self.feature_vocab:
                    self.feature_vocab.append(fk)

    def discovered_axis_count(self) -> int:
        return sum(1 for ax in self.axes if ax.origin == "self_discovered")

    def get_axis(self, axis_id: str) -> Optional[EmergentSpectrumAxis]:
        for ax in self.axes:
            if ax.axis_id == axis_id:
                return ax
        return None

    def project(self, observables: Dict[str, float]) -> Dict[str, float]:
        """Projects observables onto all known axes (`axis_id -> coord in [-1, 1]`)."""
        return {ax.axis_id: round(ax.project(observables), 5) for ax in self.axes}

    def _record_causal_delta(
        self,
        cause_tag: str,
        prev_obs: Dict[str, float],
        curr_obs: Dict[str, float],
        target_axes: Optional[Sequence[EmergentSpectrumAxis]] = None,
    ) -> None:
        ct = str(cause_tag or "").strip()
        if not ct:
            return
        for ax in (target_axes if target_axes is not None else self.axes):
            if not any(k in curr_obs or k in prev_obs for k in ax.weights):
                continue
            delta = ax.project(curr_obs) - ax.project(prev_obs)
            key = (ct, ax.axis_id)
            link = self.mechanisms.get(key)
            if link is None:
                link = CausalMechanismLink(cause_tag=ct, axis_id=ax.axis_id, mean_delta=0.0, consistency=0.0, samples=0)
                self.mechanisms[key] = link
            link.update(delta)

    def observe(
        self,
        observables: Dict[str, float],
        tags: Sequence[str] = (),
        cause_tag: str = "",
        prev_observables: Optional[Dict[str, float]] = None,
        auto_discover: bool = True,
    ) -> Dict[str, Any]:
        """
        Absorbs any domain's observables, learns causal deltas along active axes,
        and spawns new orthogonal spectrum axes when unexplained residuals accumulate.
        """
        clean_obs = {str(k): float(v) for k, v in observables.items() if math.isfinite(float(v))}
        for fk in clean_obs:
            if fk not in self.feature_vocab:
                self.feature_vocab.append(fk)

        rec = {
            "observables": clean_obs,
            "tags": [str(t) for t in tags if str(t).strip()],
            "cause_tag": str(cause_tag or "").strip(),
            "prev_observables": (
                {str(k): float(v) for k, v in prev_observables.items() if math.isfinite(float(v))}
                if prev_observables is not None
                else None
            ),
        }
        self._obs_buffer.append(rec)
        if len(self._obs_buffer) > self.MAX_BUFFER:
            self._obs_buffer = self._obs_buffer[-self.MAX_BUFFER :]

        if rec["cause_tag"] and rec["prev_observables"] is not None:
            self._record_causal_delta(rec["cause_tag"], rec["prev_observables"], clean_obs)

        new_axis: Optional[EmergentSpectrumAxis] = None
        if auto_discover:
            new_axis = self.discover_new_axis_if_needed()

        return {
            "projections": self.project(clean_obs),
            "new_axis": new_axis,
            "axes_total": len(self.axes),
            "discovered_axes_total": self.discovered_axis_count(),
        }

    def discover_new_axis_if_needed(
        self,
        min_obs: int = 6,
        min_residual_ratio: float = 0.22,
        min_singular_value: float = 0.35,
        max_axes: int = 24,
    ) -> Optional[EmergentSpectrumAxis]:
        """
        Inspects recent observations for structural variance unexplained by current axes.
        Uses standardized SVD on residual matrix `R` with mean-0 sparse handling so
        distinct domains never leak cross-domain NaN/zero artifacts.
        """
        if len(self.axes) >= max_axes or len(self._obs_buffer) < min_obs:
            return None

        window = self._obs_buffer[-36:]
        n_obs = len(window)

        # Build active standardized feature columns (missing values stay 0.0 after per-feature centering)
        active_feats: List[str] = []
        col_vecs: List[np.ndarray] = []
        col_means: Dict[str, float] = {}
        col_stds: Dict[str, float] = {}

        for fk in self.feature_vocab:
            present_idx = [i for i, rec in enumerate(window) if fk in rec["observables"]]
            if len(present_idx) < min(min_obs, 4):
                continue
            vals = np.array([window[i]["observables"][fk] for i in present_idx], dtype=np.float64)
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd <= 1e-5:
                continue
            col = np.zeros(n_obs, dtype=np.float64)
            for idx, v in zip(present_idx, vals):
                col[idx] = (float(v) - mu) / sd
            active_feats.append(fk)
            col_vecs.append(col)
            col_means[fk] = mu
            col_stds[fk] = sd

        if not active_feats:
            return None

        X = np.stack(col_vecs, axis=1)  # (n_obs, F)
        total_energy = float(np.sum(X * X))
        if total_energy <= 1e-6:
            return None

        # Project X onto existing axes represented in the same standardized coordinates
        w_rows: List[np.ndarray] = []
        for ax in self.axes:
            vec = np.array([float(ax.weights.get(fk, 0.0)) * col_stds[fk] for fk in active_feats], dtype=np.float64)
            vn = float(np.linalg.norm(vec))
            if vn > 1e-8:
                w_rows.append(vec / vn)

        if w_rows:
            W = np.stack(w_rows, axis=0)  # (A, F)
            gram = W @ W.T + 1e-6 * np.eye(W.shape[0], dtype=np.float64)
            proj_coords = (X @ W.T) @ np.linalg.inv(gram)
            R = X - proj_coords @ W
        else:
            R = X

        res_energy = float(np.sum(R * R))
        residual_ratio = res_energy / (total_energy + 1e-9)
        if residual_ratio < min_residual_ratio:
            return None

        _u, svals, vt = np.linalg.svd(R, full_matrices=False)
        if svals.size == 0 or (float(svals[0]) / math.sqrt(max(1, n_obs))) < min_singular_value:
            return None

        v0 = vt[0].copy()
        # Canonical deterministic sign: orient dominant weight positive
        dom_idx = int(np.argmax(np.abs(v0)))
        if v0[dom_idx] < 0.0:
            v0 = -v0

        # Convert from standardized coordinates back into normalized raw observable weights
        raw_weights: Dict[str, float] = {}
        raw_centers: Dict[str, float] = {}
        for j, fk in enumerate(active_feats):
            if abs(float(v0[j])) >= 0.15:
                raw_weights[fk] = float(v0[j]) / col_stds[fk]
                raw_centers[fk] = col_means[fk]

        if not raw_weights:
            return None
        w_norm = math.sqrt(sum(wv * wv for wv in raw_weights.values()))
        raw_weights = {k: round(wv / max(1e-8, w_norm), 6) for k, wv in raw_weights.items()}

        # Measure raw spread over observations possessing these active features
        raw_scores: List[Tuple[float, List[str]]] = []
        for rec in window:
            obs = rec["observables"]
            if not any(k in obs for k in raw_weights):
                continue
            sc = sum((obs[k] - raw_centers[k]) * raw_weights[k] for k in raw_weights if k in obs)
            raw_scores.append((sc, rec["tags"]))

        if len(raw_scores) < min(4, min_obs):
            return None
        vals_only = np.array([p[0] for p in raw_scores], dtype=np.float64)
        ax_scale = max(1e-4, float(np.std(vals_only)))

        # Ground `-` and `+` pole labels in tags from extreme bottom/top quartiles
        raw_scores.sort(key=lambda item: item[0])
        q_cnt = max(1, len(raw_scores) // 4)
        neg_counts: Dict[str, int] = defaultdict(int)
        pos_counts: Dict[str, int] = defaultdict(int)
        for _sc, tlist in raw_scores[:q_cnt]:
            for tg in tlist:
                neg_counts[tg] += 1
        for _sc, tlist in raw_scores[-q_cnt:]:
            for tg in tlist:
                pos_counts[tg] += 1

        neg_tags = [k for k, _v in sorted(neg_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]]
        pos_tags = [k for k, _v in sorted(pos_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]]
        if not neg_tags:
            neg_tags = [f"low_{k}" for k in list(raw_weights.keys())[:2]]
        if not pos_tags:
            pos_tags = [f"high_{k}" for k in list(raw_weights.keys())[:2]]

        idx_num = self.discovered_axis_count() + 1
        while self.get_axis(f"axis::emergent_{idx_num}") is not None:
            idx_num += 1
        new_ax = EmergentSpectrumAxis(
            axis_id=f"axis::emergent_{idx_num}",
            origin="self_discovered",
            weights=raw_weights,
            center={k: round(v, 5) for k, v in raw_centers.items()},
            scale=round(ax_scale, 5),
            neg_pole_tags=neg_tags,
            pos_pole_tags=pos_tags,
            explained_variance_gain=round(float((svals[0] ** 2) / (total_energy + 1e-9)), 4),
            observations_count=len(raw_scores),
        )
        self.axes.append(new_ax)

        # Backfill causal mechanisms over buffered transitions for this newly emerged axis
        for rec in self._obs_buffer:
            if rec["cause_tag"] and rec["prev_observables"] is not None:
                self._record_causal_delta(
                    rec["cause_tag"],
                    rec["prev_observables"],
                    rec["observables"],
                    target_axes=[new_ax],
                )
        return new_ax

    def slice_spectrum_at_resolution(
        self,
        axis_id: str,
        resolution_step: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Reclassifies buffered phenomena along `axis_id` at arbitrary discrete/continuous
        granularity (e.g. step `0.5`, `0.2`, `0.1`, `0.05`) over `[-1.0, +1.0]`.
        """
        ax = self.get_axis(axis_id)
        if ax is None:
            return []
        step = max(0.02, min(1.0, float(resolution_step)))
        n_bins = max(2, int(round(2.0 / step)))
        bins_tags: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        bins_counts: Dict[int, int] = defaultdict(int)

        for rec in self._obs_buffer:
            obs = rec["observables"]
            if not any(k in obs for k in ax.weights):
                continue
            coord = ax.project(obs)
            b_idx = int(np.clip(math.floor((coord + 1.0) / 2.0 * n_bins), 0, n_bins - 1))
            bins_counts[b_idx] += 1
            for tg in rec["tags"]:
                bins_tags[b_idx][tg] += 1

        out: List[Dict[str, Any]] = []
        actual_step = 2.0 / float(n_bins)
        for b_idx in range(n_bins):
            cnt = bins_counts.get(b_idx, 0)
            if cnt == 0:
                continue
            center = -1.0 + (b_idx + 0.5) * actual_step
            top_tags = [k for k, _v in sorted(bins_tags[b_idx].items(), key=lambda kv: (-kv[1], kv[0]))[:4]]
            out.append(
                {
                    "bin_index": b_idx,
                    "center": round(center, 4),
                    "count": cnt,
                    "tags": top_tags,
                }
            )
        return out

    def query_causal_mechanisms(
        self,
        axis_id: str,
        direction: str = "+",
    ) -> List[CausalMechanismLink]:
        """Returns learned causal causes ranked by how consistently they drive `axis_id` toward `+` or `-`."""
        sign = 1.0 if direction != "-" else -1.0
        matches = [
            lk
            for (_cause, aid), lk in self.mechanisms.items()
            if aid == axis_id and lk.mean_delta * sign > 0.01
        ]
        matches.sort(key=lambda lk: (-abs(lk.mean_delta) * lk.consistency, -lk.samples, lk.cause_tag))
        return matches

    def to_context_clifford_rotor(
        self,
        observables: Dict[str, float],
        active_axis_ids: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Projects an observation onto either the 3 requested axes or its 3 most strongly
        activated axes into an 8-D `Cl(3,0)` context rotor `[s, e1, e2, e3, e12, e23, e31, e123]`.
        """
        coords = self.project(observables)
        if active_axis_ids:
            chosen = [coords.get(aid, 0.0) for aid in list(active_axis_ids)[:3]]
        else:
            ranked = sorted(coords.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
            chosen = [val for _aid, val in ranked[:3]]
        while len(chosen) < 3:
            chosen.append(0.0)
        mv = np.array([0.0, float(chosen[0]), float(chosen[1]), float(chosen[2]), 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return normalize_multivector(mv)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axes": [ax.to_dict() for ax in self.axes],
            "feature_vocab": list(self.feature_vocab),
            "obs_buffer": list(self._obs_buffer[-128:]),
            "mechanisms": [lk.to_dict() for lk in self.mechanisms.values()],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SelfExpandingSpectrumCortex":
        obj = cls()
        raw_axes = raw.get("axes") or []
        if raw_axes:
            obj.axes = [EmergentSpectrumAxis.from_dict(ax) for ax in raw_axes]
        obj.feature_vocab = [str(k) for k in (raw.get("feature_vocab") or obj.feature_vocab)]
        obj._obs_buffer = list(raw.get("obs_buffer") or [])
        for mraw in raw.get("mechanisms") or []:
            lk = CausalMechanismLink.from_dict(mraw)
            obj.mechanisms[(lk.cause_tag, lk.axis_id)] = lk
        return obj
