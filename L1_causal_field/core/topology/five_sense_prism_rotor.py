r"""
ELYSIA Five-Sense White-Light Prism Rotor & Contrastive Self-Reflection Engine
===============================================================================

Implements human perceptual prism dispersion & combinatorial rotor synthesis:
  - White-Light Dispersion: any world phenomenon passes through the 5-sense cognitive
    prism (`visual`, `auditory`, `tactile`, `chemical`, `proprioceptive`), splitting into
    coupled multi-band sensory profiles rather than 1-D straight lines.
  - RGB-Style Rotor Synthesis: bands within each sense synthesize a sense rotor in `Cl(3,0)`,
    and active sense rotors converge via geometric products into a single `WhiteLightGestalt`
    while preserving emergent cross-sense interaction couplings (`A __x__ B`).
  - Contrastive Capability Self-Reflection: tests whether existing internal prototypes &
    couplings can reconstruct an observed target; contrasts `can_recreate=True` vs `False`
    attempts to pinpoint missing bands and missing cross-sense correlations, extracts the
    missing generative residual into her prototype manifold, and unlocks re-creation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.topology.geometric_rotor_docking import (
    MV_DIM,
    clifford_geometric_product,
    normalize_multivector,
)


FIVE_SENSE_SCHEMA: Dict[str, List[str]] = {
    "visual": ["wavelength", "luminance", "gloss", "sheerness", "weave_density"],
    "auditory": ["pitch", "loudness", "sharpness", "rhythm"],
    "tactile": ["softness", "elasticity", "tension", "tear_stress", "temperature"],
    "chemical": ["saturation", "acidity", "richness", "purity"],
    "proprioceptive": ["balance", "load_weight", "harmony", "novelty"],
}

_BAND_TO_SENSE: Dict[str, str] = {
    band: sense
    for sense, bands in FIVE_SENSE_SCHEMA.items()
    for band in bands
}
_SENSE_ORDER: Tuple[str, ...] = ("visual", "auditory", "tactile", "chemical", "proprioceptive")


def _parse_sense_band(raw_key: str) -> Tuple[str, str]:
    """Resolves `'visual.sheerness'`, `'visual:sheerness'`, or bare `'sheerness'` into `(sense, band)`."""
    k = str(raw_key or "").strip()
    if "." in k:
        s, b = k.split(".", 1)
        if s in FIVE_SENSE_SCHEMA:
            return s, b
    if ":" in k:
        s, b = k.split(":", 1)
        if s in FIVE_SENSE_SCHEMA:
            return s, b
    if k in _BAND_TO_SENSE:
        return _BAND_TO_SENSE[k], k
    return "proprioceptive", k or "signal"


def _solve_bounded_nonneg(basis: np.ndarray, target: np.ndarray, max_sum: float = 1.6) -> np.ndarray:
    """
    Pure NumPy projected FISTA / gradient descent solving `min_{w >= 0, sum(w) <= max_sum} ||basis @ w - target||^2`.
    `basis` has shape `(D, P)` and `target` has shape `(D,)`.
    """
    if basis.size == 0 or basis.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    p = basis.shape[1]
    gram = basis.T @ basis
    rhs = basis.T @ target
    lip = float(np.linalg.norm(gram, ord=2))
    if lip < 1e-9:
        return np.zeros(p, dtype=np.float64)
    step = 1.0 / lip
    w = np.zeros(p, dtype=np.float64)
    z = w.copy()
    t_k = 1.0
    for _ in range(80):
        grad = gram @ z - rhs
        w_next = np.maximum(0.0, z - step * grad)
        s = float(np.sum(w_next))
        if s > max_sum:
            w_next *= max_sum / s
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k))
        z = w_next + ((t_k - 1.0) / t_next) * (w_next - w)
        if float(np.linalg.norm(w_next - w)) < 1e-6:
            w = w_next
            break
        w = w_next
        t_k = t_next
    return w


@dataclass
class WhiteLightGestalt:
    """Dispersed 5-sense spectral profile converged into one `Cl(3,0)` White-Light Rotor."""

    phenomenon_id: str
    sense_bands: Dict[str, Dict[str, float]]
    sense_rotors: Dict[str, List[float]]
    unified_rotor: List[float]
    cross_couplings: Dict[str, float] = field(default_factory=dict)

    def flat_bands(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for sense in sorted(self.sense_bands.keys()):
            for band, val in self.sense_bands[sense].items():
                out[f"{sense}:{band}"] = float(val)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WhiteLightGestalt":
        return cls(
            phenomenon_id=str(raw.get("phenomenon_id", "")),
            sense_bands={
                str(s): {str(b): float(v) for b, v in (bmap or {}).items()}
                for s, bmap in (raw.get("sense_bands") or {}).items()
            },
            sense_rotors={
                str(s): [float(x) for x in r]
                for s, r in (raw.get("sense_rotors") or {}).items()
            },
            unified_rotor=[float(x) for x in (raw.get("unified_rotor") or [1.0, 0, 0, 0, 0, 0, 0, 0])[:MV_DIM]],
            cross_couplings={str(k): float(v) for k, v in (raw.get("cross_couplings") or {}).items()},
        )


@dataclass
class RecreationAttempt:
    """Result & honest causal self-diagnosis when trying to recreate a target from current basis."""

    target_id: str
    can_recreate: bool
    fidelity: float
    band_error: float
    coupling_error: float
    missing_bands: List[str]
    missing_couplings: List[str]
    reconstructed_bands: Dict[str, Dict[str, float]]
    diagnosis_ko: str


@dataclass
class ContrastiveReflectionReport:
    """Outcome of contrasting recreable vs non-recreable targets and expanding internal correlations."""

    recreable_ids: List[str]
    non_recreable_ids: List[str]
    identified_missing_bands: List[str]
    identified_missing_couplings: List[str]
    expanded_prototypes: int
    expanded_couplings: int
    before_avg_fidelity: float
    after_avg_fidelity: float
    newly_unlocked_ids: List[str]


class FiveSensePrismCortex:
    """
    Combines 5-Sense White-Light Prism Dispersion, RGB-like `Cl(3,0)` Rotor Synthesis,
    Combinatorial Recreation Testing, and Contrastive Self-Reflection.
    """

    def __init__(self) -> None:
        self.prototypes: Dict[str, WhiteLightGestalt] = {}
        self.known_bands: set[str] = set()
        self.known_couplings: set[str] = set()
        self.history: List[Tuple[WhiteLightGestalt, RecreationAttempt]] = []

    @staticmethod
    def _encode_sense_rotor(sense: str, bands: Dict[str, float]) -> np.ndarray:
        """Maps bands inside a single sense onto bipolar `[-1, +1]` directions & planes in `Cl(3,0)`."""
        s_idx = _SENSE_ORDER.index(sense) if sense in _SENSE_ORDER else 4
        schema_bands = FIVE_SENSE_SCHEMA.get(sense, [])
        mv = np.zeros(MV_DIM, dtype=np.float64)
        mv[0] = 0.25 + 0.04 * s_idx

        for b_idx, (band_name, raw_val) in enumerate(sorted(bands.items())):
            # Center spectrum around 0.5 so opposite poles (cold vs warm, hard vs soft) oppose in Cl(3,0)
            bipolar = float(np.clip(2.0 * (float(raw_val) - 0.5), -1.5, 1.5))
            idx = schema_bands.index(band_name) if band_name in schema_bands else (len(schema_bands) + b_idx)
            vec_slot = 1 + ((idx + s_idx) % 3)
            biv_slot = 4 + ((idx + 2 * s_idx) % 3)
            mv[vec_slot] += bipolar * math.cos(0.45 * idx)
            mv[biv_slot] += bipolar * math.sin(0.40 * (idx + 1))

        return normalize_multivector(mv)

    @staticmethod
    def _compute_cross_couplings(sense_bands: Dict[str, Dict[str, float]], threshold: float = 0.35) -> Dict[str, float]:
        """Computes nonlinear cross-sense interaction terms `v_a * v_b` between active bands across senses."""
        ordered_senses = [s for s in _SENSE_ORDER if s in sense_bands] + [
            s for s in sorted(sense_bands.keys()) if s not in _SENSE_ORDER
        ]
        flat_items: List[Tuple[str, str, float]] = []
        for s in ordered_senses:
            for b, v in sorted(sense_bands[s].items()):
                fv = float(v)
                if fv >= threshold:
                    flat_items.append((s, f"{s}:{b}", fv))

        couplings: Dict[str, float] = {}
        n = len(flat_items)
        for i in range(n):
            s1, k1, v1 = flat_items[i]
            for j in range(i + 1, n):
                s2, k2, v2 = flat_items[j]
                if s1 == s2:
                    continue
                couplings[f"{k1}__x__{k2}"] = round(float(v1 * v2), 5)
        return couplings

    def disperse_and_synthesize(
        self,
        phenomenon_id: str,
        observables: Dict[str, float],
    ) -> WhiteLightGestalt:
        """
        Disperses observables through the 5-sense prism and converges active sense rotors
        via `Cl(3,0)` geometric products into one `WhiteLightGestalt`.
        """
        sense_bands: Dict[str, Dict[str, float]] = {s: {} for s in _SENSE_ORDER}
        for raw_k, raw_v in observables.items():
            fv = float(raw_v)
            if not math.isfinite(fv) or fv < 0.0:
                fv = max(0.0, fv) if math.isfinite(fv) else 0.0
            s, b = _parse_sense_band(raw_k)
            if s not in sense_bands:
                sense_bands[s] = {}
            sense_bands[s][b] = round(fv, 5)

        sense_rotors: Dict[str, List[float]] = {}
        active_rotors: List[np.ndarray] = []
        for s in _SENSE_ORDER:
            bmap = sense_bands.get(s) or {}
            if not bmap:
                continue
            r_s = self._encode_sense_rotor(s, bmap)
            sense_rotors[s] = [round(float(x), 5) for x in r_s]
            active_rotors.append(r_s)

        if active_rotors:
            unified = active_rotors[0].copy()
            for r_nxt in active_rotors[1:]:
                unified = normalize_multivector(clifford_geometric_product(unified, r_nxt))
        else:
            unified = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        couplings = self._compute_cross_couplings(sense_bands)
        return WhiteLightGestalt(
            phenomenon_id=str(phenomenon_id),
            sense_bands={s: b for s, b in sense_bands.items() if b},
            sense_rotors=sense_rotors,
            unified_rotor=[round(float(x), 5) for x in unified],
            cross_couplings=couplings,
        )

    def register_prototype(self, gestalt: WhiteLightGestalt) -> None:
        """Internalizes a grounded gestalt as an active building block in the prism manifold."""
        self.prototypes[gestalt.phenomenon_id] = gestalt
        for bk, val in gestalt.flat_bands().items():
            if val > 0.05:
                self.known_bands.add(bk)
        for ck, val in gestalt.cross_couplings.items():
            if val >= 0.12:
                self.known_couplings.add(ck)

    def register_observables_prototype(self, phenomenon_id: str, observables: Dict[str, float]) -> WhiteLightGestalt:
        g = self.disperse_and_synthesize(phenomenon_id, observables)
        self.register_prototype(g)
        return g

    def attempt_recreation(
        self,
        target_id: str,
        target_observables: Dict[str, float],
        *,
        fidelity_threshold: float = 0.85,
        record_history: bool = True,
    ) -> RecreationAttempt:
        """
        Attempts to recreate `target_observables` strictly using registered prototypes &
        known cross-sense couplings. Honestly detects missing bands and missing couplings.
        """
        target_g = self.disperse_and_synthesize(target_id, target_observables)
        t_flat = target_g.flat_bands()
        all_keys = sorted(set(t_flat.keys()) | self.known_bands)

        if not all_keys or not self.prototypes:
            missing_b = [k for k, v in t_flat.items() if v > 0.12]
            missing_c = [k for k, v in target_g.cross_couplings.items() if v >= 0.12]
            attempt = RecreationAttempt(
                target_id=target_id,
                can_recreate=False,
                fidelity=0.0,
                band_error=1.0,
                coupling_error=1.0 if missing_c else 0.0,
                missing_bands=missing_b,
                missing_couplings=missing_c,
                reconstructed_bands={},
                diagnosis_ko="보유한 감각 원형(Prototype)이 없어 대상을 재창조할 수 없습니다.",
            )
            if record_history:
                self.history.append((target_g, attempt))
            return attempt

        proto_list = list(self.prototypes.values())
        t_vec = np.array([t_flat.get(k, 0.0) for k in all_keys], dtype=np.float64)
        basis = np.zeros((len(all_keys), len(proto_list)), dtype=np.float64)
        for j, pg in enumerate(proto_list):
            pf = pg.flat_bands()
            for i, k in enumerate(all_keys):
                basis[i, j] = pf.get(k, 0.0)

        weights = _solve_bounded_nonneg(basis, t_vec, max_sum=max(3.5, float(len(proto_list)) * 1.25))
        recon_vec = basis @ weights
        recon_flat = {k: round(float(recon_vec[i]), 5) for i, k in enumerate(all_keys) if recon_vec[i] > 1e-4}

        # Reconstruct nested sense->band map
        recon_sense_bands: Dict[str, Dict[str, float]] = {}
        for sk_band, rv in recon_flat.items():
            s, b = sk_band.split(":", 1)
            recon_sense_bands.setdefault(s, {})[b] = rv

        # Band error normalized against target band magnitude
        band_denom = max(1e-6, float(np.sum(np.abs(t_vec))))
        band_error = float(np.sum(np.abs(recon_vec - t_vec)) / band_denom)
        missing_bands = [
            k
            for k, tv in t_flat.items()
            if tv >= 0.20 and (k not in self.known_bands or abs(recon_flat.get(k, 0.0) - tv) > 0.25)
        ]

        # Evaluate cross-sense couplings ONLY across couplings this system actually understands
        missing_couplings: List[str] = []
        coupling_keys = sorted(set(target_g.cross_couplings.keys()))
        if coupling_keys:
            c_err_sum = 0.0
            c_denom = 0.0
            for ck in coupling_keys:
                tv = float(target_g.cross_couplings[ck])
                c_denom += tv
                if ck in self.known_couplings:
                    kb1, kb2 = ck.split("__x__", 1)
                    rv_pair = float(recon_flat.get(kb1, 0.0)) * float(recon_flat.get(kb2, 0.0))
                    diff = abs(rv_pair - tv)
                    c_err_sum += diff
                    if diff > 0.22 and tv >= 0.15:
                        missing_couplings.append(ck)
                else:
                    c_err_sum += tv
                    missing_couplings.append(ck)
            coupling_error = float(c_err_sum / max(1e-6, c_denom))
        else:
            coupling_error = 0.0

        fidelity = float(np.clip(1.0 - 0.60 * band_error - 0.40 * coupling_error, 0.0, 1.0))
        can_recreate = (
            fidelity >= float(fidelity_threshold)
            and len(missing_bands) == 0
            and len(missing_couplings) == 0
        )

        if can_recreate:
            diag = (
                f"[{target_id}] 보유 오감 대역과 교차 상관관계만으로 재창조 가능 "
                f"(충실도={fidelity:.3f}, 대역오차={band_error:.3f}, 결합오차={coupling_error:.3f})."
            )
        else:
            parts: List[str] = []
            if missing_bands:
                parts.append(f"미보유 대역={missing_bands}")
            if missing_couplings:
                parts.append(f"미학습 감각간 상관={missing_couplings[:4]}")
            diag = (
                f"[{target_id}] 재창조 불가 (충실도={fidelity:.3f} < {fidelity_threshold:.2f}): "
                + (", ".join(parts) if parts else f"대역/결합 잔차 과다({band_error:.2f}/{coupling_error:.2f})")
            )

        attempt = RecreationAttempt(
            target_id=target_id,
            can_recreate=can_recreate,
            fidelity=round(fidelity, 4),
            band_error=round(band_error, 4),
            coupling_error=round(coupling_error, 4),
            missing_bands=sorted(missing_bands),
            missing_couplings=sorted(missing_couplings),
            reconstructed_bands=recon_sense_bands,
            diagnosis_ko=diag,
        )
        if record_history:
            self.history.append((target_g, attempt))
            if len(self.history) > 64:
                self.history = self.history[-64:]
        return attempt

    def contrast_and_expand_correlations(
        self,
        fidelity_threshold: float = 0.85,
    ) -> ContrastiveReflectionReport:
        """
        Compares `can_recreate=True` vs `can_recreate=False` experiences in history,
        isolates missing sensory bands & cross-sense correlations, extracts the uncovered
        generative residual into new prototype basis structures, and verifies newly unlocked
        re-creation capability.
        """
        if not self.history:
            return ContrastiveReflectionReport(
                recreable_ids=[],
                non_recreable_ids=[],
                identified_missing_bands=[],
                identified_missing_couplings=[],
                expanded_prototypes=0,
                expanded_couplings=0,
                before_avg_fidelity=0.0,
                after_avg_fidelity=0.0,
                newly_unlocked_ids=[],
            )

        recreable = [(g, a) for g, a in self.history if a.can_recreate]
        failed = [(g, a) for g, a in self.history if not a.can_recreate]

        recreable_ids = [a.target_id for _g, a in recreable]
        non_recreable_ids = [a.target_id for _g, a in failed]
        before_avg = float(np.mean([a.fidelity for _g, a in failed])) if failed else (
            float(np.mean([a.fidelity for _g, a in recreable])) if recreable else 0.0
        )

        # Bands/couplings verified inside successful recreations form the "understood domain"
        covered_bands: set[str] = set(self.known_bands)
        for g, _a in recreable:
            covered_bands.update(k for k, v in g.flat_bands().items() if v >= 0.2)

        missing_bands_set: set[str] = set()
        missing_couplings_set: set[str] = set()
        expanded_proto_count = 0
        expanded_coup_count = 0

        for target_g, attempt in failed:
            for mb in attempt.missing_bands:
                missing_bands_set.add(mb)
            for mc in attempt.missing_couplings:
                missing_couplings_set.add(mc)

            # Distill the missing correlated archetype from uncovered bands & non-matching cross-band spill
            t_flat = target_g.flat_bands()
            r_flat: Dict[str, float] = {}
            for s, bmap in attempt.reconstructed_bands.items():
                for b, rv in bmap.items():
                    r_flat[f"{s}:{b}"] = float(rv)

            spillover = sum(
                abs(rv) for kb, rv in r_flat.items() if t_flat.get(kb, 0.0) < 0.05
            )
            residual_obs: Dict[str, float] = {}
            for kb, tv in t_flat.items():
                if kb not in covered_bands or spillover > 0.15 or (tv - r_flat.get(kb, 0.0)) > 0.12:
                    s, b = kb.split(":", 1)
                    # Store full missing band intensity when prototype spillover prevented clean fit
                    val = tv if (spillover > 0.15 or kb not in covered_bands) else max(0.0, tv - r_flat.get(kb, 0.0))
                    residual_obs[f"{s}.{b}"] = val
                    self.known_bands.add(kb)

            # Absorb newly discovered cross-sense correlations from this failed phenomenon
            for ck in target_g.cross_couplings:
                if ck not in self.known_couplings:
                    self.known_couplings.add(ck)
                    expanded_coup_count += 1

            if residual_obs:
                res_id = f"emergent_basis::{target_g.phenomenon_id}"
                res_gestalt = self.disperse_and_synthesize(res_id, residual_obs)
                # Ensure basis carries full missing couplings associated with its residual bands
                for ck in attempt.missing_couplings:
                    res_gestalt.cross_couplings[ck] = target_g.cross_couplings.get(ck, 0.5)
                self.register_prototype(res_gestalt)
                expanded_proto_count += 1

        # Re-evaluate all previously failing targets after correlation expansion
        after_fids: List[float] = []
        newly_unlocked: List[str] = []
        new_history: List[Tuple[WhiteLightGestalt, RecreationAttempt]] = list(recreable)
        for target_g, _old_attempt in failed:
            flat_obs = {k.replace(":", "."): v for k, v in target_g.flat_bands().items()}
            re_attempt = self.attempt_recreation(
                target_g.phenomenon_id,
                flat_obs,
                fidelity_threshold=fidelity_threshold,
                record_history=False,
            )
            after_fids.append(re_attempt.fidelity)
            if re_attempt.can_recreate:
                newly_unlocked.append(target_g.phenomenon_id)
            new_history.append((target_g, re_attempt))

        self.history = new_history
        after_avg = float(np.mean(after_fids)) if after_fids else before_avg
        return ContrastiveReflectionReport(
            recreable_ids=recreable_ids,
            non_recreable_ids=non_recreable_ids,
            identified_missing_bands=sorted(missing_bands_set),
            identified_missing_couplings=sorted(missing_couplings_set),
            expanded_prototypes=expanded_proto_count,
            expanded_couplings=expanded_coup_count,
            before_avg_fidelity=round(before_avg, 4),
            after_avg_fidelity=round(after_avg, 4),
            newly_unlocked_ids=newly_unlocked,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prototypes": {pid: pg.to_dict() for pid, pg in self.prototypes.items()},
            "known_bands": sorted(self.known_bands),
            "known_couplings": sorted(self.known_couplings),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "FiveSensePrismCortex":
        obj = cls()
        for pid, praw in (raw.get("prototypes") or {}).items():
            obj.prototypes[str(pid)] = WhiteLightGestalt.from_dict(praw)
        obj.known_bands = {str(b) for b in (raw.get("known_bands") or [])}
        obj.known_couplings = {str(c) for c in (raw.get("known_couplings") or [])}
        return obj
