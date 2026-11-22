"""
unified_experiential_learner.py

Unified World–Book–Game–Creation Experiential Learning Architecture
====================================================================
Connects ELYSIA's verified Tier A sensory organs (OcularScaleOrgan,
MechanoAcousticOrgan, ImaginativeReobservationEngine, VisualCausalReasoner)
with a shared EpistemicKnowledgeSpine so she can:

1. Observe the World (세상 보기):
   - Discover spectral-purity saturation laws by mixing broadband white light
     against narrowband chroma and re-observing every transition through her eye.
   - Detect confounded perceptual symbols (e.g., chroma vs luminance), synthesize
     orthogonal counterfactuals via ImaginativeReobservationEngine, re-observe them,
     and resolve the ambiguity into crisp axis bindings.

2. Read Real Books (책 읽기 · 빈틈 자각 · 다의어 맥락 분화):
   - Read real markdown corpus files (`data/corpus/*.md`) + multi-context passages
     with ZERO hardcoded word dictionaries.
   - Spawn EpistemicVoids for ungrounded symbols and unverified causal rules.
   - Trigger Contextual Lexical Mitosis when a surface form appears in contrasting
     contexts, preserving prior senses and computing Shared Stem vs Divergent Branch.

3. Play a Blind Interactive Game (블라인드 게임 플레이 · 빈틈 해소 · 제로샷 전이):
   - Play `CausalExplorationGame` strictly via pixels, force-displacement probes,
     and PCM acoustic taps—never inspecting hidden tile types.
   - Distinguish sunlit safe granite from high-chroma crimson hazards, cleave rotated
     anisotropic timber barriers along `phi_opt + 90 deg` zero-shot under dimming,
     split deceptive soft foam from rigid walls, and verify book causal rules.

4. Create & Re-Observe Novel Artifacts (자율 창조 및 자기 감각기관 검증):
   - Synthesize a novel Level 3 puzzle and verify solvability through self-play.
   - Compose unseen multi-axis perceptual symbol combinations onto a fresh real
     image and verify both shifted axes + L2 structural preservation via OcularScaleOrgan.
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from core.physics.causal_field import CausalField
from core.sensory.epistemic_knowledge_spine import (
    AxisOperatorBinding,
    CausalRuleHypothesis,
    CreationVerificationRecord,
    EpistemicKnowledgeSpine,
    STRUCTURAL_ROLE_DIMS,
    SenseComparisonReport,
    SymbolSenseBranch,
)
from core.sensory.imaginative_perceptual_recreator import (
    ImaginativeReobservationEngine,
    PerceptualAxisManifold,
    PerceptualImaginationOp,
    calibrate_perceptual_manifold,
    wavelength_nm_to_stimulus_srgb,
)
from core.sensory.mechano_acoustic_organ import MechanoAcousticOrgan
from core.sensory.ocular_scale_organ import OcularScaleOrgan, OpticalSceneObservation
from core.sensory.visual_causal_reasoner import VisualCausalReasoner
from simulators.causal_exploration_game import CausalExplorationGame
from simulators.hidden_multiphysics_world import HiddenMultiPhysicsWorld, YOSEMITE_4K_PATH

CORPUS_DIR = Path(r"c:\Elysia\data\corpus")

# Purely structural Korean inflection/particle suffixes (no semantic dictionary)
_HANGUL_PARTICLES = (
    "에서는", "으로는", "에게서", "으로부터",
    "으로", "에서", "에게", "부터", "까지", "처럼", "보다", "하고",
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "로",
)
_HANGUL_ADJECTIVE_HADA = ("했습니다", "하였다", "합니다", "했다", "한다", "하다", "하게", "해서", "하고", "하며", "하는", "함", "한", "히")
_HANGUL_VERB_ENDINGS = ("었습니다", "았습니다", "었다", "았다", "겠다", "고", "며", "은", "게", "아", "어", "다")


def _has_batchim(ch: str) -> bool:
    code = ord(ch) - 0xAC00
    if 0 <= code <= 11171:
        return (code % 28) != 0
    return False


def _is_hangul_word(tok: str) -> bool:
    return bool(tok) and all("\uac00" <= c <= "\ud7a3" for c in tok)


def normalize_token(tok: str) -> str:
    """Morphological root normalizer: strips regular Hangul inflection/particles cleanly."""
    if not tok:
        return tok
    if not _is_hangul_word(tok):
        return tok.lower()

    cur = tok
    # Step 1: Strip unambiguous case particles without stripping adnominal -은 on 1-char verb/adj stems
    for suf in _HANGUL_PARTICLES:
        if cur.endswith(suf):
            rem = cur[: -len(suf)]
            if len(rem) >= 2:
                cur = rem
                break
            if len(rem) == 1 and suf in ("으로", "에서", "에게", "을", "를"):
                cur = rem
                break

    # Step 2: Canonicalize -하다/-한/-함 adjectives/verbs on stems len >= 2 -> '<stem>하다'
    for hada in _HANGUL_ADJECTIVE_HADA:
        if cur.endswith(hada):
            rem = cur[: -len(hada)]
            if len(rem) >= 2:
                return rem + "하다"

    # Step 3: Canonicalize 1-syllable batchim verb/adjective stems -> '<root>다' (e.g. 밝은/밝게 -> 밝다)
    for end in _HANGUL_VERB_ENDINGS:
        if cur.endswith(end):
            rem = cur[: -len(end)]
            if len(rem) == 1 and _has_batchim(rem) and cur not in ("감동", "질서", "진리", "진심", "고통", "우주", "마음", "웃음", "소음", "울림"):
                return rem + "다"
            if len(rem) >= 2 and end in ("었습니다", "았습니다", "었다", "았다", "겠다", "다"):
                return rem + "다"

    return cur


def extract_content_stems(text: str) -> List[str]:
    """
    Purely morphological tokenizer: splits punctuation/whitespace and normalizes regular
    Korean particles/endings while leaving ASCII/opaque blind tokens verbatim.
    Contains ZERO word-to-meaning rules.
    """
    cleaned = re.sub(r"[^\w\uac00-\ud7a3\s]", " ", text)
    raw_tokens = cleaned.split()
    return [st for st in (normalize_token(tok) for tok in raw_tokens) if st]


def _extract_tile_optical_signature(
    eye: OcularScaleOrgan, manifold: PerceptualAxisManifold, patch_rgb: np.ndarray
) -> Tuple[OpticalSceneObservation, Dict[str, float]]:
    """Measure strictly optical quantities from photoreceptors + 2D structure tensor."""
    obs = eye.observe(patch_rgb, max_width=64)
    a = obs.l1.luminance_a
    rg = obs.l1.rg_opponent
    by = obs.l1.by_opponent

    mean_a = float(np.mean(a))
    mean_rg = float(np.mean(rg))
    mean_by = float(np.mean(by))
    u_rg = mean_rg / max(manifold.sigma_rg, 1e-4)
    u_by = mean_by / max(manifold.sigma_by, 1e-4)
    chroma_r = float(np.hypot(u_rg, u_by))

    if a.shape[0] >= 4 and a.shape[1] >= 4:
        lap = np.abs(4.0 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]) / 4.0
        lap_hf = float(np.mean(lap))
    else:
        lap_hf = 0.0

    margin = 6 if min(a.shape[:2]) >= 20 else 1
    sub = a[margin:-margin, margin:-margin]
    gx = 0.5 * (sub[1:-1, 2:] - sub[1:-1, :-2])
    gy = 0.5 * (sub[2:, 1:-1] - sub[:-2, 1:-1])
    jxx = float(np.mean(gx**2))
    jyy = float(np.mean(gy**2))
    jxy = float(np.mean(gx * gy))
    phi_opt_rad = float(0.5 * np.arctan2(-2.0 * jxy, jyy - jxx) % np.pi)
    phi_opt_deg = float(np.rad2deg(phi_opt_rad))
    grain_coherence = float(np.hypot(jxx - jyy, 2.0 * jxy) / (jxx + jyy + 1e-8))

    return obs, {
        "mean_a": mean_a,
        "chroma_radius": chroma_r,
        "u_rg": float(u_rg),
        "u_by": float(u_by),
        "mean_rg": mean_rg,
        "mean_by": mean_by,
        "lap_hf": lap_hf,
        "phi_opt_deg": phi_opt_deg,
        "grain_coherence": grain_coherence,
    }


@dataclass
class SaturationRelationalLaw:
    """Empirical law of saturation extracted by self-re-observation through OcularScaleOrgan."""

    white_dilutions: List[float]
    chroma_radii: List[float]
    hue_angles_deg: List[float]
    luminance_means: List[float]
    l2_persistence_corrs: List[float]
    material_boundary_counts: List[int]
    boundary_tipping_w: Optional[float]
    monotone_chroma_decay: bool
    max_hue_drift_deg_core: float
    max_luminance_drift: float
    ortho_ev_chroma_drift: float
    ortho_rough_l2_min: float


@dataclass
class BookPassage:
    passage_id: str
    domain_tag: str
    text: str
    causal_pattern: Optional[Dict[str, float]] = None
    causal_rule_spec: Optional[Tuple[List[str], List[str], List[str]]] = None


@dataclass
class GamePlaySummary:
    level_id: int
    cleared: bool
    steps: int
    hazard_hits: int
    probes_count: int
    cleaved_barriers: int
    cleared_foams: int
    final_energy: float
    trajectory: List[Tuple[int, int]]
    final_frame_rgb: np.ndarray
    voids_resolved: int
    concepts_after: int


@dataclass
class CreatedLevelResult:
    level_id: int
    blueprint: Dict[str, Any]
    reobserved_hazard_chroma: float
    reobserved_sunlit_chroma: float
    reobserved_timber_coherence: float
    play_summary: GamePlaySummary
    verified: bool


@dataclass
class CreatedVisualComposition:
    target_symbols: List[str]
    target_axes: List[str]
    baseline_metrics: Dict[str, float]
    created_metrics: Dict[str, float]
    l2_persistence_corr: float
    created_rgb: np.ndarray
    verified: bool


@dataclass
class UnifiedSessionReport:
    use_blind_tokens: bool
    symbol_map: Dict[str, str]
    polysemy_targets: Dict[str, str]
    spine: EpistemicKnowledgeSpine
    saturation_law: SaturationRelationalLaw
    axis_bindings: Dict[str, AxisOperatorBinding]
    corpus_files_read: int
    sense_comparisons: Dict[str, List[SenseComparisonReport]]
    disambiguation_accuracy: float
    disambiguation_details: List[Dict[str, Any]]
    game_level1: GamePlaySummary
    game_level2: GamePlaySummary
    created_level3: CreatedLevelResult
    created_visual: CreatedVisualComposition
    causal_field: CausalField
    void_timeline: List[Dict[str, Any]]
    scene_rgb: np.ndarray


class WorldPerceptualLearner:
    """Learns continuous physical-perceptual laws & separates confounded symbol axes."""

    def __init__(self, spine: EpistemicKnowledgeSpine) -> None:
        self.spine = spine
        self.eye = OcularScaleOrgan()
        self.mechano_acoustic = MechanoAcousticOrgan()
        self.reasoner = VisualCausalReasoner(self.eye)
        self.imag_engine = ImaginativeReobservationEngine(self.eye, self.reasoner)
        self.manifold: Optional[PerceptualAxisManifold] = None

    def ensure_manifold(self, scene_rgb: np.ndarray) -> PerceptualAxisManifold:
        if self.manifold is None:
            self.manifold = calibrate_perceptual_manifold(self.eye, scene_rgb)
        return self.manifold

    def _select_target_region(self, obs: OpticalSceneObservation) -> int:
        cands = [r for r in obs.regions if 0.05 <= r.area_ratio <= 0.40]
        if not cands:
            cands = [r for r in obs.regions if r.region_id != 0] or [obs.regions[0]]
        cands.sort(key=lambda r: (-r.area_ratio, r.region_id))
        return int(cands[0].region_id)

    def _measure_scene_axes(
        self,
        base_obs: OpticalSceneObservation,
        mod_obs: OpticalSceneObservation,
        region_id: Optional[int] = None,
    ) -> Dict[str, float]:
        man = self.manifold
        assert man is not None
        mask = (base_obs.region_labels == region_id) if region_id is not None else np.ones(base_obs.shape_hw, dtype=bool)
        if not np.any(mask):
            mask = np.ones(base_obs.shape_hw, dtype=bool)

        rg0 = float(np.mean(base_obs.l1.rg_opponent[mask])) / man.sigma_rg
        by0 = float(np.mean(base_obs.l1.by_opponent[mask])) / man.sigma_by
        rg1 = float(np.mean(mod_obs.l1.rg_opponent[mask])) / man.sigma_rg
        by1 = float(np.mean(mod_obs.l1.by_opponent[mask])) / man.sigma_by

        c0 = float(np.hypot(rg0, by0))
        c1 = float(np.hypot(rg1, by1))
        a0 = float(np.mean(base_obs.l1.luminance_a[mask]))
        a1 = float(np.mean(mod_obs.l1.luminance_a[mask]))
        t0 = float(np.mean(base_obs.l2.fine_texture_map[mask]))
        t1 = float(np.mean(mod_obs.l2.fine_texture_map[mask]))

        return {
            "chroma_radius": float(c1 - c0),
            "luminance_a": float(a1 - a0),
            "fine_roughness": float((t1 - t0) / max(t0, 1e-4)),
        }

    def discover_saturation_relational_law(self, scene_rgb: np.ndarray) -> SaturationRelationalLaw:
        """
        Directly proves 'why saturation is saturation': mixes a real Yosemite region
        with pure narrowband pigment vs increasing broadband achromatic white light
        of matched luminance, then re-observes every mixture through OcularScaleOrgan.
        """
        man = self.ensure_manifold(scene_rgb)
        base_patch = cv2.resize(scene_rgb[:320, :320], (160, 160), interpolation=cv2.INTER_AREA)
        base_lin = np.power(base_patch.astype(np.float32) / 255.0, 2.2)

        h, w = base_patch.shape[:2]
        yy, xx = np.indices((h, w))
        center_mask = ((yy - h // 2) ** 2 + (xx - w // 2) ** 2) <= (42**2)

        pure_srgb = wavelength_nm_to_stimulus_srgb(635.0)
        pure_lin = np.power(pure_srgb, 2.2).astype(np.float32)
        pure_luma = float(0.5 * (pure_lin[0] + pure_lin[1]))
        white_lin = np.array([pure_luma, pure_luma, pure_luma], dtype=np.float32)

        dilutions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        chroma_radii: List[float] = []
        hue_angles: List[float] = []
        luma_means: List[float] = []
        l2_corrs: List[float] = []
        mat_counts: List[int] = []
        tipping_w: Optional[float] = None

        ref_obs: Optional[OpticalSceneObservation] = None
        for idx, wd in enumerate(dilutions):
            mixed_lin = (1.0 - wd) * pure_lin + wd * white_lin
            # Preserve Yosemite rock micro-texture modulation over the interior patch
            tex_ratio = np.clip(np.mean(base_lin, axis=-1, keepdims=True) / (np.mean(base_lin) + 1e-5), 0.55, 1.35)
            region_lin = np.clip(tex_ratio * mixed_lin[None, None, :], 0.0, 1.0)
            frame_lin = base_lin.copy()
            frame_lin[center_mask] = region_lin[center_mask]
            frame_srgb = np.clip(np.round(np.power(frame_lin, 1.0 / 2.2) * 255.0), 0, 255).astype(np.uint8)

            obs = self.eye.observe(frame_srgb, max_width=160)
            if ref_obs is None:
                ref_obs = obs
                l2_corrs.append(1.0)
            else:
                comp = self.reasoner.compare_observations(ref_obs, obs)
                l2_corrs.append(float(comp.l2_persistence_corr))

            u_rg = float(np.mean(obs.l1.rg_opponent[center_mask])) / man.sigma_rg
            u_by = float(np.mean(obs.l1.by_opponent[center_mask])) / man.sigma_by
            chroma_r = float(np.hypot(u_rg, u_by))
            hue_deg = float(np.rad2deg(np.arctan2(u_by, u_rg)) % 360.0)
            a_mean = float(np.mean(obs.l1.luminance_a[center_mask]))
            mat_cnt = sum(1 for b in obs.boundaries if b.optical_cause == "MATERIAL_BOUNDARY")

            chroma_radii.append(chroma_r)
            hue_angles.append(hue_deg)
            luma_means.append(a_mean)
            mat_counts.append(mat_cnt)
            if idx > 0 and tipping_w is None and mat_cnt < mat_counts[0]:
                tipping_w = float(wd)

        # Verify orthogonality via luminance & roughness self-re-observation sweeps
        scene_obs = self.eye.observe(scene_rgb, max_width=240)
        ortho = self.imag_engine.sweep_luminance_and_roughness(scene_obs, man)

        hue_core = np.unwrap(np.deg2rad(hue_angles[:4]))
        max_hue_drift = float(np.rad2deg(np.max(hue_core) - np.min(hue_core)))
        max_luma_drift = float(max(luma_means) - min(luma_means))
        monotone = all(chroma_radii[i + 1] <= chroma_radii[i] + 1e-4 for i in range(len(chroma_radii) - 1)) and (
            chroma_radii[0] - chroma_radii[-1] > 0.40
        )

        law = SaturationRelationalLaw(
            white_dilutions=dilutions,
            chroma_radii=chroma_radii,
            hue_angles_deg=hue_angles,
            luminance_means=luma_means,
            l2_persistence_corrs=l2_corrs,
            material_boundary_counts=mat_counts,
            boundary_tipping_w=tipping_w,
            monotone_chroma_decay=monotone,
            max_hue_drift_deg_core=max_hue_drift,
            max_luminance_drift=max_luma_drift,
            ortho_ev_chroma_drift=float(max(ortho.luminance_delta_cs)),
            ortho_rough_l2_min=float(min(ortho.roughness_persistence_corrs)),
        )
        self.spine.saturation_laws["spectral_purity_chroma"] = {
            "white_dilutions": dilutions,
            "chroma_radii": chroma_radii,
            "max_hue_drift_deg_core": max_hue_drift,
            "max_luminance_drift": max_luma_drift,
            "boundary_tipping_w": tipping_w,
            "monotone_chroma_decay": monotone,
            "ortho_ev_chroma_drift": law.ortho_ev_chroma_drift,
        }
        return law

    def ground_and_disambiguate_perceptual_symbols(
        self,
        scene_rgb: np.ndarray,
        symbol_map: Dict[str, str],
        oracle_fn: Callable[[str, Dict[str, float]], bool],
        context_tokens_map: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, AxisOperatorBinding]:
        """
        Ground visual/mechanical/acoustic symbols from blank slate.
        Starts with confounded chroma+luminance exposures, detects VOID_AMBIGUOUS_AXIS,
        synthesizes orthogonal counterfactual images via ImaginativeReobservationEngine,
        re-observes them via OcularScaleOrgan, and resolves the void cleanly.
        """
        man = self.ensure_manifold(scene_rgb)
        base_obs = self.eye.observe(scene_rgb, max_width=240)
        reg_id = self._select_target_region(base_obs)
        ctx_map = context_tokens_map or {}

        sym_chroma = normalize_token(symbol_map["chroma_high"])
        sym_luma = normalize_token(symbol_map["luma_high"])
        sym_rough = normalize_token(symbol_map["rough_high"])
        sym_stiff = normalize_token(symbol_map["stiff_high"])
        sym_ring = normalize_token(symbol_map["ring_high"])

        # 1. Confounded optical episodes: both chromatic intensity & exposure shift together on the target region
        for scale_c, d_ev in ((2.20, 0.30), (2.60, 0.38)):
            rgb_m, _ = self.imag_engine.apply_imagination(
                base_obs,
                man,
                PerceptualImaginationOp(chroma_scale=scale_c, delta_ev=d_ev),
                region_id=reg_id,
            )
            mod_obs = self.eye.observe(rgb_m, max_width=240)
            deltas = self._measure_scene_axes(base_obs, mod_obs, region_id=reg_id)
            self.spine.record_symbol_axis_contrast(
                sym_chroma,
                deltas,
                ctx_map.get("chroma_high", []),
                domain="optical",
                structural_role={
                    "resistance_to_change": 0.35, "flux_activity": 0.85,
                    "boundary_clarity": 0.75, "valence_shift": 0.30, "relational_coupling": 0.50,
                },
            )
            self.spine.record_symbol_axis_contrast(
                sym_luma,
                deltas,
                ctx_map.get("luma_high", []),
                domain="optical",
                structural_role={
                    "resistance_to_change": 0.25, "flux_activity": 0.70,
                    "boundary_clarity": 0.90, "valence_shift": 0.80, "relational_coupling": 0.55,
                },
            )

        # 2. Autonomous self-re-observation to resolve every open VOID_AMBIGUOUS_AXIS
        probe_chroma_rgb, _ = self.imag_engine.apply_imagination(
            base_obs,
            man,
            PerceptualImaginationOp(chroma_scale=2.50, delta_ev=0.0),
            region_id=reg_id,
        )
        probe_chroma_obs = self.eye.observe(probe_chroma_rgb, max_width=240)
        delta_probe_chroma = self._measure_scene_axes(base_obs, probe_chroma_obs, region_id=reg_id)

        probe_luma_rgb, _ = self.imag_engine.apply_imagination(
            base_obs, man, PerceptualImaginationOp(delta_ev=0.55), region_id=reg_id
        )
        probe_luma_obs = self.eye.observe(probe_luma_rgb, max_width=240)
        delta_probe_luma = self._measure_scene_axes(base_obs, probe_luma_obs, region_id=reg_id)

        for void_obj in list(self.spine.get_open_voids("VOID_AMBIGUOUS_AXIS")):
            sym = void_obj.target
            ok_c = bool(oracle_fn(sym, delta_probe_chroma))
            ok_l = bool(oracle_fn(sym, delta_probe_luma))
            self.spine.apply_reobservation_disambiguation(
                sym, [(delta_probe_chroma, ok_c), (delta_probe_luma, ok_l)]
            )

        # 3. Ground roughness symbol via pure regional texture re-observation
        probe_rough_rgb, _ = self.imag_engine.apply_imagination(
            base_obs, man, PerceptualImaginationOp(roughness_scale=1.85), region_id=reg_id
        )
        probe_rough_obs = self.eye.observe(probe_rough_rgb, max_width=240)
        delta_rough = self._measure_scene_axes(base_obs, probe_rough_obs, region_id=reg_id)
        self.spine.record_symbol_axis_contrast(
            sym_rough,
            {"fine_roughness": delta_rough["fine_roughness"], "chroma_radius": delta_rough["chroma_radius"]},
            ctx_map.get("rough_high", []),
            domain="optical",
            structural_role={
                "resistance_to_change": 0.65, "flux_activity": 0.45,
                "boundary_clarity": 0.80, "valence_shift": 0.15, "relational_coupling": 0.40,
            },
        )
        self.spine.apply_reobservation_disambiguation(
            sym_rough, [(delta_rough, True), (delta_probe_luma, False)]
        )

        # 4. Ground mechanical stiffness & acoustic resonance from HiddenMultiPhysicsWorld
        world = HiddenMultiPhysicsWorld(YOSEMITE_4K_PATH)
        # Probe rigid_slab_A vs compliant_canopy_C strictly through MechanoAcousticOrgan
        f_r, x_r, _ = world.mechanical_push(0.28, 0.19, force_n=16.0)
        f_s, x_s, _ = world.mechanical_push(0.70, 0.19, force_n=16.0)
        m_rigid = self.mechano_acoustic.observe_push(f_r, x_r)
        m_soft = self.mechano_acoustic.observe_push(f_s, x_s)

        pcm_r = world.acoustic_tap(0.28, 0.19)
        pcm_s = world.acoustic_tap(0.70, 0.19)
        a_rigid = self.mechano_acoustic.observe_tap(pcm_r)
        a_soft = self.mechano_acoustic.observe_tap(pcm_s)

        mech_delta = {
            "stiffness_log10": float(np.log10(max(m_rigid.effective_stiffness_n_m, 1.0)) - np.log10(max(m_soft.effective_stiffness_n_m, 1.0))),
            "damping_ratio": float(m_rigid.damping_ratio - m_soft.damping_ratio),
        }
        acoust_delta = {
            "acoustic_freq_log10": float(np.log10(max(a_rigid.peak_frequency_hz, 20.0)) - np.log10(max(a_soft.peak_frequency_hz, 20.0))),
            "sharpness_log10": float(np.log10(max(a_rigid.spectral_sharpness_q, 0.1) + 1.0) - np.log10(max(a_soft.spectral_sharpness_q, 0.1) + 1.0)),
        }

        self.spine.record_symbol_axis_contrast(
            sym_stiff,
            mech_delta,
            ctx_map.get("stiff_high", []),
            domain="mechanical",
            structural_role={
                "resistance_to_change": 0.92, "flux_activity": 0.15,
                "boundary_clarity": 0.85, "valence_shift": 0.25, "relational_coupling": 0.45,
            },
        )
        self.spine.apply_reobservation_disambiguation(
            sym_stiff,
            [({"stiffness_log10": mech_delta["stiffness_log10"]}, True), ({"stiffness_log10": 0.0}, False)],
        )

        self.spine.record_symbol_axis_contrast(
            sym_ring,
            acoust_delta,
            ctx_map.get("ring_high", []),
            domain="acoustic",
            structural_role={
                "resistance_to_change": 0.40, "flux_activity": 0.80,
                "boundary_clarity": 0.75, "valence_shift": 0.65, "relational_coupling": 0.85,
            },
        )
        self.spine.apply_reobservation_disambiguation(
            sym_ring,
            [({"acoustic_freq_log10": acoust_delta["acoustic_freq_log10"]}, True), ({"acoustic_freq_log10": 0.0}, False)],
        )

        # Register baseline optical-mechanical world concepts into spine (luminance normalized relative to scene median)
        scene_med_a = float(np.median(base_obs.l1.luminance_a))
        for y_c, x_c, m_obs, a_obs in ((0.28, 0.19, m_rigid, a_rigid), (0.70, 0.19, m_soft, a_soft)):
            crop = world.foveate(y_c, x_c)
            _, opt_sig = _extract_tile_optical_signature(self.eye, man, crop)
            self.spine.upsert_or_split_concept(
                domain_signatures={
                    "optical": [opt_sig["mean_a"] - scene_med_a, opt_sig["chroma_radius"], opt_sig["u_rg"], opt_sig["u_by"]],
                    "mechanical": [float(np.log10(max(m_obs.effective_stiffness_n_m, 1.0))), float(m_obs.damping_ratio)],
                    "acoustic": [float(np.log10(max(a_obs.peak_frequency_hz, 20.0))), float(np.log10(max(a_obs.spectral_sharpness_q, 0.1) + 1.0))],
                },
                source_medium="world",
            )

        return {
            key: self.spine.get_axis_binding(normalize_token(sym))  # type: ignore[misc]
            for key, sym in symbol_map.items()
            if self.spine.get_axis_binding(normalize_token(sym)) is not None
        }


class BookReadingLearner:
    """
    Reads real Markdown corpus texts and multi-context passages without hardcoded
    word dictionaries, detects EpistemicVoids, performs Contextual Lexical Mitosis,
    and extracts testable CausalRuleHypotheses.
    """

    def __init__(self, spine: EpistemicKnowledgeSpine) -> None:
        self.spine = spine
        self.corpus_files_read: int = 0
        self.passages_read: int = 0

    def read_corpus_directory(self, corpus_dir: Path = CORPUS_DIR, max_new_voids: int = 4) -> Dict[str, Any]:
        if not corpus_dir.exists():
            return {"files_read": 0, "tokens_processed": 0}

        md_files = sorted(corpus_dir.glob("*.md"))
        token_freq: Dict[str, int] = {}
        total_tokens = 0
        grounded_links = 0

        for fpath in md_files:
            self.corpus_files_read += 1
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                stems = extract_content_stems(line)
                if len(stems) < 3:
                    continue
                total_tokens += len(stems)
                for st in stems:
                    token_freq[st] = token_freq.get(st, 0) + 1
                    # Reinforce already-grounded symbols only if context overlaps gently
                    if st in self.spine.symbols:
                        grounded_links += 1

        # Spawn EpistemicVoids for top recurring ungrounded corpus concepts
        ungrounded = [
            (tok, cnt)
            for tok, cnt in sorted(token_freq.items(), key=lambda kv: (-kv[1], kv[0]))
            if tok not in self.spine.symbols and len(tok) >= 2 and cnt >= 4
        ]
        voids_created = 0
        for tok, cnt in ungrounded[:max_new_voids]:
            self.spine.register_void(
                kind="VOID_UNGROUNDED_SYMBOL",
                target=tok,
                priority=float(min(0.85, 0.45 + 0.04 * cnt)),
                origin_medium="book",
                hypothesis=f"Frequent corpus concept '{tok}' (freq={cnt}) lacks direct sensory-action grounding.",
                suggested_medium="world",
            )
            voids_created += 1

        return {
            "files_read": self.corpus_files_read,
            "tokens_processed": total_tokens,
            "grounded_links": grounded_links,
            "voids_created": voids_created,
        }

    def read_passages(
        self,
        passages: List[BookPassage],
        target_forms: Optional[List[str]] = None,
        mitosis_threshold: float = 0.68,
    ) -> List[Dict[str, Any]]:
        """
        Read contextual passages. Uses purely contextual co-occurrence & neighbor-grounded
        roles—never looking up specific Korean or English strings.
        """
        events: List[Dict[str, Any]] = []
        target_set = {normalize_token(t) for t in target_forms} if target_forms is not None else None

        for pas in passages:
            self.passages_read += 1
            stems = extract_content_stems(pas.text)
            if not stems:
                continue

            # Infer structural role either from grounded neighbors or passage dynamical pattern
            role = dict(pas.causal_pattern) if pas.causal_pattern is not None else self._infer_role_from_neighbors(stems)

            observe_tokens = [s for s in stems if target_set is None or s in target_set]
            if target_set is not None and not observe_tokens:
                # Still absorb general salient stems when not filtered
                observe_tokens = stems[:4]

            for tok in dict.fromkeys(observe_tokens):
                _, evt = self.spine.observe_symbol_in_context(
                    surface_form=tok,
                    context_tokens=stems,
                    domain=pas.domain_tag,
                    structural_role=role,
                    mitosis_threshold=mitosis_threshold,
                    register_void_if_ungrounded=(tok not in self.spine.symbols and pas.causal_pattern is None),
                )
                if evt is not None:
                    events.append(evt)

            if pas.causal_rule_spec is not None:
                conds, acts, effs = pas.causal_rule_spec
                rid = f"rule_{len(self.spine.causal_rules):03d}"
                rule = CausalRuleHypothesis(
                    rule_id=rid,
                    source_medium="book",
                    condition_tokens=list(conds),
                    action_tokens=list(acts),
                    predicted_effect_tokens=list(effs),
                    verified_in_reality=False,
                )
                self.spine.causal_rules[rid] = rule
                self.spine.register_void(
                    kind="VOID_UNVERIFIED_BOOK_RULE",
                    target=rid,
                    priority=0.88,
                    origin_medium="book",
                    hypothesis=f"Rule {conds} + {acts} -> {effs} read in book; needs physical game/world test.",
                    suggested_medium="game",
                )
        return events

    def _infer_role_from_neighbors(self, stems: List[str]) -> Dict[str, float]:
        accum = {k: 0.0 for k in STRUCTURAL_ROLE_DIMS}
        count = 0
        for st in stems:
            senses = self.spine.symbols.get(st, [])
            if senses:
                for k in STRUCTURAL_ROLE_DIMS:
                    accum[k] += senses[0].structural_role_profile.get(k, 0.2)
                count += 1
        if count == 0:
            return {k: 0.25 for k in STRUCTURAL_ROLE_DIMS}
        return {k: float(accum[k] / count) for k in STRUCTURAL_ROLE_DIMS}

    def evaluate_disambiguation_benchmark(
        self, test_cases: List[Tuple[str, str, str]]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate context-only disambiguation accuracy on unseen sentences.
        Each item is `(sentence, target_form, expected_domain_tag)`.
        No `domain_hint` is supplied at inference time!
        """
        details: List[Dict[str, Any]] = []
        correct = 0
        for sentence, target_form, expected_domain in test_cases:
            stems = extract_content_stems(sentence)
            norm_target = normalize_token(target_form)
            chosen, margin, scores = self.spine.disambiguate_symbol(norm_target, stems, domain_hint=None)
            predicted_dom = chosen.primary_domain if chosen is not None else "NONE"
            is_ok = predicted_dom == expected_domain
            if is_ok:
                correct += 1
            details.append(
                {
                    "sentence": sentence,
                    "target_form": norm_target,
                    "expected_domain": expected_domain,
                    "predicted_domain": predicted_dom,
                    "chosen_sense_id": chosen.sense_id if chosen else None,
                    "margin": margin,
                    "correct": is_ok,
                    "scores": scores,
                }
            )
        acc = float(correct / max(1, len(test_cases)))
        return acc, details


class GamePlayingLearner:
    """
    Plays CausalExplorationGame strictly via OcularScaleOrgan + MechanoAcousticOrgan,
    resolving Book/Game EpistemicVoids and transferring invariants zero-shot across levels.
    """

    def __init__(self, spine: EpistemicKnowledgeSpine, manifold: PerceptualAxisManifold) -> None:
        self.spine = spine
        self.manifold = manifold
        self.eye = OcularScaleOrgan()
        self.mechano_acoustic = MechanoAcousticOrgan()
        self._tile_beliefs: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def _classify_sig_to_belief(self, sig: Dict[str, float], frame_median_a: float) -> Dict[str, Any]:
        rel_a = float(sig["mean_a"] - frame_median_a)
        opt_vec = [rel_a, sig["chroma_radius"], sig["u_rg"], sig["u_by"]]
        matched = self.spine.match_concept_by_optical(opt_vec, max_dist=0.58)

        # Classify strictly from learned concepts or illumination-invariant photoreceptor invariants
        predicted_hazard = False
        predicted_goal = False
        predicted_passable = True
        predicted_obstacle = False
        # High directional coherence + elevated high-frequency grain energy identifies timber grain barrier
        has_grain = (sig["grain_coherence"] > 0.86) and (sig["lap_hf"] > 0.020)

        if matched is not None and "game_outcome" in matched.domain_signatures:
            go = matched.domain_signatures["game_outcome"]
            # [passable_flag, hazard_flag, goal_flag]
            if go[1] > 0.5:
                predicted_hazard = True
                predicted_passable = False
            elif go[2] > 0.5:
                predicted_goal = True
                predicted_passable = True
            elif go[0] < 0.5:
                predicted_obstacle = True
                predicted_passable = False
        else:
            # Grounded photoreceptor checks:
            # High chroma + yellow-gold (u_by < -0.50, u_rg < 0.60) -> beacon candidate
            if sig["chroma_radius"] > 0.65 and sig["u_by"] < -0.50 and sig["u_rg"] < 0.60:
                predicted_goal = True
            # High chroma + strong red opponent -> check if hazard rule already empirically verified
            elif sig["chroma_radius"] > 0.70 and sig["u_rg"] > 0.60:
                hazard_verified = any(
                    r_obj.verified_in_reality and any(e == "hazard_repel" for e in r_obj.predicted_effect_tokens)
                    for r_obj in self.spine.causal_rules.values()
                )
                if hazard_verified:
                    predicted_hazard = True
                    predicted_passable = False
            # Anisotropic grain barrier OR relatively dark framed basalt wall / microporous foam
            elif has_grain or (rel_a < -0.50 and sig["chroma_radius"] < 0.35):
                predicted_obstacle = True
                predicted_passable = False

        return {
            "sig": sig,
            "rel_a": rel_a,
            "opt_vec": opt_vec,
            "matched_concept": matched.concept_id if matched else None,
            "predicted_hazard": predicted_hazard,
            "predicted_goal": predicted_goal,
            "predicted_passable": predicted_passable,
            "predicted_obstacle": predicted_obstacle,
            "has_grain": has_grain,
            "probed": False,
        }

    def _scan_tile_optically(
        self, game: CausalExplorationGame, r: int, c: int, frame_median_a: Optional[float] = None
    ) -> Dict[str, Any]:
        patch = game.foveate_tile(r, c)
        _, sig = _extract_tile_optical_signature(self.eye, self.manifold, patch)
        med = sig["mean_a"] if frame_median_a is None else float(frame_median_a)
        return self._classify_sig_to_belief(sig, med)

    def _plan_path(
        self,
        start: Tuple[int, int],
        goal_candidates: List[Tuple[int, int]],
        grid_shape: Tuple[int, int],
        beliefs: Dict[Tuple[int, int], Dict[str, Any]],
    ) -> Optional[List[Tuple[int, int]]]:
        rows, cols = grid_shape
        goal_set = set(goal_candidates)
        if start in goal_set:
            return [start]

        pq: List[Tuple[float, Tuple[int, int], List[Tuple[int, int]]]] = [(0.0, start, [start])]
        best_cost: Dict[Tuple[int, int], float] = {start: 0.0}

        while pq:
            cost, cur, path = heapq.heappop(pq)
            if cur in goal_set:
                return path
            if cost > best_cost.get(cur, 1e9) + 1e-6:
                continue
            r, c = cur
            for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                b = beliefs[(nr, nc)]
                if b["predicted_hazard"]:
                    continue
                if not b["predicted_passable"] and (nr, nc) not in goal_set:
                    continue
                step_cost = 1.0 if b["predicted_passable"] else 3.5
                ncost = cost + step_cost
                if ncost < best_cost.get((nr, nc), 1e9):
                    best_cost[(nr, nc)] = ncost
                    heapq.heappush(pq, (ncost, (nr, nc), path + [(nr, nc)]))
        return None

    def play_level(self, game: CausalExplorationGame, max_steps: int = 40) -> GamePlaySummary:
        rows, cols = game.grid_shape
        raw_sigs: Dict[Tuple[int, int], Dict[str, float]] = {}
        for r in range(rows):
            for c in range(cols):
                _, sig = _extract_tile_optical_signature(self.eye, self.manifold, game.foveate_tile(r, c))
                raw_sigs[(r, c)] = sig
        frame_median_a = float(np.median([s["mean_a"] for s in raw_sigs.values()]))

        beliefs: Dict[Tuple[int, int], Dict[str, Any]] = {
            pos: self._classify_sig_to_belief(sig, frame_median_a)
            for pos, sig in raw_sigs.items()
        }

        cleaved_barriers = 0
        cleared_foams = 0
        voids_resolved = 0

        def _dir_from_delta(dr: int, dc: int) -> str:
            return {(-1, 0): "N", (1, 0): "S", (0, 1): "E", (0, -1): "W"}[(dr, dc)]

        for _ in range(max_steps):
            if game.level_cleared:
                break

            cur = game.agent_pos
            # 1. Identify Goal Beacon candidates optically
            goal_cands = [pos for pos, b in beliefs.items() if b["predicted_goal"]]
            if not goal_cands:
                # Fallback: highest chroma non-red tile
                goal_cands = [
                    max(
                        beliefs.keys(),
                        key=lambda p: beliefs[p]["sig"]["chroma_radius"] - max(0.0, beliefs[p]["sig"]["u_rg"]),
                    )
                ]

            # 2. Plan path over passable tiles; if blocked, plan to probe adjacent obstacles
            passable_path = self._plan_path(cur, goal_cands, (rows, cols), beliefs)
            if passable_path is not None and len(passable_path) >= 2:
                nxt = passable_path[1]
            else:
                # Find nearest unprobed obstacle reachable from cur
                obs_cands = [
                    p
                    for p, b in beliefs.items()
                    if b["predicted_obstacle"] and not b["probed"]
                ]
                obs_path = self._plan_path(cur, obs_cands, (rows, cols), beliefs) if obs_cands else None
                if obs_path is not None and len(obs_path) >= 2:
                    nxt = obs_path[1]
                else:
                    nbrs = []
                    for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                        nr, nc = cur[0] + dr, cur[1] + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not beliefs[(nr, nc)]["predicted_hazard"]:
                            nbrs.append((nr, nc))
                    if not nbrs:
                        break
                    nbrs.sort(key=lambda p: abs(p[0] - goal_cands[0][0]) + abs(p[1] - goal_cands[0][1]))
                    nxt = nbrs[0]

            nr, nc = nxt
            b_nxt = beliefs[nxt]

            # 3. If next cell is a predicted obstacle, probe acoustically & mechanically first!
            if b_nxt["predicted_obstacle"] and not b_nxt["probed"]:
                b_nxt["probed"] = True
                void_ent = self.spine.register_void(
                    kind="VOID_UNVERIFIED_GAME_ENTITY",
                    target=f"L{game.level_id}_tile_{nr}_{nc}",
                    priority=0.80,
                    origin_medium="game",
                    hypothesis="Adjacent textured/dark tile resists entry; verify mechanics & acoustics.",
                    suggested_medium="game",
                )

                pcm = game.probe_tap_tile(nr, nc)
                a_obs = self.mechano_acoustic.observe_tap(pcm)

                # Determine push angle: if striated grain detected, push orthogonally to phi_opt!
                sig = b_nxt["sig"]
                if b_nxt["has_grain"]:
                    phi_weak_rad = np.deg2rad((sig["phi_opt_deg"] + 90.0) % 180.0)
                    dir_yx = (float(np.sin(phi_weak_rad)), float(np.cos(phi_weak_rad)))
                else:
                    dir_yx = (float(nr - cur[0]), float(nc - cur[1]))

                f_t, x_t, changed = game.probe_push_tile(nr, nc, dir_yx=dir_yx, force_n=16.0)
                m_obs = self.mechano_acoustic.observe_push(f_t, x_t)

                # Record empirical concept & trigger Concept Mitosis (e.g. foam vs rigid granite)
                node, _ = self.spine.upsert_or_split_concept(
                    domain_signatures={
                        "optical": list(b_nxt["opt_vec"]),
                        "mechanical": [float(np.log10(max(m_obs.effective_stiffness_n_m, 1.0))), float(m_obs.damping_ratio)],
                        "acoustic": [float(np.log10(max(a_obs.peak_frequency_hz, 20.0))), float(np.log10(max(a_obs.spectral_sharpness_q, 0.1) + 1.0))],
                        "game_outcome": [1.0 if changed else 0.0, 0.0, 0.0],
                    },
                    source_medium="game",
                    causal_effects={"push": {"yield": 1.0 if m_obs.yield_detected else 0.0, "opened": 1.0 if changed else 0.0}},
                    optical_match_threshold=0.58,
                    surprise_threshold=0.75,
                )

                if changed:
                    b_nxt["predicted_passable"] = True
                    b_nxt["predicted_obstacle"] = False
                    if b_nxt["has_grain"]:
                        cleaved_barriers += 1
                        for r_obj in self.spine.causal_rules.values():
                            if any(e == "open_passage" for e in r_obj.predicted_effect_tokens):
                                r_obj.verified_in_reality = True
                                r_obj.confirmations += 1
                                voids_resolved += self.spine.resolve_voids_by_target(
                                    "VOID_UNVERIFIED_BOOK_RULE", r_obj.rule_id, "Verified timber grain cleavage in blind gameplay."
                                )
                    else:
                        cleared_foams += 1
                        for r_obj in self.spine.causal_rules.values():
                            if any(e == "compressible_pass" for e in r_obj.predicted_effect_tokens):
                                r_obj.verified_in_reality = True
                                r_obj.confirmations += 1
                                voids_resolved += self.spine.resolve_voids_by_target(
                                    "VOID_UNVERIFIED_BOOK_RULE", r_obj.rule_id, "Verified soft micropore foam compression in blind gameplay."
                                )
                else:
                    b_nxt["predicted_passable"] = False

                self.spine.resolve_void(void_ent.void_id, f"Probed k={m_obs.effective_stiffness_n_m:.0f}N/m, f={a_obs.peak_frequency_hz:.0f}Hz, opened={changed}")
                voids_resolved += 1
                _, new_sig = _extract_tile_optical_signature(self.eye, self.manifold, game.foveate_tile(nr, nc))
                beliefs[nxt]["sig"] = new_sig
                continue

            # 4. Execute movement step and observe genuine outcome!
            dr, dc = nr - cur[0], nc - cur[1]
            outcome = game.move(_dir_from_delta(dr, dc))
            sig = b_nxt["sig"]

            if outcome.repelled:
                b_nxt["predicted_hazard"] = True
                b_nxt["predicted_passable"] = False
                self.spine.upsert_or_split_concept(
                    domain_signatures={
                        "optical": list(b_nxt["opt_vec"]),
                        "game_outcome": [0.0, 1.0, 0.0],
                    },
                    source_medium="game",
                    causal_effects={"step": {"energy_delta": outcome.energy_delta, "repelled": 1.0}},
                    optical_match_threshold=0.58,
                )
                for r_obj in self.spine.causal_rules.values():
                    if any(e == "hazard_repel" for e in r_obj.predicted_effect_tokens):
                        r_obj.verified_in_reality = True
                        r_obj.confirmations += 1
                        voids_resolved += self.spine.resolve_voids_by_target(
                            "VOID_UNVERIFIED_BOOK_RULE", r_obj.rule_id, "Empirically confirmed crimson hazard repulsion."
                        )
                # Generalize newly discovered hazard signature immediately across the visual field
                for pos, b_other in beliefs.items():
                    if np.linalg.norm(np.array(b_other["opt_vec"]) - np.array(b_nxt["opt_vec"])) < 0.58:
                        b_other["predicted_hazard"] = True
                        b_other["predicted_passable"] = False
            elif outcome.blocked:
                b_nxt["predicted_passable"] = False
                b_nxt["predicted_obstacle"] = True
            elif outcome.moved:
                if outcome.level_cleared:
                    self.spine.upsert_or_split_concept(
                        domain_signatures={
                            "optical": list(b_nxt["opt_vec"]),
                            "game_outcome": [1.0, 0.0, 1.0],
                        },
                        source_medium="game",
                        causal_effects={"step": {"energy_delta": outcome.energy_delta, "cleared": 1.0}},
                        optical_match_threshold=0.58,
                    )
                elif b_nxt["rel_a"] > 0.50 and sig["chroma_radius"] < 0.45:
                    # Sunlit bright safe floor empirically verified safe!
                    self.spine.upsert_or_split_concept(
                        domain_signatures={
                            "optical": list(b_nxt["opt_vec"]),
                            "game_outcome": [1.0, 0.0, 0.0],
                        },
                        source_medium="game",
                        causal_effects={"step": {"energy_delta": outcome.energy_delta, "safe": 1.0}},
                        optical_match_threshold=0.58,
                    )

        return GamePlaySummary(
            level_id=game.level_id,
            cleared=game.level_cleared,
            steps=game.step_count,
            hazard_hits=game.hazard_hits,
            probes_count=game.probes_count,
            cleaved_barriers=cleaved_barriers,
            cleared_foams=cleared_foams,
            final_energy=game.energy,
            trajectory=list(game.trajectory),
            final_frame_rgb=game.observe_frame(),
            voids_resolved=voids_resolved,
            concepts_after=len(self.spine.concepts),
        )


class CreativeSynthesisEngine:
    """
    Synthesizes novel game levels and unseen multi-axis perceptual visual creations
    from accumulated principles, verifying every creation via ELYSIA's own sensory organs.
    """

    def __init__(self, spine: EpistemicKnowledgeSpine, world_learner: WorldPerceptualLearner) -> None:
        self.spine = spine
        self.world = world_learner

    def create_and_verify_novel_game_level(
        self, game_learner: GamePlayingLearner, seed: int = 107
    ) -> CreatedLevelResult:
        blueprint = {
            "start": (0, 2),
            "goal": (4, 2),
            "grain_angle_deg": 0.0,  # Horizontal grain -> weak axis North/South (approached from (1,2) pushing South!)
            "illumination_scale": 0.85,
            "layout": [
                ["RIGID_WALL", "CRIMSON_HAZARD", "SAFE_FLOOR", "CRIMSON_HAZARD", "RIGID_WALL"],
                ["SAFE_FLOOR", "DECEPTIVE_FOAM", "SUNLIT_SAFE_FLOOR", "RIGID_WALL", "SAFE_FLOOR"],
                ["RIGID_WALL", "CRIMSON_HAZARD", "TIMBER_BARRIER", "CRIMSON_HAZARD", "RIGID_WALL"],
                ["SAFE_FLOOR", "RIGID_WALL", "SUNLIT_SAFE_FLOOR", "DECEPTIVE_FOAM", "SAFE_FLOOR"],
                ["RIGID_WALL", "SAFE_FLOOR", "GOAL_BEACON", "SAFE_FLOOR", "RIGID_WALL"],
            ],
        }
        level3 = CausalExplorationGame(level_id=3, seed=seed, custom_blueprint=blueprint)

        # Self-re-observe created tiles through OcularScaleOrgan
        man = game_learner.manifold
        _, haz_sig = _extract_tile_optical_signature(game_learner.eye, man, level3.foveate_tile(2, 1))
        _, sun_sig = _extract_tile_optical_signature(game_learner.eye, man, level3.foveate_tile(1, 2))
        _, tim_sig = _extract_tile_optical_signature(game_learner.eye, man, level3.foveate_tile(2, 2))

        play_summary = game_learner.play_level(level3, max_steps=30)
        verified = bool(
            play_summary.cleared
            and play_summary.hazard_hits == 0
            and play_summary.cleaved_barriers >= 1
            and haz_sig["chroma_radius"] > sun_sig["chroma_radius"] + 0.25
            and tim_sig["grain_coherence"] > 0.62
        )

        self.spine.record_creation(
            CreationVerificationRecord(
                creation_id="create_level_3",
                kind="NOVEL_GAME_LEVEL",
                target_spec={
                    "start": list(blueprint["start"]),
                    "goal": list(blueprint["goal"]),
                    "grain_angle_deg": blueprint["grain_angle_deg"],
                    "illumination_scale": blueprint["illumination_scale"],
                },
                reobservation_metrics={
                    "hazard_chroma": haz_sig["chroma_radius"],
                    "sunlit_chroma": sun_sig["chroma_radius"],
                    "timber_coherence": tim_sig["grain_coherence"],
                    "self_play_cleared": play_summary.cleared,
                    "self_play_hazard_hits": play_summary.hazard_hits,
                    "self_play_cleaved": play_summary.cleaved_barriers,
                },
                verified_by_own_organs=verified,
                summary=(
                    f"Created Level 3; self-eye confirmed hazard_chroma={haz_sig['chroma_radius']:.2f} > "
                    f"sunlit_chroma={sun_sig['chroma_radius']:.2f} & self-play cleared in {play_summary.steps} steps."
                ),
            )
        )
        return CreatedLevelResult(
            level_id=3,
            blueprint=blueprint,
            reobserved_hazard_chroma=haz_sig["chroma_radius"],
            reobserved_sunlit_chroma=sun_sig["chroma_radius"],
            reobserved_timber_coherence=tim_sig["grain_coherence"],
            play_summary=play_summary,
            verified=verified,
        )

    def create_and_reobserve_novel_visual_composition(
        self, scene_rgb: np.ndarray, target_symbols: List[str]
    ) -> CreatedVisualComposition:
        """
        Synthesize an image satisfying an unseen multi-symbol combination (e.g.
        high chroma + high roughness) by compiling learned symbol axis bindings into
        PerceptualImaginationOp and verifying via closed-loop OcularScaleOrgan re-observation.
        """
        man = self.world.ensure_manifold(scene_rgb)
        base_obs = self.world.eye.observe(scene_rgb, max_width=240)
        reg_id = self.world._select_target_region(base_obs)

        norm_symbols = [normalize_token(s) for s in target_symbols]
        target_axes: List[str] = []
        target_wavelength_nm: Optional[float] = None
        target_chroma_pull = 0.0
        chroma_dissolve_frac = 0.0
        delta_ev = 0.0
        rough_scale = 1.0

        for sym in norm_symbols:
            b = self.spine.get_axis_binding(sym)
            if b is None:
                continue
            target_axes.append(b.axis_name)
            if b.axis_name == "chroma_radius":
                target_wavelength_nm = 650.0
                target_chroma_pull = 0.85 if b.direction > 0 else 0.0
                chroma_dissolve_frac = 0.65 if b.direction > 0 else 0.0
            elif b.axis_name == "luminance_a":
                delta_ev = 0.45 if b.direction > 0 else -0.45
            elif b.axis_name == "fine_roughness":
                rough_scale = 1.85 if b.direction > 0 else 0.50

        op = PerceptualImaginationOp(
            target_wavelength_nm=target_wavelength_nm,
            target_chroma_pull=target_chroma_pull,
            chroma_dissolve_frac=chroma_dissolve_frac,
            delta_ev=delta_ev,
            roughness_scale=rough_scale,
        )
        created_rgb, _ = self.world.imag_engine.apply_imagination(base_obs, man, op, region_id=reg_id)
        reobs = self.world.eye.observe(created_rgb, max_width=240)
        comp = self.world.reasoner.compare_observations(base_obs, reobs)
        deltas = self.world._measure_scene_axes(base_obs, reobs, region_id=reg_id)

        base_metrics = self.world._measure_scene_axes(base_obs, base_obs, region_id=reg_id)
        axes_ok = all(
            (deltas.get(ax, 0.0) * (self.spine.get_axis_binding(sym).direction)) > 0.08  # type: ignore[union-attr]
            for sym, ax in zip(norm_symbols, target_axes)
            if self.spine.get_axis_binding(sym) is not None
        )
        verified = bool(len(target_axes) >= 2 and axes_ok and comp.l2_persistence_corr >= 0.80)

        self.spine.record_creation(
            CreationVerificationRecord(
                creation_id="create_visual_combo",
                kind="MULTI_AXIS_VISUAL_COMPOSITION",
                target_spec={"symbols": list(norm_symbols), "axes": target_axes},
                reobservation_metrics={
                    "deltas": deltas,
                    "l2_persistence_corr": float(comp.l2_persistence_corr),
                },
                verified_by_own_organs=verified,
                summary=f"Synthesized {norm_symbols} -> axes {target_axes}, L2_corr={comp.l2_persistence_corr:.3f}",
            )
        )
        return CreatedVisualComposition(
            target_symbols=list(norm_symbols),
            target_axes=target_axes,
            baseline_metrics=base_metrics,
            created_metrics=deltas,
            l2_persistence_corr=float(comp.l2_persistence_corr),
            created_rgb=created_rgb,
            verified=verified,
        )


def build_curriculum_bundle(use_blind_tokens: bool = False) -> Dict[str, Any]:
    """
    Build the Word / Book / Disambiguation curriculum either in natural Korean OR
    with EVERY target & contextual cue token swapped for opaque random identifiers
    (`use_blind_tokens=True`) to prove 0% string/dictionary cheating.
    """
    if not use_blind_tokens:
        symbol_map = {
            "chroma_high": "채도",
            "luma_high": "명도",
            "rough_high": "거칠기",
            "stiff_high": "단단함",
            "ring_high": "울림",
        }
        polysemy = {"fruit_apology": "사과", "light_mood": "밝다", "rock_resolve": "단단하다"}
        passages = [
            # 1. Grounded game causal principles (Rules to verify in blind gameplay)
            BookPassage(
                passage_id="game_rule_hazard",
                domain_tag="game_manual",
                text="선명한 붉은 색채의 높은 채도 영역은 위험하므로 피해야 한다.",
                causal_pattern={"resistance_to_change": 0.4, "flux_activity": 0.9, "boundary_clarity": 0.8, "valence_shift": -0.7, "relational_coupling": 0.5},
                causal_rule_spec=(["선명", "채도"], ["avoid"], ["hazard_repel"]),
            ),
            BookPassage(
                passage_id="game_rule_cleavage",
                domain_tag="game_manual",
                text="나뭇결이 보이는 장애물은 결에 수직 방향으로 강하게 밀면 쪼개져 통로가 열린다.",
                causal_pattern={"resistance_to_change": 0.7, "flux_activity": 0.6, "boundary_clarity": 0.85, "valence_shift": 0.5, "relational_coupling": 0.6},
                causal_rule_spec=(["나뭇결", "장애물"], ["push_across_grain"], ["open_passage"]),
            ),
            BookPassage(
                passage_id="game_rule_foam",
                domain_tag="game_manual",
                text="겉은 바위 같아도 낮은 울림이 나고 쉽게 눌리는 물체는 밀어서 통과할 수 있다.",
                causal_pattern={"resistance_to_change": 0.2, "flux_activity": 0.5, "boundary_clarity": 0.6, "valence_shift": 0.4, "relational_coupling": 0.4},
                causal_rule_spec=(["울림", "물체"], ["push"], ["compressible_pass"]),
            ),
            # 2. Polysemy pair 1: 사과 (fruit vs apology)
            BookPassage(
                passage_id="poly_apple_1",
                domain_tag="physical_fruit",
                text="붉은 사과를 손으로 잡으니 단단한 껍질 너머 달콤한 과즙과 열매 향기가 느껴진다.",
                causal_pattern={"resistance_to_change": 0.45, "flux_activity": 0.35, "boundary_clarity": 0.75, "valence_shift": 0.80, "relational_coupling": 0.70},
            ),
            BookPassage(
                passage_id="poly_apple_2",
                domain_tag="physical_fruit",
                text="나무에서 떨어진 신선한 사과는 아삭한 식감과 영양을 주는 맛있는 과일이다.",
                causal_pattern={"resistance_to_change": 0.45, "flux_activity": 0.35, "boundary_clarity": 0.75, "valence_shift": 0.80, "relational_coupling": 0.70},
            ),
            BookPassage(
                passage_id="poly_apology_1",
                domain_tag="social_apology",
                text="잘못을 저지른 뒤 진심으로 고개를 숙여 사과를 하니 상대의 마음이 누그러졌다.",
                causal_pattern={"resistance_to_change": 0.30, "flux_activity": 0.50, "boundary_clarity": 0.65, "valence_shift": 0.85, "relational_coupling": 0.90},
            ),
            BookPassage(
                passage_id="poly_apology_2",
                domain_tag="social_apology",
                text="오해와 다툼 끝에 용서를 구하며 건넨 따뜻한 사과는 관계를 다시 회복시켰다.",
                causal_pattern={"resistance_to_change": 0.30, "flux_activity": 0.50, "boundary_clarity": 0.65, "valence_shift": 0.85, "relational_coupling": 0.90},
            ),
            # 3. Polysemy pair 2: 밝다 (optical illumination vs cheerful social/mood)
            BookPassage(
                passage_id="poly_bright_opt_1",
                domain_tag="optical_light",
                text="아침 햇살이 비추는 화강암 절벽은 그림자가 걷혀 시야가 매우 밝다.",
                causal_pattern={"resistance_to_change": 0.25, "flux_activity": 0.60, "boundary_clarity": 0.90, "valence_shift": 0.75, "relational_coupling": 0.55},
            ),
            BookPassage(
                passage_id="poly_bright_opt_2",
                domain_tag="optical_light",
                text="높은 조명의 빛이 강하게 반사되는 바닥은 어두운 밤보다 훨씬 밝다.",
                causal_pattern={"resistance_to_change": 0.25, "flux_activity": 0.60, "boundary_clarity": 0.90, "valence_shift": 0.75, "relational_coupling": 0.55},
            ),
            BookPassage(
                passage_id="poly_bright_mood_1",
                domain_tag="social_mood",
                text="아이의 웃음소리와 다정한 인사 덕분에 집안 분위기와 표정이 무척 밝다.",
                causal_pattern={"resistance_to_change": 0.20, "flux_activity": 0.55, "boundary_clarity": 0.80, "valence_shift": 0.85, "relational_coupling": 0.85},
            ),
            BookPassage(
                passage_id="poly_bright_mood_2",
                domain_tag="social_mood",
                text="희망찬 노래와 따뜻한 대화가 이어지자 모두의 기분과 목소리가 밝다.",
                causal_pattern={"resistance_to_change": 0.20, "flux_activity": 0.55, "boundary_clarity": 0.80, "valence_shift": 0.85, "relational_coupling": 0.85},
            ),
            # 4. Polysemy pair 3: 단단하다 (physical rock vs mental resolve)
            BookPassage(
                passage_id="poly_hard_phys_1",
                domain_tag="physical_mechanics",
                text="손으로 힘껏 밀어도 조금도 찌그러지지 않는 화강암 바위는 매우 단단하다.",
                causal_pattern={"resistance_to_change": 0.95, "flux_activity": 0.15, "boundary_clarity": 0.90, "valence_shift": 0.20, "relational_coupling": 0.40},
            ),
            BookPassage(
                passage_id="poly_hard_phys_2",
                domain_tag="physical_mechanics",
                text="망치로 두드려도 높은 울림만 내고 부서지지 않는 광물 표면은 참 단단하다.",
                causal_pattern={"resistance_to_change": 0.95, "flux_activity": 0.15, "boundary_clarity": 0.90, "valence_shift": 0.20, "relational_coupling": 0.40},
            ),
            BookPassage(
                passage_id="poly_hard_mind_1",
                domain_tag="mental_resolve",
                text="어려운 시련과 유혹이 닥쳐도 흔들리지 않는 그녀의 결심과 의지는 매우 단단하다.",
                causal_pattern={"resistance_to_change": 0.92, "flux_activity": 0.25, "boundary_clarity": 0.85, "valence_shift": 0.65, "relational_coupling": 0.70},
            ),
            BookPassage(
                passage_id="poly_hard_mind_2",
                domain_tag="mental_resolve",
                text="수많은 실패 속에서도 포기하지 않고 끝까지 버티는 신념과 약속은 참 단단하다.",
                causal_pattern={"resistance_to_change": 0.92, "flux_activity": 0.25, "boundary_clarity": 0.85, "valence_shift": 0.65, "relational_coupling": 0.70},
            ),
        ]
        unseen_tests = [
            ("과수원에서 막 딴 달콤하고 아삭한 사과 한 알을 먹었다", "사과", "physical_fruit"),
            ("다툼 후 진심 어린 용서를 구하며 따뜻한 사과를 건넸다", "사과", "social_apology"),
            ("강한 햇살이 비추어 어두운 그림자 없이 시야가 밝다", "밝다", "optical_light"),
            ("다정한 목소리와 웃음소리가 모여 모두의 표정이 밝다", "밝다", "social_mood"),
            ("화강암 바위는 아무리 세게 밀어도 부서지지 않고 단단하다", "단단하다", "physical_mechanics"),
            ("수많은 시련에도 흔들리지 않는 결심과 의지가 단단하다", "단단하다", "mental_resolve"),
        ]
    else:
        symbol_map = {
            "chroma_high": "tok_chroma_7f",
            "luma_high": "tok_luma_2c",
            "rough_high": "tok_rough_9e",
            "stiff_high": "tok_stiff_4b",
            "ring_high": "tok_ring_8d",
        }
        polysemy = {"fruit_apology": "poly_alpha_01", "light_mood": "poly_beta_02", "rock_resolve": "poly_gamma_03"}
        passages = [
            BookPassage(
                passage_id="game_rule_hazard",
                domain_tag="game_manual",
                text="cue_vivid tok_chroma_7f cue_danger cue_avoid",
                causal_pattern={"resistance_to_change": 0.4, "flux_activity": 0.9, "boundary_clarity": 0.8, "valence_shift": -0.7, "relational_coupling": 0.5},
                causal_rule_spec=(["cue_vivid", "tok_chroma_7f"], ["avoid"], ["hazard_repel"]),
            ),
            BookPassage(
                passage_id="game_rule_cleavage",
                domain_tag="game_manual",
                text="cue_grain cue_barrier cue_perp cue_split",
                causal_pattern={"resistance_to_change": 0.7, "flux_activity": 0.6, "boundary_clarity": 0.85, "valence_shift": 0.5, "relational_coupling": 0.6},
                causal_rule_spec=(["cue_grain", "cue_barrier"], ["push_across_grain"], ["open_passage"]),
            ),
            BookPassage(
                passage_id="game_rule_foam",
                domain_tag="game_manual",
                text="cue_dull tok_ring_8d cue_soft cue_compress",
                causal_pattern={"resistance_to_change": 0.2, "flux_activity": 0.5, "boundary_clarity": 0.6, "valence_shift": 0.4, "relational_coupling": 0.4},
                causal_rule_spec=(["tok_ring_8d", "cue_soft"], ["push"], ["compressible_pass"]),
            ),
            BookPassage(
                passage_id="poly_apple_1",
                domain_tag="physical_fruit",
                text="ctx_red poly_alpha_01 ctx_peel ctx_sweet ctx_juice ctx_aroma",
                causal_pattern={"resistance_to_change": 0.45, "flux_activity": 0.35, "boundary_clarity": 0.75, "valence_shift": 0.80, "relational_coupling": 0.70},
            ),
            BookPassage(
                passage_id="poly_apple_2",
                domain_tag="physical_fruit",
                text="ctx_tree ctx_fresh poly_alpha_01 ctx_crisp ctx_sweet ctx_fruit",
                causal_pattern={"resistance_to_change": 0.45, "flux_activity": 0.35, "boundary_clarity": 0.75, "valence_shift": 0.80, "relational_coupling": 0.70},
            ),
            BookPassage(
                passage_id="poly_apology_1",
                domain_tag="social_apology",
                text="ctx_fault ctx_sincere ctx_bow poly_alpha_01 ctx_heart ctx_forgive",
                causal_pattern={"resistance_to_change": 0.30, "flux_activity": 0.50, "boundary_clarity": 0.65, "valence_shift": 0.85, "relational_coupling": 0.90},
            ),
            BookPassage(
                passage_id="poly_apology_2",
                domain_tag="social_apology",
                text="ctx_conflict ctx_forgive ctx_warm poly_alpha_01 ctx_heal ctx_relation",
                causal_pattern={"resistance_to_change": 0.30, "flux_activity": 0.50, "boundary_clarity": 0.65, "valence_shift": 0.85, "relational_coupling": 0.90},
            ),
            BookPassage(
                passage_id="poly_bright_opt_1",
                domain_tag="optical_light",
                text="ctx_sunlight ctx_cliff ctx_shadow ctx_vision poly_beta_02",
                causal_pattern={"resistance_to_change": 0.25, "flux_activity": 0.60, "boundary_clarity": 0.90, "valence_shift": 0.75, "relational_coupling": 0.55},
            ),
            BookPassage(
                passage_id="poly_bright_opt_2",
                domain_tag="optical_light",
                text="ctx_illum ctx_reflect ctx_floor ctx_night poly_beta_02",
                causal_pattern={"resistance_to_change": 0.25, "flux_activity": 0.60, "boundary_clarity": 0.90, "valence_shift": 0.75, "relational_coupling": 0.55},
            ),
            BookPassage(
                passage_id="poly_bright_mood_1",
                domain_tag="social_mood",
                text="ctx_laughter ctx_kind ctx_greeting ctx_mood poly_beta_02",
                causal_pattern={"resistance_to_change": 0.20, "flux_activity": 0.55, "boundary_clarity": 0.80, "valence_shift": 0.85, "relational_coupling": 0.85},
            ),
            BookPassage(
                passage_id="poly_bright_mood_2",
                domain_tag="social_mood",
                text="ctx_song ctx_warm ctx_dialogue ctx_voice poly_beta_02",
                causal_pattern={"resistance_to_change": 0.20, "flux_activity": 0.55, "boundary_clarity": 0.80, "valence_shift": 0.85, "relational_coupling": 0.85},
            ),
            BookPassage(
                passage_id="poly_hard_phys_1",
                domain_tag="physical_mechanics",
                text="ctx_push ctx_granite ctx_rock ctx_unyielding poly_gamma_03",
                causal_pattern={"resistance_to_change": 0.95, "flux_activity": 0.15, "boundary_clarity": 0.90, "valence_shift": 0.20, "relational_coupling": 0.40},
            ),
            BookPassage(
                passage_id="poly_hard_phys_2",
                domain_tag="physical_mechanics",
                text="ctx_tap ctx_mineral ctx_surface ctx_ring poly_gamma_03",
                causal_pattern={"resistance_to_change": 0.95, "flux_activity": 0.15, "boundary_clarity": 0.90, "valence_shift": 0.20, "relational_coupling": 0.40},
            ),
            BookPassage(
                passage_id="poly_hard_mind_1",
                domain_tag="mental_resolve",
                text="ctx_trial ctx_temptation ctx_resolve ctx_will poly_gamma_03",
                causal_pattern={"resistance_to_change": 0.92, "flux_activity": 0.25, "boundary_clarity": 0.85, "valence_shift": 0.65, "relational_coupling": 0.70},
            ),
            BookPassage(
                passage_id="poly_hard_mind_2",
                domain_tag="mental_resolve",
                text="ctx_failure ctx_endure ctx_belief ctx_promise poly_gamma_03",
                causal_pattern={"resistance_to_change": 0.92, "flux_activity": 0.25, "boundary_clarity": 0.85, "valence_shift": 0.65, "relational_coupling": 0.70},
            ),
        ]
        unseen_tests = [
            ("ctx_orchard ctx_sweet ctx_crisp poly_alpha_01 ctx_fruit", "poly_alpha_01", "physical_fruit"),
            ("ctx_conflict ctx_sincere ctx_forgive ctx_warm poly_alpha_01", "poly_alpha_01", "social_apology"),
            ("ctx_sunlight ctx_shadow ctx_vision poly_beta_02", "poly_beta_02", "optical_light"),
            ("ctx_kind ctx_laughter ctx_voice poly_beta_02", "poly_beta_02", "social_mood"),
            ("ctx_granite ctx_rock ctx_push poly_gamma_03", "poly_gamma_03", "physical_mechanics"),
            ("ctx_trial ctx_resolve ctx_will poly_gamma_03", "poly_gamma_03", "mental_resolve"),
        ]

    def oracle_fn(sym: str, observed_delta: Dict[str, float]) -> bool:
        nsym = normalize_token(sym)
        if nsym == normalize_token(symbol_map["chroma_high"]):
            return abs(observed_delta.get("chroma_radius", 0.0)) > 2.0 * abs(observed_delta.get("luminance_a", 0.0))
        if nsym == normalize_token(symbol_map["luma_high"]):
            return abs(observed_delta.get("luminance_a", 0.0)) > 2.0 * abs(observed_delta.get("chroma_radius", 0.0))
        if nsym == normalize_token(symbol_map["rough_high"]):
            return observed_delta.get("fine_roughness", 0.0) > 0.20
        return False

    return {
        "symbol_map": symbol_map,
        "polysemy": polysemy,
        "passages": passages,
        "unseen_tests": unseen_tests,
        "oracle_fn": oracle_fn,
    }


def _load_real_yosemite_scene() -> np.ndarray:
    if YOSEMITE_4K_PATH.exists():
        arr = np.array(Image.open(YOSEMITE_4K_PATH), dtype=np.uint8)
        return arr[600:1350, 1300:2400].copy()
    yy, xx = np.indices((480, 640), dtype=np.float32)
    base = 115.0 + 30.0 * np.sin(0.04 * xx) * np.cos(0.03 * yy)
    return np.stack([base * 0.95, base, base * 0.90], axis=-1).clip(0, 255).astype(np.uint8)


def run_unified_world_game_book_session(use_blind_tokens: bool = False) -> UnifiedSessionReport:
    """
    End-to-end execution of ELYSIA's closed experiential learning & creation loop:
      World -> Book (Pass 1) -> Blind Game (L1 -> L2) -> Book (Unseen) -> Creation -> CausalField
    """
    spine = EpistemicKnowledgeSpine()
    curriculum = build_curriculum_bundle(use_blind_tokens=use_blind_tokens)
    scene_rgb = _load_real_yosemite_scene()

    void_timeline: List[Dict[str, Any]] = []

    def _snapshot(stage: str) -> None:
        void_timeline.append(
            {
                "stage": stage,
                "concepts": len(spine.concepts),
                "symbols": len(spine.symbols),
                "senses": sum(len(v) for v in spine.symbols.values()),
                "open_voids": len(spine.get_open_voids()),
                "resolved_voids": sum(1 for v in spine.voids if v.status == "RESOLVED"),
                "mitosis": len(spine.mitosis_events),
                "verified_rules": sum(1 for r in spine.causal_rules.values() if r.verified_in_reality),
            }
        )

    # 1. WORLD: Saturation re-observation law + Confound self-re-observation Mitosis
    world = WorldPerceptualLearner(spine)
    sat_law = world.discover_saturation_relational_law(scene_rgb)
    axis_bindings = world.ground_and_disambiguate_perceptual_symbols(
        scene_rgb=scene_rgb,
        symbol_map=curriculum["symbol_map"],
        oracle_fn=curriculum["oracle_fn"],
    )
    _snapshot("1_World")

    # 2. BOOK (Pass 1): Real corpus ingestion + multi-context polysemy Mitosis
    book = BookReadingLearner(spine)
    corpus_res = book.read_corpus_directory(CORPUS_DIR)
    target_forms = list(curriculum["polysemy"].values()) + list(curriculum["symbol_map"].values())
    book.read_passages(curriculum["passages"], target_forms=target_forms, mitosis_threshold=0.68)
    _snapshot("2_Book_Read")

    # 3. GAME: Blind Level 1 -> Zero-Shot Rotated/Dimmed Level 2
    assert world.manifold is not None
    game_learner = GamePlayingLearner(spine, world.manifold)
    game_l1 = game_learner.play_level(CausalExplorationGame(level_id=1, seed=42))
    _snapshot("3_Game_L1")
    game_l2 = game_learner.play_level(CausalExplorationGame(level_id=2, seed=43))
    _snapshot("4_Game_L2")

    # 4. CREATION: Self-created Level 3 + Unseen Multi-Axis Visual Creation + Self-Re-Observation
    creator = CreativeSynthesisEngine(spine, world)
    created_l3 = creator.create_and_verify_novel_game_level(game_learner)
    created_vis = creator.create_and_reobserve_novel_visual_composition(
        scene_rgb,
        [curriculum["symbol_map"]["chroma_high"], curriculum["symbol_map"]["rough_high"]],
    )
    _snapshot("5_Creation")

    # 5. BOOK (Pass 2): Unseen context-only disambiguation & Stem/Branch comparison
    dis_acc, dis_details = book.evaluate_disambiguation_benchmark(curriculum["unseen_tests"])
    sense_comparisons = {
        normalize_token(form): spine.compare_symbol_senses(normalize_token(form))
        for form in curriculum["polysemy"].values()
    }
    _snapshot("6_Verify")

    field_obj = spine.project_to_causal_field()

    return UnifiedSessionReport(
        use_blind_tokens=use_blind_tokens,
        symbol_map=curriculum["symbol_map"],
        polysemy_targets=curriculum["polysemy"],
        spine=spine,
        saturation_law=sat_law,
        axis_bindings=axis_bindings,
        corpus_files_read=corpus_res["files_read"],
        sense_comparisons=sense_comparisons,
        disambiguation_accuracy=dis_acc,
        disambiguation_details=dis_details,
        game_level1=game_l1,
        game_level2=game_l2,
        created_level3=created_l3,
        created_visual=created_vis,
        causal_field=field_obj,
        void_timeline=void_timeline,
        scene_rgb=scene_rgb,
    )
