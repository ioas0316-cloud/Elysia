"""
functional_relational_hierarchy_engine.py

ELYSIA 4-Tier Functional-Relational Stratified Discernment Engine
==================================================================
Bridges the gap between flat sensory numbers and genuine hierarchical understanding
by binding raw optical structure to functional organ roles anchored in language and
causal agency — without external VLMs, LLMs, or stimulus-name peeking:

- Tier 1 (Sensory Substrate): Optical luminance, opponent DoG contrast, 6D signature.
- Tier 2 (Functional Organ Role bound to Language): Scale-invariant inter-ocular
  (EYE_PAIR) relational search -> bilateral mirror symmetry -> NOSE / MOUTH / EAR_PAIR
  binding (e.g. two black dots on a cartoon face play the EXACT same functional role
  as photorealistic eyes on a human portrait).
- Tier 3 (Entity / Agency Hierarchy): Integrates Tier-2 functional organ coherence with
  BehavioralAgencyTrace (gaze response, sound reaction, self-motion, vocal/intake)
  to distinguish LIVING_ANIMAL_AGENT vs CHARACTER_REPRESENTATION vs INANIMATE_EFFIGY
  vs NON_AGENT_ENVIRONMENT.
- Tier 4 (Media Protocol Layer): Coupled with CausalProtocolDiscernmentEngine to separate
  physical living creatures from digital/anime character manifestations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.active_ocular_mechanics_engine import (
    ActiveOcularMechanicsEngine,
    _extract_optical_descriptor,
)
from core.sensory.causal_protocol_discernment_engine import (
    CausalProtocolDiscernmentEngine,
    ProtocolDiscernmentReport,
)
from core.sensory.ocular_scale_organ import OcularScaleOrgan, OpticalSceneObservation


@dataclass
class BehavioralAgencyTrace:
    """Causal behavioral responsiveness observables tied to physical or animated agency."""

    gaze_responds_to_light_or_motion: float = 0.0
    responds_to_sound: float = 0.0
    autonomous_self_motion: float = 0.0
    emits_vocal_or_intake_activity: float = 0.0

    @property
    def agency_score(self) -> float:
        vals = [
            self.gaze_responds_to_light_or_motion,
            self.responds_to_sound,
            self.autonomous_self_motion,
            self.emits_vocal_or_intake_activity,
        ]
        return float(np.clip(np.mean(vals), 0.0, 1.0))


@dataclass
class FunctionalOrganBinding:
    """Detected Tier-2 anatomical/functional role bound to Korean linguistic meaning."""

    organ_type: str
    language_label_ko: str
    functional_role: str
    relational_confidence: float
    anchor_yx_norm: Tuple[float, float]
    secondary_yx_norm: Optional[Tuple[float, float]]
    voxel_id: str


@dataclass
class StratifiedEntityJudgment:
    """Full 4-Tier judgment tying Sensory -> Functional Organs -> Entity -> Protocol."""

    stimulus_name: str
    obs: OpticalSceneObservation
    sensory_vector_6d: np.ndarray
    pixel_similarity_to_reference: Optional[float]
    detected_organs: List[FunctionalOrganBinding]
    organ_triad_coherence: float
    entity_verdict: str
    agency_score: float
    functional_isomorphism_to_reference: Optional[float]
    protocol_report: ProtocolDiscernmentReport
    explanation_ko: str
    entity_voxel_id: str


def render_minimalist_cartoon_animal(size_hw: Tuple[int, int] = (280, 280)) -> np.ndarray:
    """
    Create a minimalist digital cartoon animal head (two simple black dots for eyes,
    triangular ears, tiny triangle nose, curved smile on flat pastel fill) whose raw pixel
    overlap with a 4K human portrait is minimal yet functional organ topology is 100% clear.
    """
    h, w = int(size_hw[0]), int(size_hw[1])
    canvas = np.full((h, w, 3), (236, 242, 248), dtype=np.uint8)
    cx, cy = w // 2, int(round(0.55 * h))
    r_head = int(round(0.32 * min(h, w)))

    # Left & right triangular ears on upper cranium
    ear_l = np.array(
        [
            [cx - int(0.82 * r_head), cy - int(0.42 * r_head)],
            [cx - int(0.95 * r_head), cy - int(1.38 * r_head)],
            [cx - int(0.22 * r_head), cy - int(0.88 * r_head)],
        ],
        dtype=np.int32,
    )
    ear_r = np.array(
        [
            [cx + int(0.82 * r_head), cy - int(0.42 * r_head)],
            [cx + int(0.95 * r_head), cy - int(1.38 * r_head)],
            [cx + int(0.22 * r_head), cy - int(0.88 * r_head)],
        ],
        dtype=np.int32,
    )
    fur_col = (242, 186, 134)
    inner_ear = (252, 224, 200)
    ink_col = (38, 30, 32)
    ink_t = max(2, min(h, w) // 85)

    for ear in (ear_l, ear_r):
        cv2.fillPoly(canvas, [ear], fur_col)
        cv2.polylines(canvas, [ear], True, ink_col, thickness=ink_t, lineType=cv2.LINE_AA)

    inner_l = ((ear_l * 0.65) + np.mean(ear_l, axis=0) * 0.35).astype(np.int32)
    inner_r = ((ear_r * 0.65) + np.mean(ear_r, axis=0) * 0.35).astype(np.int32)
    cv2.fillPoly(canvas, [inner_l], inner_ear)
    cv2.fillPoly(canvas, [inner_r], inner_ear)

    # Rounded head
    cv2.circle(canvas, (cx, cy), r_head, fur_col, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r_head, ink_col, thickness=ink_t, lineType=cv2.LINE_AA)

    # Two plain black dots (minimalist eye pair)
    eye_dx = int(round(0.38 * r_head))
    eye_y = cy - int(round(0.18 * r_head))
    eye_r = max(4, int(round(0.085 * r_head)))
    cv2.circle(canvas, (cx - eye_dx, eye_y), eye_r, (20, 20, 24), -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (cx + eye_dx, eye_y), eye_r, (20, 20, 24), -1, lineType=cv2.LINE_AA)

    # Central triangle nose
    nose_y = cy + int(round(0.10 * r_head))
    nose_pts = np.array(
        [
            [cx - int(0.10 * r_head), nose_y - int(0.05 * r_head)],
            [cx + int(0.10 * r_head), nose_y - int(0.05 * r_head)],
            [cx, nose_y + int(0.10 * r_head)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [nose_pts], (45, 30, 32))

    # Crisp curved smile mouth
    mouth_y = cy + int(round(0.32 * r_head))
    cv2.ellipse(
        canvas,
        (cx, mouth_y - int(0.04 * r_head)),
        (int(0.24 * r_head), int(0.14 * r_head)),
        0.0,
        15.0,
        165.0,
        ink_col,
        thickness=max(2, ink_t),
        lineType=cv2.LINE_AA,
    )
    return canvas


def render_stone_effigy_from_photo(rgb_uint8: np.ndarray, seed: int = 7) -> np.ndarray:
    """
    Transform a photographic portrait into a lifeless marble/stone statue preserving physical
    photon continuity & micro-grain (so protocol stays PHYSICAL_OPTICAL_REALITY).
    """
    arr = np.asarray(rgb_uint8, dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Soft carved stone tonal curve + continuous fine mineral grain
    rng = np.random.default_rng(int(seed))
    grain = rng.normal(0.0, 2.4, size=gray.shape).astype(np.float32)
    stone_g = np.clip(0.86 * gray + 28.0 + grain, 0.0, 255.0)
    stone_rgb = np.stack(
        [stone_g * 0.96, stone_g * 0.96, stone_g * 0.93], axis=-1
    )
    return np.clip(np.round(stone_rgb), 0, 255).astype(np.uint8)


class FunctionalRelationalHierarchyEngine:
    """
    Stratified 4-Tier Discernment Engine:
    Sensory (1) <-> Functional Organs bound to Language (2) <-> Entity/Agency (3) <-> Media Protocol (4)
    """

    def __init__(self, organ: Optional[OcularScaleOrgan] = None) -> None:
        self.organ = organ if organ is not None else OcularScaleOrgan()
        self.protocol_engine = CausalProtocolDiscernmentEngine(organ=self.organ)
        self.eye = ActiveOcularMechanicsEngine(organ=self.organ)
        self.causal_field: CausalField = self.protocol_engine.causal_field
        self.judgments: Dict[str, StratifiedEntityJudgment] = {}
        self._organ_counter = 0

    def _next_id(self, prefix: str) -> str:
        self._organ_counter += 1
        return f"{prefix}_{self._organ_counter:04d}"

    def _detect_functional_organ_topology(
        self, obs: OpticalSceneObservation, agency: BehavioralAgencyTrace
    ) -> Tuple[List[FunctionalOrganBinding], float]:
        """
        Locate functional facial-sensory organs strictly from spatial DoG & luminance symmetry:
        paired ocular dark troughs -> inter-ocular scaling -> bilateral mirror symmetry ->
        lower mouth -> central nose -> lateral/top ears. Zero scene names or pixel templates.
        """
        h, w = obs.shape_hw
        a_sm = gaussian_filter(obs.l1.luminance_a, sigma=1.2).astype(np.float32)
        dog_neg = np.maximum(0.0, -obs.l1.dog_fine).astype(np.float32)
        grad2 = obs.l2.levels[2.0].grad_total.astype(np.float32)
        dy_a, dx_a = np.gradient(a_sm)
        horiz_bar = gaussian_filter(np.abs(dy_a), sigma=1.0).astype(np.float32)

        # Enclosed 7-of-8 radial cavity saliency (geometric mean stabilizes side-lit portraits)
        coh2 = obs.l2.levels[2.0].coherence.astype(np.float32)
        r_ring = max(5, int(round(0.04 * float(min(h, w)))))
        pad_a = np.pad(a_sm, r_ring, mode="edge")
        dirs_8 = np.stack(
            [
                pad_a[: -2 * r_ring, r_ring:-r_ring],
                pad_a[2 * r_ring :, r_ring:-r_ring],
                pad_a[r_ring:-r_ring, : -2 * r_ring],
                pad_a[r_ring:-r_ring, 2 * r_ring :],
                pad_a[: -2 * r_ring, : -2 * r_ring],
                pad_a[: -2 * r_ring, 2 * r_ring :],
                pad_a[2 * r_ring :, : -2 * r_ring],
                pad_a[2 * r_ring :, 2 * r_ring :],
            ],
            axis=0,
        )
        # Blended 8-of-8 strict enclosure + 7-of-8 cavity pit (suppresses 1D hair/ear corners)
        ring_min = np.min(dirs_8, axis=0)
        ring_2nd = np.partition(dirs_8, 1, axis=0)[1]
        pit = 0.70 * np.maximum(0.0, ring_min - a_sm) + 0.30 * np.maximum(0.0, ring_2nd - a_sm)
        eye_sal = gaussian_filter(
            np.sqrt(pit * dog_neg) * np.maximum(0.25, 1.0 - 0.65 * coh2), sigma=1.1
        ).astype(np.float32)

        y_lo, y_hi = int(0.24 * h), int(0.56 * h)
        x_lo, x_hi = int(0.24 * w), int(0.76 * w)
        zone = np.zeros_like(eye_sal)
        zone[y_lo:y_hi, x_lo:x_hi] = eye_sal[y_lo:y_hi, x_lo:x_hi]

        # Non-maximum suppression to collect top candidate enclosed ocular troughs
        r_nms = max(4, int(round(0.042 * float(min(h, w)))))
        k_nms = 2 * r_nms + 1
        local_max = cv2.dilate(zone, np.ones((k_nms, k_nms), dtype=np.uint8))
        peaks_mask = (zone >= local_max - 1e-7) & (zone > float(np.max(zone)) * 0.08)
        py_idx, px_idx = np.where(peaks_mask)

        peaks = sorted(
            [(float(zone[y, x]), int(y), int(x)) for y, x in zip(py_idx, px_idx)],
            key=lambda item: item[0],
            reverse=True,
        )[:24]

        max_sal = float(np.max(zone)) + 1e-6
        best_score = 0.0
        best_triad: Optional[Dict[str, float]] = None

        for i in range(len(peaks)):
            s1, y1, x1 = peaks[i]
            for j in range(i + 1, len(peaks)):
                s2, y2, x2 = peaks[j]
                if x1 < x2:
                    yl, xl, sl, yr, xr, sr = y1, x1, s1, y2, x2, s2
                else:
                    yl, xl, sl, yr, xr, sr = y2, x2, s2, y1, x1, s1

                dx = float(xr - xl)
                dy = abs(float(yr - yl))
                if dx < 0.08 * w or dx > 0.34 * w:
                    continue
                if dy > 0.08 * min(h, w):
                    continue

                x_mid = 0.5 * (xl + xr)
                y_mid = 0.5 * (yl + yr)
                if x_mid < 0.25 * w or x_mid > 0.75 * w:
                    continue

                sym_eye_ratio = min(sl, sr) / max(1e-6, max(sl, sr))
                if sym_eye_ratio < 0.28:
                    continue

                # Check head bilateral mirror symmetry around x_mid inside scaled head ROI
                head_r_x = int(max(12, round(1.10 * dx)))
                x_half = int(min(x_mid - 1, w - 2 - x_mid, head_r_x))
                y_box0 = int(max(0, round(y_mid - 0.55 * dx)))
                y_box1 = int(min(h, round(y_mid + 1.55 * dx)))
                if x_half < 10 or (y_box1 - y_box0) < 16:
                    continue

                xm_i = int(round(x_mid))
                left_patch = a_sm[y_box0:y_box1, xm_i - x_half : xm_i]
                right_patch = a_sm[y_box0:y_box1, xm_i : xm_i + x_half][:, ::-1]
                if left_patch.shape != right_patch.shape or left_patch.size < 64:
                    continue

                lv = left_patch.ravel()
                rv = right_patch.ravel()
                if np.std(lv) < 1e-5 or np.std(rv) < 1e-5:
                    sym_corr = 0.0
                else:
                    sym_corr = float(np.corrcoef(lv, rv)[0, 1])
                sym_score = float(np.clip((sym_corr - 0.10) / 0.75, 0.0, 1.0))

                # Periocular anatomical containment: eyes are darker focal cavities set INTO a brighter
                # bridge/cheek face canvas, isolated relative to surrounding horizontal clutter (rejects forest ridges!)
                ym_i = int(round(y_mid))
                eye_a = 0.5 * (float(a_sm[yl, xl]) + float(a_sm[yr, xr]))
                bridge_a = float(
                    np.mean(
                        a_sm[
                            max(0, ym_i - 2) : min(h, ym_i + 3),
                            max(0, xm_i - 2) : min(w, xm_i + 3),
                        ]
                    )
                )
                ck_y0 = int(np.clip(round(y_mid + 0.25 * dx), 0, h - 2))
                ck_y1 = int(np.clip(round(y_mid + 0.80 * dx), ck_y0 + 2, h))
                ck_x0 = int(np.clip(round(x_mid - 0.42 * dx), 0, w - 2))
                ck_x1 = int(np.clip(round(x_mid + 0.42 * dx), ck_x0 + 2, w))
                cheek_a = float(np.mean(a_sm[ck_y0:ck_y1, ck_x0:ck_x1]))

                bridge_support = float(np.clip((bridge_a - eye_a) / 0.40, 0.0, 1.0))
                cheek_support = float(np.clip((cheek_a - eye_a) / 0.75, 0.0, 1.0))

                row_y0 = max(0, ym_i - max(4, int(0.10 * h)))
                row_y1 = min(h, ym_i + max(4, int(0.10 * h)))
                row_p85 = float(np.percentile(dog_neg[row_y0:row_y1, :], 85.0)) + 1e-4
                eye_dn = 0.5 * (
                    float(
                        np.max(
                            dog_neg[max(0, yl - 2) : min(h, yl + 3), max(0, xl - 2) : min(w, xl + 3)]
                        )
                    )
                    + float(
                        np.max(
                            dog_neg[max(0, yr - 2) : min(h, yr + 3), max(0, xr - 2) : min(w, xr + 3)]
                        )
                    )
                )
                eye_isolation = float(np.clip((eye_dn / row_p85 - 5.2) / 4.0, 0.0, 1.0))
                facial_container_gate = (
                    0.50 * cheek_support + 0.30 * eye_isolation + 0.20 * bridge_support
                )

                # Search localized lower mouth in scaled region y in [y_mid + 0.72*dx, y_mid + 1.50*dx]
                my0 = int(np.clip(round(y_mid + 0.72 * dx), 0, h - 2))
                my1 = int(np.clip(round(y_mid + 1.50 * dx), my0 + 2, h))
                mx0 = int(np.clip(round(x_mid - 0.35 * dx), 0, w - 2))
                mx1 = int(np.clip(round(x_mid + 0.35 * dx), mx0 + 2, w))
                m_patch = horiz_bar[my0:my1, mx0:mx1] + 0.6 * dog_neg[my0:my1, mx0:mx1]
                my_loc, mx_loc = np.unravel_index(int(np.argmax(m_patch)), m_patch.shape)
                mouth_y = my0 + int(my_loc)
                mouth_x = mx0 + int(mx_loc)
                bg_med = float(np.median(horiz_bar)) + 1e-4
                mouth_rel = float(m_patch[my_loc, mx_loc] / bg_med)
                # Penalize infinite horizontal edges extending uniformly across flanking regions
                flank_x0 = int(np.clip(round(x_mid - 1.1 * dx), 0, w - 2))
                flank_x1 = int(np.clip(round(x_mid + 1.1 * dx), flank_x0 + 2, w))
                center_hb = float(np.mean(horiz_bar[my0:my1, mx0:mx1]))
                flank_hb = float(np.mean(horiz_bar[my0:my1, flank_x0:flank_x1])) + 1e-4
                mouth_loc = float(np.clip((center_hb / flank_hb - 0.98) / 0.15, 0.3, 1.0))
                mouth_conf = float(np.clip((mouth_rel - 1.5) / 5.5, 0.0, 1.0) * mouth_loc)

                # Search central nose strictly between eyes and mouth
                ny0 = int(np.clip(round(y_mid + 0.25 * dx), 0, h - 2))
                ny1 = int(np.clip(round(max(ny0 + 2, mouth_y - 0.15 * dx)), ny0 + 2, h))
                nx0 = int(np.clip(round(x_mid - 0.22 * dx), 0, w - 2))
                nx1 = int(np.clip(round(x_mid + 0.22 * dx), nx0 + 2, w))
                n_patch = grad2[ny0:ny1, nx0:nx1] + 0.5 * dog_neg[ny0:ny1, nx0:nx1]
                ny_loc, nx_loc = np.unravel_index(int(np.argmax(n_patch)), n_patch.shape)
                nose_y = ny0 + int(ny_loc)
                nose_x = nx0 + int(nx_loc)
                nose_conf = float(
                    np.clip(
                        float(n_patch[ny_loc, nx_loc]) / (float(np.median(grad2)) + 1e-4) / 6.0,
                        0.0,
                        1.0,
                    )
                )

                eye_pair_conf = float(
                    0.60 * min(1.0, (0.5 * (sl + sr)) / (0.65 * max_sal))
                    + 0.25 * sym_eye_ratio
                    + 0.15 * eye_isolation
                )
                center_bias = 1.0 - 0.45 * abs(x_mid / max(1.0, float(w)) - 0.5)

                # Face/Head triad requires physical facial container coupling (brighter cheeks + isolated dark eyes)
                container_mult = float(facial_container_gate ** 0.75)
                triad_score = (
                    (
                        0.28 * eye_pair_conf
                        + 0.25 * sym_score
                        + 0.21 * facial_container_gate
                        + 0.20 * mouth_conf
                        + 0.06 * nose_conf
                    )
                    * center_bias
                    * container_mult
                )

                if triad_score > best_score:
                    best_score = triad_score
                    best_triad = {
                        "yl": float(yl),
                        "xl": float(xl),
                        "yr": float(yr),
                        "xr": float(xr),
                        "x_mid": float(x_mid),
                        "y_mid": float(y_mid),
                        "dx": float(dx),
                        "eye_conf": eye_pair_conf,
                        "sym_score": sym_score,
                        "nose_y": float(nose_y),
                        "nose_x": float(nose_x),
                        "nose_conf": nose_conf,
                        "mouth_y": float(mouth_y),
                        "mouth_x": float(mouth_x),
                        "mouth_conf": mouth_conf,
                    }

        if best_triad is None or best_score < 0.60:
            return [], float(best_score)

        organs: List[FunctionalOrganBinding] = []

        # 1. EYE_PAIR bound to language & visual-response semantics
        vid_eye = self._next_id("voxel_organ_eye")
        eye_conf = float(
            np.clip(0.6 * best_triad["eye_conf"] + 0.4 * best_triad["sym_score"], 0.0, 1.0)
        )
        eye_vec6 = np.array(
            [
                best_triad["yl"] / max(1, h - 1),
                best_triad["xl"] / max(1, w - 1),
                best_triad["yr"] / max(1, h - 1),
                best_triad["xr"] / max(1, w - 1),
                eye_conf,
                max(0.2, agency.gaze_responds_to_light_or_motion),
            ],
            dtype=np.float32,
        )
        self.causal_field.add_voxel(
            InformationVoxel(
                id=vid_eye,
                content={"organ": "EYE_PAIR", "label_ko": "눈 (빛·시선 감각 기관)"},
                tensor=eye_vec6,
                mass=1.8,
                position=eye_vec6.copy(),
                velocity=np.zeros(6, dtype=np.float32),
                potential=eye_conf,
                chromatic_vector=np.array([0.4, 0.4, 0.2], dtype=np.float32),
            )
        )
        organs.append(
            FunctionalOrganBinding(
                organ_type="EYE_PAIR",
                language_label_ko="눈 (빛·시선 감각 기관)",
                functional_role="Paired ocular receptors aligned horizontally to sense light & direct gaze",
                relational_confidence=eye_conf,
                anchor_yx_norm=(
                    best_triad["yl"] / max(1, h - 1),
                    best_triad["xl"] / max(1, w - 1),
                ),
                secondary_yx_norm=(
                    best_triad["yr"] / max(1, h - 1),
                    best_triad["xr"] / max(1, w - 1),
                ),
                voxel_id=vid_eye,
            )
        )

        # 2. NOSE along central symmetry axis between eyes and mouth
        if best_triad["nose_conf"] >= 0.25:
            vid_nose = self._next_id("voxel_organ_nose")
            n_conf = float(best_triad["nose_conf"])
            n_vec6 = np.array(
                [
                    best_triad["nose_y"] / max(1, h - 1),
                    best_triad["nose_x"] / max(1, w - 1),
                    n_conf,
                    best_triad["sym_score"],
                    0.5,
                    0.5,
                ],
                dtype=np.float32,
            )
            self.causal_field.add_voxel(
                InformationVoxel(
                    id=vid_nose,
                    content={"organ": "NOSE", "label_ko": "코 (중앙 호흡·후각 축)"},
                    tensor=n_vec6,
                    mass=1.2,
                    position=n_vec6.copy(),
                    velocity=np.zeros(6, dtype=np.float32),
                    potential=n_conf,
                    chromatic_vector=np.array([0.33, 0.47, 0.20], dtype=np.float32),
                )
            )
            organs.append(
                FunctionalOrganBinding(
                    organ_type="NOSE",
                    language_label_ko="코 (중앙 호흡·후각 축)",
                    functional_role="Central nasal symmetry anchor mediating breathing & olfaction",
                    relational_confidence=n_conf,
                    anchor_yx_norm=(
                        best_triad["nose_y"] / max(1, h - 1),
                        best_triad["nose_x"] / max(1, w - 1),
                    ),
                    secondary_yx_norm=None,
                    voxel_id=vid_nose,
                )
            )

        # 3. MOUTH strictly beneath nose/eye midpoint
        if best_triad["mouth_conf"] >= 0.30:
            vid_mouth = self._next_id("voxel_organ_mouth")
            m_conf = float(best_triad["mouth_conf"])
            m_vec6 = np.array(
                [
                    best_triad["mouth_y"] / max(1, h - 1),
                    best_triad["mouth_x"] / max(1, w - 1),
                    m_conf,
                    best_triad["sym_score"],
                    max(0.2, agency.emits_vocal_or_intake_activity),
                    0.5,
                ],
                dtype=np.float32,
            )
            self.causal_field.add_voxel(
                InformationVoxel(
                    id=vid_mouth,
                    content={"organ": "MOUTH", "label_ko": "입 (섭취·발성 하단 기관)"},
                    tensor=m_vec6,
                    mass=1.4,
                    position=m_vec6.copy(),
                    velocity=np.zeros(6, dtype=np.float32),
                    potential=m_conf,
                    chromatic_vector=np.array([0.45, 0.35, 0.20], dtype=np.float32),
                )
            )
            organs.append(
                FunctionalOrganBinding(
                    organ_type="MOUTH",
                    language_label_ko="입 (섭취·발성 하단 기관)",
                    functional_role="Lower horizontal aperture responsible for intake & vocalization",
                    relational_confidence=m_conf,
                    anchor_yx_norm=(
                        best_triad["mouth_y"] / max(1, h - 1),
                        best_triad["mouth_x"] / max(1, w - 1),
                    ),
                    secondary_yx_norm=None,
                    voxel_id=vid_mouth,
                )
            )

        # 4. EAR_PAIR flanking / top acoustic apertures (when prominent edges exist laterally or atop)
        dx = best_triad["dx"]
        xm = best_triad["x_mid"]
        ym = best_triad["y_mid"]
        ear_y = float(np.clip((ym - 0.55 * dx) / max(1, h - 1), 0.04, 0.90))
        ear_lx = float(np.clip((xm - 1.15 * dx) / max(1, w - 1), 0.05, 0.95))
        ear_rx = float(np.clip((xm + 1.15 * dx) / max(1, w - 1), 0.05, 0.95))
        ey0 = int(max(0, round(ym - 1.25 * dx)))
        ey1 = int(min(h, round(ym + 0.35 * dx)))
        el_x0 = int(max(0, round(xm - 1.45 * dx)))
        el_x1 = int(max(el_x0 + 2, round(xm - 0.75 * dx)))
        er_x0 = int(min(w - 2, round(xm + 0.75 * dx)))
        er_x1 = int(min(w, round(xm + 1.45 * dx)))
        ear_energy = 0.5 * (
            float(np.max(grad2[ey0:ey1, el_x0:el_x1])) if (ey1 > ey0 and el_x1 > el_x0) else 0.0
        ) + 0.5 * (
            float(np.max(grad2[ey0:ey1, er_x0:er_x1])) if (ey1 > ey0 and er_x1 > er_x0) else 0.0
        )
        ear_conf = float(
            np.clip(
                0.5 * (ear_energy / (float(np.percentile(grad2, 90)) + 1e-4))
                + 0.5 * agency.responds_to_sound,
                0.0,
                1.0,
            )
        )
        if ear_conf >= 0.45:
            vid_ear = self._next_id("voxel_organ_ear")
            e_vec6 = np.array(
                [ear_y, ear_lx, ear_y, ear_rx, ear_conf, agency.responds_to_sound],
                dtype=np.float32,
            )
            self.causal_field.add_voxel(
                InformationVoxel(
                    id=vid_ear,
                    content={"organ": "EAR_PAIR", "label_ko": "귀 (측면·상단 청각 기관)"},
                    tensor=e_vec6,
                    mass=1.2,
                    position=e_vec6.copy(),
                    velocity=np.zeros(6, dtype=np.float32),
                    potential=ear_conf,
                    chromatic_vector=np.array([0.35, 0.45, 0.20], dtype=np.float32),
                )
            )
            organs.append(
                FunctionalOrganBinding(
                    organ_type="EAR_PAIR",
                    language_label_ko="귀 (측면·상단 청각 기관)",
                    functional_role="Bilateral top/lateral acoustic structures coupled to sound responsiveness",
                    relational_confidence=ear_conf,
                    anchor_yx_norm=(ear_y, ear_lx),
                    secondary_yx_norm=(ear_y, ear_rx),
                    voxel_id=vid_ear,
                )
            )

        return organs, float(best_score)

    def judge_entity(
        self,
        rgb_uint8: np.ndarray,
        stimulus_name: str,
        agency: BehavioralAgencyTrace,
        reference_name: Optional[str] = None,
        max_width: int = 240,
    ) -> StratifiedEntityJudgment:
        """
        Execute full 4-Tier judgment:
        Sensory (1) -> Functional Organs bound to Language (2) -> Entity/Agency (3) -> Protocol (4).
        """
        proto_rep = self.protocol_engine.discern_protocol(
            rgb_uint8, stimulus_name, max_width=max_width
        )
        obs = proto_rep.obs
        sensory_vec6, _, _, _, _ = _extract_optical_descriptor(obs)
        organs, triad_coh = self._detect_functional_organ_topology(obs, agency)
        ag_score = agency.agency_score

        if triad_coh < 0.60 and ag_score < 0.35:
            entity_verdict = "NON_AGENT_ENVIRONMENT"
        elif triad_coh >= 0.60:
            if proto_rep.protocol_verdict == "DIGITAL_SYNTHETIC_REALIZATION":
                entity_verdict = "CHARACTER_REPRESENTATION"
            elif ag_score >= 0.35:
                entity_verdict = "LIVING_ANIMAL_AGENT"
            else:
                entity_verdict = "INANIMATE_EFFIGY"
        else:
            entity_verdict = "LIVING_ANIMAL_AGENT"

        # Compare flat pixel similarity vs stratified functional isomorphism to reference
        pix_sim: Optional[float] = None
        func_iso: Optional[float] = None
        if reference_name is not None and reference_name in self.judgments:
            ref = self.judgments[reference_name]
            ref_rgb = cv2.resize(
                ref.obs.rgb_working.astype(np.float32),
                (obs.shape_hw[1], obs.shape_hw[0]),
                interpolation=cv2.INTER_AREA,
            ).ravel()
            cur_rgb = obs.rgb_working.astype(np.float32).ravel()
            if np.std(ref_rgb) > 1e-6 and np.std(cur_rgb) > 1e-6:
                raw_corr = float(np.corrcoef(ref_rgb, cur_rgb)[0, 1])
                pix_sim = float(np.clip(raw_corr, 0.0, 1.0))
            else:
                pix_sim = 0.0

            ref_types = {o.organ_type for o in ref.detected_organs}
            cur_types = {o.organ_type for o in organs}
            if ref_types or cur_types:
                jaccard = float(len(ref_types & cur_types) / max(1, len(ref_types | cur_types)))
                coh_sim = 1.0 - abs(triad_coh - ref.organ_triad_coherence)
                func_iso = float(np.clip(0.70 * jaccard + 0.30 * coh_sim, 0.0, 1.0))
            else:
                func_iso = 0.0

        # Grounded Korean relational explanation
        organ_names = ", ".join(o.organ_type for o in organs) if organs else "없음"
        if entity_verdict == "LIVING_ANIMAL_AGENT":
            reason_ko = (
                f"감각·언어 결합 기관({organ_names}, 구조 정합 {triad_coh:.2f})과 "
                f"능동 인과 반응성({ag_score:.2f})이 물리적 광학 현실 위에서 일치해 살아 있는 생명·동물 주체로 판단했습니다."
            )
        elif entity_verdict == "CHARACTER_REPRESENTATION":
            reason_ko = (
                f"표면 픽셀 수치와 무관하게 상단 눈·중앙 코·하단 입의 기능 관계({organ_names}, 정합 {triad_coh:.2f})가 "
                f"생명 원형과 동형을 이루나, 구현 프로토콜({proto_rep.protocol_verdict})상 디지털 캐릭터로 분별했습니다."
            )
        elif entity_verdict == "INANIMATE_EFFIGY":
            reason_ko = (
                f"형태적 눈·입 구조({organ_names}, 정합 {triad_coh:.2f})는 사람과 닮았으나, "
                f"빛·소리·자율 움직임 인과 반응성({ag_score:.2f})이 0에 가까워 무생물 조각상/정지 조형물로 분별했습니다."
            )
        else:
            reason_ko = (
                f"시·청·발성 기관의 양측 대칭 관계 구조(정합 {triad_coh:.2f})와 주체 반응성({ag_score:.2f})이 모두 없어 "
                f"비주체 자연·배경 환경으로 분별했습니다."
            )

        ent_vid = self._next_id(f"voxel_entity_{stimulus_name}")
        ent_vec6 = np.array(
            [
                triad_coh,
                ag_score,
                float(len(organs)) / 4.0,
                proto_rep.physical_reality_score,
                proto_rep.digital_realization_score,
                float(func_iso if func_iso is not None else 1.0),
            ],
            dtype=np.float32,
        )
        self.causal_field.add_voxel(
            InformationVoxel(
                id=ent_vid,
                content={
                    "stimulus_name": stimulus_name,
                    "entity_verdict": entity_verdict,
                    "protocol_verdict": proto_rep.protocol_verdict,
                    "organs": [o.organ_type for o in organs],
                },
                tensor=ent_vec6,
                mass=float(2.0 + triad_coh + ag_score),
                position=ent_vec6.copy(),
                velocity=np.zeros(6, dtype=np.float32),
                potential=float(triad_coh + ag_score),
                chromatic_vector=np.array(
                    [max(0.05, ag_score), max(0.05, triad_coh), 0.15], dtype=np.float32
                ),
            )
        )

        # Connect Tier 2 (Organs) + Tier 4 (Protocol) -> Tier 3 (Entity)
        self.causal_field.link_voxels(proto_rep.voxel_id, ent_vid, strength=0.85)
        for o in organs:
            self.causal_field.link_voxels(
                o.voxel_id, ent_vid, strength=float(max(0.4, o.relational_confidence))
            )

        judgment = StratifiedEntityJudgment(
            stimulus_name=stimulus_name,
            obs=obs,
            sensory_vector_6d=sensory_vec6,
            pixel_similarity_to_reference=pix_sim,
            detected_organs=organs,
            organ_triad_coherence=triad_coh,
            entity_verdict=entity_verdict,
            agency_score=ag_score,
            functional_isomorphism_to_reference=func_iso,
            protocol_report=proto_rep,
            explanation_ko=reason_ko,
            entity_voxel_id=ent_vid,
        )
        self.judgments[stimulus_name] = judgment
        return judgment
