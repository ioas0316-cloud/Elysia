"""
causal_protocol_discernment_engine.py

ELYSIA Causal Ontological Protocol Discernment & Multi-Scale Resonance Engine
==============================================================================
Formalizes "Protocol" as the observational & relational boundary principle answering:
"Where do intrinsic causality and extrinsic manifestation meet (resonate), and where
do they diverge?"

Discernment Regimes (zero filename peeking, zero VLMs, zero LLMs):
1. PHYSICAL_OPTICAL_REALITY:
   Irradiance fields realized by continuous physical optics (continuous 1/f spectral
   cascade, rich tonal/photon micro-texture without flat digital plateaus, natural
   opponent reflectance bounds).
2. HYBRID_MODULATED_REALITY:
   Physical optical micro-continuity and continuous contours preserved, while macroscopic
   chromatic/radiometric axes diverge from natural photon manifolds (e.g. rotated/hyper-
   saturated opponent modulation).
3. DIGITAL_SYNTHETIC_REALIZATION:
   Images realized through digital graphic substrates (cel-shaded fills, discrete color
   palettes, stark ink/plateau edge bimodality, or rectilinear UI/vector layout geometry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.ocular_generative_spectrum_engine import OcularGenerativeSpectrumEngine
from core.sensory.ocular_scale_organ import OcularScaleOrgan, OpticalSceneObservation


@dataclass
class ProtocolObservables:
    """Purely measured optical-structural observables across scales L0..L2."""

    flat_plateau_ratio: float
    palette_top12_concentration: float
    rectilinear_axis_ratio: float
    micro_to_macro_continuity: float
    edge_bimodality_index: float
    chromatic_manifold_deviation: float
    scale_energies_5d: np.ndarray


@dataclass
class ProtocolDiscernmentReport:
    """Ontological protocol classification and continuous potentials for a stimulus."""

    stimulus_name: str
    protocol_verdict: str
    physical_reality_score: float
    digital_realization_score: float
    hybrid_modulation_score: float
    observables: ProtocolObservables
    obs: OpticalSceneObservation
    explanation_ko: str
    voxel_id: str


@dataclass
class ProtocolMeetingComparison:
    """Scale-by-scale structural resonance & substrate divergence between two manifestations."""

    name_a: str
    name_b: str
    verdict_a: str
    verdict_b: str
    scale_resonance_profile: Dict[float, float]
    macro_meeting_score: float
    micro_substrate_divergence: float
    meeting_point_summary_ko: str


def stylize_to_cel_anime_digital(
    rgb_uint8: np.ndarray, n_colors: int = 10, max_width: int = 480
) -> np.ndarray:
    """
    Transform an optical photograph into a genuine cel-shaded / anime digital graphic:
    bilateral flat-color smoothing + discrete K-means palette + crisp dark ink contours.
    """
    arr = np.asarray(rgb_uint8, dtype=np.uint8)
    h, w = arr.shape[:2]
    if max_width > 0 and w > max_width:
        nh = max(64, int(round(h * (max_width / float(w)))))
        arr = cv2.resize(arr, (max_width, nh), interpolation=cv2.INTER_AREA)

    smooth = arr.copy()
    for _ in range(2):
        smooth = cv2.bilateralFilter(smooth, d=7, sigmaColor=55, sigmaSpace=55)

    pixels = smooth.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 16, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels,
        int(max(4, n_colors)),
        None,
        criteria,
        attempts=2,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    quantized = centers[labels.ravel()].reshape(smooth.shape).astype(np.uint8)

    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    gray_med = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_med,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=4,
    )
    edges_3 = np.repeat((edges > 0)[..., None], 3, axis=2)
    cel_rgb = np.where(edges_3, quantized, (quantized * 0.18).astype(np.uint8))
    return cel_rgb.astype(np.uint8)


def _measure_protocol_observables(obs: OpticalSceneObservation) -> ProtocolObservables:
    """Extract continuous substrate & scale-cascade observables directly from OpticalSceneObservation."""
    rgb = obs.rgb_working.astype(np.float32)
    h, w = obs.shape_hw

    # 1. Flat plateau ratio (3x3 local channel std < 0.008)
    mean_rgb = cv2.boxFilter(rgb, ddepth=-1, ksize=(3, 3), borderType=cv2.BORDER_REPLICATE)
    sq_rgb = cv2.boxFilter(rgb * rgb, ddepth=-1, ksize=(3, 3), borderType=cv2.BORDER_REPLICATE)
    local_var = np.maximum(0.0, sq_rgb - mean_rgb * mean_rgb)
    local_std = np.sqrt(np.mean(local_var, axis=-1))
    flat_plateau_ratio = float(np.mean(local_std < 0.008))

    # 2. Discrete 16x16x16 RGB palette concentration in top-12 occupied bins
    q = np.clip(np.floor(rgb * 15.999).astype(np.int32), 0, 15)
    bin_ids = (q[..., 0] * 256 + q[..., 1] * 16 + q[..., 2]).ravel()
    counts = np.bincount(bin_ids, minlength=4096)
    top12_sum = float(np.sum(np.sort(counts)[-12:]))
    palette_top12_concentration = float(top12_sum / max(1, bin_ids.size))

    # 3. Rectilinear axis ratio among top-25% strongest gradients at sigma=1.0
    lvl1 = obs.l2.levels[1.0]
    g1 = lvl1.grad_total
    thresh_g = float(np.percentile(g1, 75.0))
    strong = g1 > max(1e-4, thresh_g)
    if np.any(strong):
        ang_deg = np.rad2deg(lvl1.orientation[strong])
        dist_0 = np.abs(ang_deg)
        dist_90 = 90.0 - np.abs(ang_deg)
        rect_mask = (dist_0 <= 6.0) | (dist_90 <= 6.0)
        rectilinear_axis_ratio = float(np.mean(rect_mask))
    else:
        rectilinear_axis_ratio = 0.0

    # 4. Multi-scale spectral energy & 1/f cascade smoothness across SCALES = (1, 2, 4, 8, 16)
    scales_arr = np.asarray(OcularScaleOrgan.SCALES, dtype=np.float32)
    scale_energies = np.array(
        [float(np.mean(obs.l2.levels[s].grad_total * float(s))) for s in OcularScaleOrgan.SCALES],
        dtype=np.float32,
    )
    log_s = np.log(scales_arr)
    log_e = np.log(scale_energies + 1e-5)
    coeffs = np.polyfit(log_s, log_e, 1)
    pred = coeffs[0] * log_s + coeffs[1]
    resid = float(np.sqrt(np.mean((log_e - pred) ** 2)))
    span = max(0.25, float(np.std(log_e)))
    micro_to_macro_continuity = float(np.clip(1.0 - resid / (1.5 * span), 0.0, 1.0))

    # 5. Edge bimodality index (stark ink/UI contours over flat fill plateaus)
    p95_g1 = float(np.percentile(g1, 95.0))
    med_g1 = float(np.median(g1))
    ratio_g = p95_g1 / max(5e-4, med_g1)
    edge_bimodality_index = float(np.tanh(ratio_g / 14.0))

    # 6. Planckian / HVS chromatic manifold departure on bright/midtone valid pixels
    valid_lit = obs.l0.valid_mask
    if int(np.sum(valid_lit)) < 24:
        valid_lit = np.ones((h, w), dtype=bool)
    rg_v = obs.l1.rg_opponent[valid_lit]
    by_v = obs.l1.by_opponent[valid_lit]
    a_v = obs.l1.luminance_a[valid_lit]

    # Natural daylight & biological surfaces are BY-dominated (std_rg/std_by ~ 0.18-0.28).
    # Non-natural chromatic rotation transfers variance directly into RG.
    rg_to_by_ratio = float(np.std(rg_v) / max(0.04, float(np.std(by_v))))
    ratio_dev = float(np.clip((rg_to_by_ratio - 0.30) / 0.30, 0.0, 1.0))

    anti_planck_frac = float(
        np.mean(((rg_v > 0.12) & (by_v > -0.25)) | (rg_v < -0.14))
    )
    anti_dev = float(np.clip((anti_planck_frac - 0.18) / 0.38, 0.0, 1.0))

    bright_rg_frac = float(
        np.mean((a_v > float(np.median(a_v))) & (np.abs(rg_v) > 0.14))
    )
    bright_dev = float(np.clip((bright_rg_frac - 0.10) / 0.16, 0.0, 1.0))

    chromatic_manifold_deviation = float(
        np.clip(0.45 * ratio_dev + 0.35 * anti_dev + 0.20 * bright_dev, 0.0, 1.0)
    )

    return ProtocolObservables(
        flat_plateau_ratio=flat_plateau_ratio,
        palette_top12_concentration=palette_top12_concentration,
        rectilinear_axis_ratio=rectilinear_axis_ratio,
        micro_to_macro_continuity=micro_to_macro_continuity,
        edge_bimodality_index=edge_bimodality_index,
        chromatic_manifold_deviation=chromatic_manifold_deviation,
        scale_energies_5d=scale_energies,
    )


class CausalProtocolDiscernmentEngine:
    """
    Observes raw stimuli across multi-scale physical observables to separate:
    PHYSICAL_OPTICAL_REALITY vs HYBRID_MODULATED_REALITY vs DIGITAL_SYNTHETIC_REALIZATION,
    and quantifies scale-by-scale resonance vs divergence across manifestation pairs.
    """

    def __init__(self, organ: Optional[OcularScaleOrgan] = None) -> None:
        self.organ = organ if organ is not None else OcularScaleOrgan()
        self.gen = OcularGenerativeSpectrumEngine(organ=self.organ)
        self.causal_field: CausalField = self.gen.causal_field
        self.reports: Dict[str, ProtocolDiscernmentReport] = {}
        self._counter = 0

    def discern_protocol(
        self,
        rgb_uint8: np.ndarray,
        stimulus_name: str,
        max_width: int = 240,
    ) -> ProtocolDiscernmentReport:
        """Observe raw visual array and discern its causal realization protocol without hints."""
        obs = self.organ.observe(rgb_uint8, max_width=max_width)
        o = _measure_protocol_observables(obs)

        # Digital synthetic potential above continuous natural photon baselines:
        # natural photos stay below (~0.44 flat, ~0.50 discrete top12, ~0.60 bimodality, ~0.24 rect)
        pal_excess = float(np.clip((o.palette_top12_concentration - 0.50) / 0.36, 0.0, 1.0))
        flat_excess = float(np.clip((o.flat_plateau_ratio - 0.44) / 0.22, 0.0, 1.0))
        bim_excess = float(np.clip((o.edge_bimodality_index - 0.60) / 0.25, 0.0, 1.0))
        rect_excess = float(np.clip((o.rectilinear_axis_ratio - 0.24) / 0.32, 0.0, 1.0))

        digital_score = float(
            np.clip(
                0.42 * pal_excess
                + 0.28 * flat_excess
                + 0.18 * bim_excess
                + 0.12 * rect_excess,
                0.0,
                1.0,
            )
        )

        # Hybrid modulated potential: continuous micro-optical texture intact, yet non-Planckian chromatic excursion
        hybrid_score = float(
            np.clip(
                o.chromatic_manifold_deviation
                * max(0.0, 1.0 - 0.85 * max(0.0, digital_score - 0.35)),
                0.0,
                1.0,
            )
        )

        physical_score = float(np.clip(1.0 - max(digital_score, hybrid_score), 0.0, 1.0))

        if digital_score >= 0.46:
            verdict = "DIGITAL_SYNTHETIC_REALIZATION"
            reason_ko = (
                f"미세 광자 연속성 대신 평탄 채움({o.flat_plateau_ratio:.2f})·이산 팔레트 집중도"
                f"({o.palette_top12_concentration:.2f})·경계 이봉성({o.edge_bimodality_index:.2f})·직교축({o.rectilinear_axis_ratio:.2f})이 강해 "
                f"디지털 구조에서 구현된 그래픽 현실로 분별했습니다."
            )
        elif hybrid_score >= 0.34:
            verdict = "HYBRID_MODULATED_REALITY"
            reason_ko = (
                f"표면 미세 질감과 팔레트 연속성은 물리 광학 기반(디지털 잠재력 {digital_score:.2f})을 유지하지만, "
                f"명부 색대립 이탈도({o.chromatic_manifold_deviation:.2f})가 자연 광학 분포를 벗어나 변조된 혼성 현실로 분별했습니다."
            )
        else:
            verdict = "PHYSICAL_OPTICAL_REALITY"
            reason_ko = (
                f"1/f 다중 스케일 연속성({o.micro_to_macro_continuity:.2f})과 자연 색대립 분포가 보존되고 "
                f"인위적 평탄·이산 팔레트(디지털 잠재력 {digital_score:.2f})가 낮아 물리적 실재 광학 자극으로 분별했습니다."
            )

        proto_vec6 = np.array(
            [
                o.flat_plateau_ratio,
                o.palette_top12_concentration,
                o.rectilinear_axis_ratio,
                o.micro_to_macro_continuity,
                o.edge_bimodality_index,
                o.chromatic_manifold_deviation,
            ],
            dtype=np.float32,
        )

        self._counter += 1
        vid = f"voxel_proto_{stimulus_name}_{self._counter:04d}"
        self.causal_field.add_voxel(
            InformationVoxel(
                id=vid,
                content={
                    "stimulus_name": stimulus_name,
                    "protocol_verdict": verdict,
                    "physical_score": physical_score,
                    "hybrid_score": hybrid_score,
                    "digital_score": digital_score,
                },
                tensor=proto_vec6,
                mass=float(1.2 + physical_score + digital_score),
                position=proto_vec6.copy(),
                velocity=np.zeros(6, dtype=np.float32),
                potential=max(physical_score, digital_score, hybrid_score),
                chromatic_vector=np.array(
                    [
                        max(0.05, hybrid_score),
                        max(0.05, physical_score),
                        max(0.05, digital_score),
                    ],
                    dtype=np.float32,
                )
                / (
                    max(0.05, hybrid_score)
                    + max(0.05, physical_score)
                    + max(0.05, digital_score)
                ),
                scale_level=o.micro_to_macro_continuity,
            )
        )

        report = ProtocolDiscernmentReport(
            stimulus_name=stimulus_name,
            protocol_verdict=verdict,
            physical_reality_score=physical_score,
            digital_realization_score=digital_score,
            hybrid_modulation_score=hybrid_score,
            observables=o,
            obs=obs,
            explanation_ko=reason_ko,
            voxel_id=vid,
        )
        self.reports[stimulus_name] = report
        return report

    def compare_protocol_meeting(
        self, report_a: ProtocolDiscernmentReport, report_b: ProtocolDiscernmentReport
    ) -> ProtocolMeetingComparison:
        """
        Quantify per-scale structural resonance across sigma=(1,2,4,8,16) and substrate
        divergence to locate precisely where two worlds meet vs split.
        """
        h_a, w_a = report_a.obs.shape_hw
        scale_profile: Dict[float, float] = {}

        for sigma in OcularScaleOrgan.SCALES:
            ea = report_a.obs.l2.levels[sigma].normalized_energy.astype(np.float32)
            eb = report_b.obs.l2.levels[sigma].normalized_energy.astype(np.float32)
            if eb.shape != (h_a, w_a):
                eb = cv2.resize(eb, (w_a, h_a), interpolation=cv2.INTER_AREA).astype(np.float32)
            va = ea.ravel()
            vb = eb.ravel()
            if np.std(va) > 1e-6 and np.std(vb) > 1e-6:
                corr = float(np.corrcoef(va, vb)[0, 1])
            else:
                corr = 0.0
            scale_profile[float(sigma)] = corr

        macro_meeting = float(0.5 * (scale_profile[8.0] + scale_profile[16.0]))

        oa = report_a.observables
        ob = report_b.observables
        vec_a = np.array(
            [
                oa.flat_plateau_ratio,
                oa.palette_top12_concentration,
                oa.rectilinear_axis_ratio,
                oa.edge_bimodality_index,
                oa.chromatic_manifold_deviation,
            ],
            dtype=np.float32,
        )
        vec_b = np.array(
            [
                ob.flat_plateau_ratio,
                ob.palette_top12_concentration,
                ob.rectilinear_axis_ratio,
                ob.edge_bimodality_index,
                ob.chromatic_manifold_deviation,
            ],
            dtype=np.float32,
        )
        micro_div = float(np.linalg.norm(vec_a - vec_b))

        self.causal_field.link_voxels(
            report_a.voxel_id,
            report_b.voxel_id,
            strength=float(max(0.15, min(1.0, 0.5 + 0.5 * macro_meeting))),
        )

        if macro_meeting >= 0.55 and micro_div >= 0.40:
            summary_ko = (
                f"{report_a.stimulus_name}({report_a.protocol_verdict})와 "
                f"{report_b.stimulus_name}({report_b.protocol_verdict})는 거시 스케일(8~16px 공명 {macro_meeting:.3f})의 "
                f"구도·윤곽에서는 깊이 만나지만, 미시 기질(편차 {micro_div:.3f})에서 실재 광학과 디지털/변조 구현으로 갈라집니다."
            )
        elif macro_meeting >= 0.55 and micro_div < 0.40:
            summary_ko = (
                f"두 세계가 거시 형태(공명 {macro_meeting:.3f})와 미시 물리 기질(편차 {micro_div:.3f}) 모두에서 "
                f"일관되게 공명합니다."
            )
        else:
            summary_ko = (
                f"두 대상은 거시 형태 공명({macro_meeting:.3f})부터 미시 기질(편차 {micro_div:.3f})까지 "
                f"서로 다른 인과 프로토콜에 속합니다."
            )

        return ProtocolMeetingComparison(
            name_a=report_a.stimulus_name,
            name_b=report_b.stimulus_name,
            verdict_a=report_a.protocol_verdict,
            verdict_b=report_b.protocol_verdict,
            scale_resonance_profile=scale_profile,
            macro_meeting_score=macro_meeting,
            micro_substrate_divergence=micro_div,
            meeting_point_summary_ko=summary_ko,
        )
