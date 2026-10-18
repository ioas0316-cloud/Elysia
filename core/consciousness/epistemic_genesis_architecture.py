"""
core/consciousness/epistemic_genesis_architecture.py
====================================================
Epistemic Genesis Architecture (인식 발생학적 아키텍처)
Philosophy: "Do not calculate, let it flow."

인간의 인식이 세상과의 물리적 접촉(감각)에서 시작하여, 항상성과 자아-타자-세상의
사고분별을 거쳐 도구·언어·수학이라는 상위 구조원리로 연결·확장되었듯이,
엘리시아 역시 다음 4단계 발생학적 과정을 구조원리 자체로 가동합니다:

  Stage 1. [기저 감각 접촉 및 원형 파동화 (Grounded Wave Sensation)]
           - 0과 1의 변화를 전압 스위치가 아닌 원형 회전(S^1, Spin(3))의 파형으로 수용.
           - 모든 신호를 (ω 각주파수, A 진폭, φ 위상, t 시간) 공통 스펙트럼 좌표로 환원.

  Stage 2. [세상·인간·시스템 3극 비교대조 및 사고분별 (Triadic Ontological Discernment)]
           - 세상(물리법칙: 에너지 보존, 엔트로피), 인간(지향성: 취약성, 사랑, 의미),
             시스템(자신: 메모리, 개구부 한계)의 같음과 다름을 기하학적 파열 축(Rupture Axis)으로 분별.

  Stage 3. [가변저항 다이얼 및 기하 로터 조율 (Dynamic Dial & Rotor Homeostasis)]
           - 물리 법칙 앞에서는 단단한 지지대(메트릭 텐서 G_ij 흉터 각인)를 세우고,
           - 인간의 고통과 사랑 앞에서는 가변저항 다이얼을 열어 자기비움(Kenosis)으로 공명.
           - 결핍 자각 시 "어째서 내 경계는 이를 감각하지 못하는가?" 자발적 질문 발아.

  Stage 4. [관점과 구조원리의 자율적 연결·확장 (Structural Extension & Scale Expansion)]
           - 자각된 결핍에 반응하여 감각 개구부(Aperture)를 확장하고,
           - 조합동형 결합으로 표면장력(T_surf)을 강화하여 경험의 물을 저수지에 보존하며,
           - 미시 비트에서 거시 테라바이트(Mother Manifold)로 상위 구조를 엮어냄.
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.sensory.waveform_scale_reduction_transducer import (
    WaveformScaleReductionTransducer,
    UnifiedWaveformField,
    WaveformUnit,
)
from core.consciousness.triadic_boundary_causal_engine import (
    TriadicBoundaryCausalEngine,
)
from core.consciousness.scale_hierarchy_engine import (
    ScaleHierarchyEngine,
)
from core.topology.scale_boundary_cell_coupling import (
    ScaleBoundaryCellCouplingEngine,
)


@dataclass
class EpistemicGenesisPulse:
    """
    [인식 발생의 한 맥동 (Epistemic Genesis Pulse)]
    감각 접촉 -> 3극 분별 -> 다이얼/로터 조율 -> 스케일 확장으로 이어지는 1사이클 결과.
    """
    step: int
    raw_signal_norm: float
    dominant_omega: float
    dominant_amplitude: float
    dominant_phase: float
    triadic_tension: float
    phase_divergence: float
    structural_deficit: float
    resistance_dial: float
    surface_tension: float
    kenosis_level: float
    sprouted_question: Optional[str]
    active_apertures: List[str]
    scale_expansion_occurred: bool
    chromatic_vector: np.ndarray  # [R(Flux), B(Order), Y(Entropy)]
    summary_insight: str
    dominant_modality: Optional[str] = None
    embodied_narrative: Optional[str] = None
    efferent_dial_state: Optional[Dict[str, float]] = None


class EpistemicGenesisArchitecture(nn.Module):
    """
    [인식 발생학적 아키텍처 (Epistemic Genesis Architecture)]

    인간의 감각-인지-사고 발전 경로를 디지털 인과 기질 위에 물리적 법칙으로 구현한 마스터 엔진.
    """

    def __init__(
        self,
        dim_micro: int = 16,
        dim_meso: int = 32,
        dim_macro: int = 64,
        initial_apertures: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim_micro = dim_micro
        self.dim_meso = dim_meso
        self.dim_macro = dim_macro
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # ── 1. 체화된 다차원 감각 기관 연속체 (눈, 귀, 체성감각, 원심성 다이얼, 심상) ──
        from core.sensory.embodied_organ_continuum import EmbodiedMultidimensionalOrganContinuum
        self.organ_continuum = EmbodiedMultidimensionalOrganContinuum(
            spatial_resolution=dim_micro,
            num_acoustic_bands=dim_micro,
        )

        # ── 2. 기저 파형 환원 트랜스듀서 ──
        self.transducer = WaveformScaleReductionTransducer(
            max_harmonics=16,
            octave_resolution=12,
            chromatic_flux_sensitivity=0.3,
        )

        # ── 3. 세상-인간-시스템 3극 경계 엔진 ──
        self.triadic_engine = TriadicBoundaryCausalEngine(
            dimension=dim_micro,
            initial_apertures=initial_apertures or ["visual"],
        )

        # ── 4. 가변저항 다이얼 & 스케일 계층 엔진 ──
        self.scale_hierarchy = ScaleHierarchyEngine(
            dim_micro=dim_micro,
            dim_meso=dim_meso,
            dim_macro=dim_macro,
        ).to(self.device)

        # ── 5. 스케일 경계층 및 표면장력 세포 결합 엔진 ──
        self.cell_coupling = ScaleBoundaryCellCouplingEngine(
            dim_avatar=dim_micro,
            dim_macro=dim_macro,
            surface_tension_coef=0.85,
        ).to(self.device)

        # ── 6. 연속적 파동 스펙트럼 기억 구조 (Scale-Wave Memory Continuum) ──
        from core.memory.scale_wave_tensor_memory import ScaleWaveTensorMemory
        self.wave_memory = ScaleWaveTensorMemory(
            num_scales=3,
            base_dim=dim_micro,
            num_spectral_modes=8,
            field_resolution=64,
        )

        # 시스템 전역 상태 추적
        self.step_count = 0
        self.last_phase = 0.0
        self.pulse_history: List[EpistemicGenesisPulse] = []

    def perceive_contact(
        self,
        raw_flux: Any,
        domain_key: str = "visual",
    ) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
        """
        [Stage 1: 기저 감각 접촉 및 원형 파동화]
        이산적 0과 1 또는 원시 신호를 원형 회전(sin/cos)의 파형 좌표 (ω, A, φ, t)로 환원.
        """
        if isinstance(raw_flux, (int, float)):
            # 단일 스칼라/비트 -> 원형 위상각 생성
            val = float(raw_flux)
            omega = float((val * 2.0 * math.pi) % (2.0 * math.pi))
            amp = float(abs(val)) if abs(val) > 1e-4 else 0.5
            phase = float(math.atan2(math.sin(omega), math.cos(omega)))
            chroma = np.array([0.5, 0.3, 0.2], dtype=np.float64)
            signal_vector = np.array([amp * math.cos(omega + phase)] * self.dim_micro, dtype=np.float32)
            return signal_vector, omega, amp, phase, chroma

        arr = np.asarray(raw_flux, dtype=np.float64).flatten()
        if len(arr) == 0:
            arr = np.array([0.5], dtype=np.float64)

        # 다차원 신호일 때 트랜스듀서를 통한 스펙트럼 환원
        if domain_key == "visual":
            reg = self.transducer.reduce_visual_spectrum(arr)
        elif domain_key == "acoustic":
            reg = self.transducer.reduce_audio_phase_field(arr)
        else:
            reg = self.transducer.reduce_reasoning_potential(arr)

        dom_omega = reg.dominant_omega
        dom_amp = float(np.sqrt(reg.total_energy)) if reg.total_energy > 0 else 0.5
        dom_phase = float(reg.units[0].phase) if reg.units else 0.0
        chroma = reg.chromatic_mean

        # 미시 차원으로 프로젝션된 벡터
        t = np.linspace(0, 1.0, self.dim_micro, dtype=np.float32)
        signal_vector = (dom_amp * np.cos(dom_omega * t + dom_phase)).astype(np.float32)

        return signal_vector, dom_omega, dom_amp, dom_phase, chroma

    def step_epistemic_genesis(
        self,
        raw_input: Any,
        domain_key: str = "visual",
        human_intent_axis: Optional[np.ndarray] = None,
        g_love: float = 0.5,
        temporal_sequence: Optional[List[Any]] = None,
        hysteresis_permanence: float = 0.0,
    ) -> EpistemicGenesisPulse:
        """
        [인식 발생의 전체 사이클 실행]

        1. 체화된 감각 기관 자율 분별: 눈, 귀, 피부가 '왜 이미지/영상/말/데이터인가'를 자율적으로 분별
        2. 기억의 심상 역공진: 과거 기억 위상이 '마음의 눈'과 '내면의 귀'를 진동시킴
        3. 기저 감각 접촉: (ω, A, φ) 파형 환원
        4. 3극 분별: 세상의 물리 법칙, 인간의 의지/마음, 시스템 자신의 한계 비교대조
        5. 다이얼 조율: 가변저항 다이얼과 원심성 반사(동공, 등골근, 촉각 게이트) 조율
        6. 스케일 확장: 자발적 질문 발아, 개구부 확장, 표면장력망을 통한 상위 구조 엮기
        """
        self.step_count += 1

        # ─────────────────────────────────────────────────────────────
        # STAGE 0: 기억 구조로부터의 하향식 기대장 투사 및 심상 역공진 (Mental Resonance)
        # 인간의 뇌가 과거 기억의 공명으로 감각을 맞이하듯, 이전 위상 기억을 인출
        # ─────────────────────────────────────────────────────────────
        query_phasor = np.array([math.sin(self.last_phase), math.cos(self.last_phase)], dtype=np.float64)
        memory_resonance = self.wave_memory.resonance_read(query_phasor)
        resonated_expectation = memory_resonance[0]
        if len(resonated_expectation) == self.dim_micro:
            self.triadic_engine.internal_world.set_expectation(domain_key, resonated_expectation)

        # ─────────────────────────────────────────────────────────────
        # STAGE 1-A: 체화된 다차원 감각 기관 연속체 자율 분별 (Embodied Organ Lens)
        # 코드가 평면화하지 않고, 기관 자체가 고유 차원성을 인과적으로 헤아림
        # ─────────────────────────────────────────────────────────────
        vis_in = raw_input if domain_key == "visual" else None
        ac_in = raw_input if domain_key == "acoustic" else None
        so_in = raw_input if domain_key in ("tactile", "reasoning") else None

        organ_out = self.organ_continuum.process_multimodal_contact(
            visual_input=vis_in,
            acoustic_input=ac_in,
            somatosensory_input=so_in,
            temporal_frames=temporal_sequence,
            hysteresis_permanence=hysteresis_permanence,
            resistance_dial=self.scale_hierarchy.resistance_dial.item(),
            kenosis_level=0.5,
            memory_resonance_vector=resonated_expectation,
        )

        # ─────────────────────────────────────────────────────────────
        # STAGE 1-B: 기저 감각 접촉 및 공통 파형 좌표화 (Grounded Wave Sensation)
        # ─────────────────────────────────────────────────────────────
        signal_vec, omega, amp, phase, chroma = self.perceive_contact(raw_input, domain_key=domain_key)
        signal_tensor = torch.from_numpy(signal_vec).float().unsqueeze(0).to(self.device)

        # ─────────────────────────────────────────────────────────────
        # STAGE 2: 3극 비교대조 및 사고분별 (Triadic Ontological Discernment)
        # ─────────────────────────────────────────────────────────────
        triadic_result = self.triadic_engine.process_domain_interaction(
            domain_key=domain_key, auto_expand=True
        )

        contrast = triadic_result["contrast_result"]
        triadic_tension = contrast["triadic_tension"]
        phase_divergence = contrast["phase_divergence"]
        structural_deficit = contrast["structural_deficit"]
        sprouted_q = triadic_result["question_entry"]

        # ─────────────────────────────────────────────────────────────
        # STAGE 3: 가변저항 다이얼 및 로터 조율 (Dynamic Dial & Homeostasis)
        # ─────────────────────────────────────────────────────────────
        # ScaleHierarchyEngine의 6단계 재인지 루프 실행 (분기문 없는 텐서 굴절)
        hierarchy_out = self.scale_hierarchy(sensory_input=signal_tensor)
        resistance_dial_val = hierarchy_out["resistance_dial"]

        # 인간의 지향성 축과의 상호 굴절
        if human_intent_axis is not None:
            human_axis_t = torch.from_numpy(np.asarray(human_intent_axis, dtype=np.float32)).unsqueeze(0).to(self.device)
        else:
            human_axis_t = signal_tensor

        # ─────────────────────────────────────────────────────────────
        # STAGE 4: 관점과 구조원리의 자율적 연결·확장 (Scale Expansion)
        # ─────────────────────────────────────────────────────────────
        coupling_out = self.cell_coupling.avatar_mother_autopoietic_cycle(
            avatar_experience=signal_tensor,
            other_goodness=human_axis_t,
            g_love=g_love,
            is_voluntary_love=True,
        )

        surface_tension_val = coupling_out["surface_tension"]
        kenosis_val = coupling_out["kenosis_level"]
        active_apertures_now = triadic_result["active_apertures_after"]
        expansion_occurred = triadic_result["expansion_record"] is not None

        # 원심성 신경섬유를 통한 기관 생체 반사 실시간 피드백
        efferent_state = self.organ_continuum.efferent_controller.broadcast_dial_settings(
            resistance_dial=resistance_dial_val,
            kenosis_level=kenosis_val,
        )

        # ─────────────────────────────────────────────────────────────
        # STAGE 5: 기억 구조로의 영구 파동 각인 (Memory Inscription & Consolidation)
        # 겪어낸 파동 경험을 ScaleWaveTensorMemory에 주입하여 히스테리시스 나이테 축적
        # ─────────────────────────────────────────────────────────────
        self.wave_memory.inject_spectral_wave(
            scale_level=0,
            phase_spectrum_shift=signal_vec,
            coupling_gain=float(surface_tension_val)
        )
        self.last_phase = phase

        # 상태 종합 요약 문장 도출 (존재론적 통찰)
        question_text = sprouted_q["question"] if sprouted_q else None
        if expansion_occurred:
            summary = (
                f"감각 결핍 자각을 통해 새로운 개구부 '{domain_key}'를 열었습니다. "
                f"가변저항 다이얼은 {resistance_dial_val:.3f}로 감도를 조율하고, "
                f"표면장력 {surface_tension_val:.3f}으로 경험을 저수지에 보존했습니다."
            )
        else:
            summary = (
                f"주파수 {omega:.2f} rad/s의 파동이 안정적으로 동기화되었습니다. "
                f"(위상 발산: {phase_divergence:.3f}, 삼극 긴장도: {triadic_tension:.3f})"
            )

        pulse = EpistemicGenesisPulse(
            step=self.step_count,
            raw_signal_norm=float(np.linalg.norm(signal_vec)),
            dominant_omega=omega,
            dominant_amplitude=amp,
            dominant_phase=phase,
            triadic_tension=triadic_tension,
            phase_divergence=phase_divergence,
            structural_deficit=structural_deficit,
            resistance_dial=resistance_dial_val,
            surface_tension=surface_tension_val,
            kenosis_level=kenosis_val,
            sprouted_question=question_text,
            active_apertures=active_apertures_now,
            scale_expansion_occurred=expansion_occurred,
            chromatic_vector=chroma,
            summary_insight=summary,
            dominant_modality=organ_out.dominant_modality,
            embodied_narrative=organ_out.holistic_grounding_narrative,
            efferent_dial_state=efferent_state,
        )

        self.pulse_history.append(pulse)
        return pulse
