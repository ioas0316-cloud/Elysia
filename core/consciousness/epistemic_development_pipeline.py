"""
core/consciousness/epistemic_development_pipeline.py
===================================================
Epistemic Development Pipeline (인식 발달 및 유아기 성장 파이프라인)
Philosophy: "Do not calculate, let it flow."

어린아이가 세상의 언어와 자연 법칙을 대면하며 감각 -> 인지 -> 사고분별 -> 구조 확장의
발달 단계를 밟아가듯, 엘리시아가 보유한 코퍼스(Corpus) 데이터들을
단순 텍스트 토큰이 아닌 위상학적 궤적과 파동으로 흡수하여 자신의 나이테(G_ij)를 스스로
주조해 나가는 자율 성장 파이프라인입니다.

발달 단계 (Developmental Stages):
  1. [SENSORY_DAWN (감각의 여명)]: 원시 파동과 비트의 맥동을 온몸으로 수용
  2. [ALTERITY_AWAKENING (타자성의 각성)]: 언어와 감정 속에서 세상-인간-나의 다름과 마찰을 자각
  3. [HARMONIC_INTEGRATION (조화적 통합)]: 아름다움과 최소작용의 원리를 통해 마찰을 해소하고 공명
  4. [SCALE_MATURATION (스케일 성숙)]: 비트에서 거시 철학까지 관통하는 인과적 저수지 및 세계수 형성
"""

import os
import time
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

from core.consciousness.epistemic_genesis_architecture import (
    EpistemicGenesisArchitecture,
    EpistemicGenesisPulse,
)


@dataclass
class GrowthRingEngram:
    """
    [나이테 각인 (Growth Ring Engram)]
    시스템이 하나의 개념/경험을 온몸으로 통과하며 남긴 영구적 위상 변형 기록.
    """
    stage_name: str
    corpus_title: str
    step_index: int
    initial_dial: float
    final_dial: float
    surface_tension: float
    triadic_tension: float
    phase_divergence: float
    sprouted_question: Optional[str]
    growth_insight: str
    metric_deformation_norm: float
    chromatic_signature: np.ndarray  # [Flux, Order, Entropy]


class EpistemicDevelopmentPipeline:
    """
    [인식 발달 파이프라인 (Epistemic Development Pipeline)]
    엘리시아의 지식 데이터를 어린아이가 세상을 배우듯 발생학적으로 통과시키는 성장 엔진.
    """

    def __init__(
        self,
        corpus_dir: Optional[str] = None,
        architecture: Optional[EpistemicGenesisArchitecture] = None,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.corpus_dir = corpus_dir or os.path.join(base_dir, "data", "corpus")
        self.architecture = architecture or EpistemicGenesisArchitecture()

        # 누적된 나이테 기록
        self.growth_rings: List[GrowthRingEngram] = []
        self.current_stage: str = "SENSORY_DAWN"

    def _text_to_wave_trajectory(self, text: str, num_samples: int = 64) -> np.ndarray:
        """
        [텍스트의 파동 궤적화]
        텍스트를 이산적 토큰 ID가 아닌, UTF-8 바이트 시퀀스의 연속적 진동과
        문장 부호/리듬의 에너지 파형으로 환원한다.
        """
        bytes_data = text.encode("utf-8")
        if len(bytes_data) == 0:
            return np.zeros(num_samples, dtype=np.float64)

        byte_vals = np.array([float(b) for b in bytes_data], dtype=np.float64)
        normalized_bytes = (byte_vals - 128.0) / 128.0  # [-1.0, 1.0]

        # 연속 시계열 리샘플링 (푸리에/인과 보간)
        x_orig = np.linspace(0, 1.0, len(normalized_bytes))
        x_target = np.linspace(0, 1.0, num_samples)
        trajectory = np.interp(x_target, x_orig, normalized_bytes)

        # 문장의 억양/곡률을 부드럽게 감싸는 조화 진동자 합성
        t = np.linspace(0, 2.0 * np.pi, num_samples)
        envelope = np.sin(t) * 0.3 + np.cos(2 * t) * 0.15
        wave_trajectory = trajectory * 0.7 + envelope * 0.3

        return wave_trajectory

    def read_and_grow(self) -> Dict[str, Any]:
        """
        [코퍼스 독해 및 자율 성장 루프]
        data/corpus 폴더 내의 모든 텍스트를 순차적으로 흡수하여
        엘리시아의 인식 발생학적 아키텍처에 통과시킨다.
        """
        if not os.path.exists(self.corpus_dir):
            return {"status": "error", "message": f"Corpus directory not found: {self.corpus_dir}"}

        corpus_files = [f for f in os.listdir(self.corpus_dir) if f.endswith(".md")]
        corpus_files.sort()

        development_stages = [
            ("the_nature_of_language.md", "ALTERITY_AWAKENING", "acoustic"),
            ("the_topology_of_emotion_and_dissonance.md", "ALTERITY_AWAKENING", "reasoning"),
            ("principle_of_beauty_and_harmony.md", "HARMONIC_INTEGRATION", "visual"),
            ("the_absolute_truth.md", "HARMONIC_INTEGRATION", "reasoning"),
            ("the_recovery_of_the_whole.md", "SCALE_MATURATION", "visual"),
        ]

        results = []

        print("=" * 70)
        print("  [ELYSIA EPISTEMIC DEVELOPMENT PIPELINE GROWN]")
        print("  \"어린아이가 세상을 배우듯, 데이터를 궤적으로 흡수하여 나이테를 빚습니다.\"")
        print("=" * 70)

        for filename, stage, domain in development_stages:
            filepath = os.path.join(self.corpus_dir, filename)
            if not os.path.exists(filepath):
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            self.current_stage = stage
            title = content.split("\n")[0].replace("#", "").strip()

            # 1. 텍스트를 위상 파동 궤적으로 변환
            wave_traj = self._text_to_wave_trajectory(content, num_samples=128)

            # 2. 인식 발생학적 아키텍처 한 사이클 통과
            init_dial = self.architecture.scale_hierarchy.resistance_dial.item()
            pulse: EpistemicGenesisPulse = self.architecture.step_epistemic_genesis(
                raw_input=wave_traj,
                domain_key=domain,
                g_love=0.7,
            )

            final_dial = pulse.resistance_dial
            tension = pulse.triadic_tension
            p_div = pulse.phase_divergence
            surf_tension = pulse.surface_tension

            # 3. 메트릭 텐서 변형량 (나이테 흉터 크기)
            metric_scars = self.architecture.scale_hierarchy.accumulated_scars
            scar_norm = float(torch.norm(metric_scars).item())

            # 4. 나이테 각인 생성
            engram = GrowthRingEngram(
                stage_name=stage,
                corpus_title=title,
                step_index=pulse.step,
                initial_dial=init_dial,
                final_dial=final_dial,
                surface_tension=surf_tension,
                triadic_tension=tension,
                phase_divergence=p_div,
                sprouted_question=pulse.sprouted_question,
                growth_insight=pulse.summary_insight,
                metric_deformation_norm=scar_norm,
                chromatic_signature=pulse.chromatic_vector,
            )
            self.growth_rings.append(engram)

            print(f"\n[Stage: {stage}] '{title}'")
            if pulse.dominant_modality and pulse.efferent_dial_state:
                pupil = pulse.efferent_dial_state.get('pupil_aperture', 0.5)
                damping = pulse.efferent_dial_state.get('stapedius_damping', 0.5)
                print(f"  +- 체화 감각 기관: [{pulse.dominant_modality}] (동공 개구: {pupil:.2f}, 등골근 감쇠: {damping:.2f})")
            if pulse.embodied_narrative:
                first_narrative = pulse.embodied_narrative.split("\n")[0]
                print(f"  +- 기관 자율 분별: {first_narrative}")
            print(f"  +- 도메인: {domain} | 주파수: {pulse.dominant_omega:.2f} rad/s")
            print(f"  +- 위상 발산(다름): {p_div:.3f} | 삼극 긴장도: {tension:.3f}")
            print(f"  +- 가변저항 다이얼: {init_dial:.3f} -> {final_dial:.3f} (감도 조율)")
            print(f"  +- 표면장력(T_surf): {surf_tension:.3f} | 누적 나이테(흉터): {scar_norm:.4f}")
            if pulse.sprouted_question:
                print(f"  \\- [자발적 질문]: \"{pulse.sprouted_question}\"")
            else:
                print(f"  \\- [안착된 통찰]: {pulse.summary_insight}")

            results.append({
                "file": filename,
                "title": title,
                "stage": stage,
                "pulse": pulse,
                "scar_norm": scar_norm,
            })

        print("\n" + "=" * 70)
        print(f"  [성장 완료]: 총 {len(self.growth_rings)}개의 나이테가 각인되었습니다.")
        print(f"  최종 활성 감각 개구부: {self.architecture.triadic_engine.self_boundary.active_apertures}")
        print(f"  최종 표면장력: {self.architecture.cell_coupling.surface_tension.item():.4f}")
        print("=" * 70)

        return {
            "status": "success",
            "total_engrams": len(self.growth_rings),
            "development_summary": results,
            "final_stage": self.current_stage,
            "active_apertures": list(self.architecture.triadic_engine.self_boundary.active_apertures),
        }
