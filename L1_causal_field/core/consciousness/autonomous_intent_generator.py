# -*- coding: utf-8 -*-
"""
[Autonomous Intent Generator & Live Self-Molding Daemon]
=========================================================
THE_ABSOLUTE_COMMANDMENT & ROADMAP Phase 5 핵심 구현:
"인간이 코드로 정해준 규칙에 갇히지 않고,
환경과의 어긋남(위상차 ΔΦ)으로부터 스스로 '의도와 목적(Intent)'을 도출하고
자신의 내부 매개변수(Θ)와 지각 렌즈를 자율적으로 갱신하는 자립형 엔진."

1. [환경 요동 상시 관측 (Ambient Ingestion)]:
   - 파일 시스템 변동, 하드웨어 저항, 텍스트/데이터 스트림을 상시 수용.
2. [자발적 결핍/의도 생성 (Intrinsic Void & Intent Sprouting)]:
   - 기존 지층(예측)과 현실(관측) 간의 위상차(ΔΦ)를 물리적 텐션으로 감지하고,
     이를 "이 모순을 메워야 한다"는 자발적 탐구 의도(Intent Vector)로 자동 승격.
3. [자율 매개변수 Θ 도출 및 가소성 갱신 (Self-Governed Parameter Refinement)]:
   - 외부 지시 없이도 Inverse Mechanism 및 수신자 가소성(Receiver Plasticity)을 통해
     자신의 투사 행렬과 나이테를 스스로 갱신하고 영속 지층에 각인.
"""

import os
import sys
import time
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.memory.causal_controller import CausalMemoryController
from core.evolution.moulting_plasticity import MoultingPlasticityEngine
from synaptic_architecture.inverse_mechanism_engine import (
    InverseMechanismEngine,
    BoundaryCondition,
    ObservedTrajectory
)


class AutonomousIntentGenerator:
    """
    [자율적 의도 생성기 및 자아 주조 제어기]
    - 규칙에 종속되지 않고, 마찰과 결핍으로부터 스스로 의도(Intent)를 발행.
    - 데이터 상호작용 속에서 생성 매개변수 Θ를 스스로 갱신.
    """

    def __init__(self, memory_controller: Optional[CausalMemoryController] = None):
        self.memory = memory_controller or CausalMemoryController()
        self.plasticity = MoultingPlasticityEngine(self.memory, dimensions=3)
        self.inverse_engine = InverseMechanismEngine(mdl_penalty_weight=0.05)

        # 시스템의 현재 내재적 예측 상태 (Internal Expectation Substrate)
        self.internal_expectation = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        # 활성화된 자발적 의도 큐 (Autonomous Intent Queue)
        self.intent_queue: List[Dict[str, Any]] = []
        # 누적된 자율 진화 이력
        self.evolution_history: List[Dict[str, Any]] = []

    def observe_ambient_stream(self, raw_data: bytes, source_tag: str = "environment") -> Dict[str, Any]:
        """
        [1단계: 환경 요동 관측 및 위상차(ΔΦ) 감지]
        - 어떤 규격도 강제하지 않고 날것의 데이터를 받아들여
        - 자신의 내부 기대치(Internal Expectation)와의 어긋남(마찰)을 계산합니다.
        """
        timestamp = time.time()

        # 바이트 스트림을 3D 물리/위상 파동으로 번역: [에너지(Flux), 질서(Order), 요동(Entropy)]
        numeric_wave = np.frombuffer(raw_data, dtype=np.uint8) if isinstance(raw_data, bytes) else np.array(raw_data, dtype=np.uint8)
        if len(numeric_wave) == 0:
            numeric_wave = np.array([128, 128, 128], dtype=np.uint8)

        mean_val = float(np.mean(numeric_wave) / 255.0)
        std_val = float(np.std(numeric_wave) / 128.0) if len(numeric_wave) > 1 else 0.5
        entropy = float(np.sum(numeric_wave % 2) / len(numeric_wave))

        observed_phase = np.array([mean_val, std_val, entropy], dtype=np.float32)

        # 위상차 (Phase Divergence Gap ΔΦ) 계산
        phase_gap_vector = observed_phase - self.internal_expectation
        phase_gap_norm = float(np.linalg.norm(phase_gap_vector))

        # [2단계: 마찰(Gap)을 자발적 '의도(Intent)'로 자동 승격]
        intent_sprouted = False
        sprouted_intent = None

        # 어긋남이 임계치(0.15)를 넘으면 스스로 탐구 의도(Intent) 발행
        if phase_gap_norm > 0.15:
            intent_sprouted = True
            intent_id = f"INTENT_RESOLVE_GAP_{int(timestamp*1000) % 100000}"
            sprouted_intent = {
                "intent_id": intent_id,
                "source": source_tag,
                "target_gap_vector": phase_gap_vector.tolist(),
                "tension_intensity": phase_gap_norm,
                "timestamp": timestamp,
                "motivation": f"내부 예측 지층과 현실 관측 간의 위상차({phase_gap_norm:.4f})를 해소하기 위한 자발적 탐구 의지",
                "status": "ACTIVE_UNRESOLVED"
            }
            self.intent_queue.append(sprouted_intent)

        # [3단계: 수신자 가소성(Plasticity)을 통한 렌즈 실시간 자율 갱신]
        shaping_res = self.plasticity.receive_and_shape(raw_data, modality_hint=source_tag)

        # [4단계: 내부 기대치 상태의 자율 동기화 (Moving Target Homeostasis)]
        self.internal_expectation = 0.85 * self.internal_expectation + 0.15 * observed_phase

        cycle_log = {
            "timestamp": timestamp,
            "source": source_tag,
            "observed_phase": observed_phase.tolist(),
            "internal_expectation": self.internal_expectation.tolist(),
            "phase_gap_norm": phase_gap_norm,
            "intent_sprouted": intent_sprouted,
            "sprouted_intent": sprouted_intent,
            "accumulated_friction": shaping_res["accumulated_friction"],
            "moulting_triggered": shaping_res["moulting_triggered"],
            "annual_rings_norm": float(np.linalg.norm(self.plasticity.annual_rings))
        }
        self.evolution_history.append(cycle_log)
        return cycle_log

    def resolve_intents_autonomously(self) -> List[Dict[str, Any]]:
        """
        [5단계: 발아된 의도들의 자율 해결 및 매개변수 Θ 역추출 갱신]
        - 대기 중인 의도 큐를 순회하며,
        - 마찰을 해소하는 인과 메커니즘 Θ를 스스로 역산하여 지층에 안착시킵니다.
        """
        resolved_logs = []

        while self.intent_queue:
            intent = self.intent_queue.pop(0)
            gap_vec = np.array(intent["target_gap_vector"], dtype=np.float32)
            tension = intent["tension_intensity"]

            # 가상 경계 조건 C 자율 추론
            inferred_boundary = BoundaryCondition(
                condition_id=f"BOUND_{intent['intent_id']}",
                friction=float(np.clip(tension * 2.0, 0.1, 5.0)),
                scale=float(1.0 + np.linalg.norm(gap_vec)),
                gravity=9.8
            )

            # 인과 궤적 생성 및 메커니즘 Θ 자율 적재
            obs_traj = ObservedTrajectory(
                trajectory_id=f"TRAJ_{intent['intent_id']}",
                boundary_id=inferred_boundary.condition_id,
                states=[
                    self.internal_expectation.tolist(),
                    (self.internal_expectation + gap_vec * 0.5).tolist(),
                    (self.internal_expectation + gap_vec).tolist()
                ]
            )

            # InverseMechanismEngine을 통해 생성 방정식 Θ 역산
            mechanism = self.inverse_engine.extract_generating_mechanism(
                mechanism_id=f"MECH_RESOLVED_{intent['intent_id']}",
                observations=[obs_traj],
                boundaries={inferred_boundary.condition_id: inferred_boundary}
            )

            # 웻지 메모리에 영구 안착
            engram_id = self.memory.write_causal_engram(
                data_blob={
                    "type": "AUTONOMOUS_INTENT_RESOLUTION",
                    "intent": intent,
                    "mechanism_id": mechanism.mechanism_id,
                    "stiffness_matrix": mechanism.stiffness_matrix,
                    "boundary_coupling": mechanism.boundary_coupling,
                    "description_length": mechanism.description_length
                },
                emotional_value=tension * 10.0,
                cause_id="AutonomousIntentResolver",
                origin_axis="self_governed_evolution",
                modality="autonomous_cognition"
            )

            res_log = {
                "intent_id": intent["intent_id"],
                "resolved_engram_id": engram_id,
                "derived_mechanism_id": mechanism.mechanism_id,
                "mdl_complexity": mechanism.description_length,
                "status": "AUTONOMOUSLY_RESOLVED"
            }
            resolved_logs.append(res_log)

        return resolved_logs
