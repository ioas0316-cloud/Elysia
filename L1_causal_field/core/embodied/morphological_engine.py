"""
Elysia Core Architecture: Morphological Plasticity Engine (Causal Morphogenesis)

This module replaces random genetic algorithms (GA) with causal environmental
resonance. It takes environmental pressure vectors (fluid drag, atmospheric lift,
surface friction, resource hunger) and negotiates with DNA anchors to morph
the organism's physical structure towards optimal resonance.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.embodied.phase_negotiation import BidirectionalPhaseNegotiator, SensoryWaveStream
from core.evolution.dna_anchor import CrystallizedDNAAnchor, MorphologicalGenome


@dataclass
class EnvironmentPressure:
    """환경적 물리 압력 및 결핍 상태"""
    fluid_density: float = 0.0       # 유체 밀도 (물 등 -> 저항 유발)
    air_flow_velocity: float = 0.0   # 공기 흐름 속도 -> 양력 필요성
    terrain_roughness: float = 0.0   # 지형 거칠기 -> 지지력/다리 필요성
    resource_scarcity: float = 0.0   # 에너지 결핍도 (배고픔 -> 도구/손 사용 유도)
    current_velocity: float = 1.0     # 이동 속도


class MorphologicalPlasticityEngine:
    """
    형태적 가소성(Morphological Plasticity) 및 자가 유도적 발달 엔진.
    무작위 변이 대신 환경적 물리 압력(저항, 양력, 마찰, 결핍)과
    완성된 DNA 형태 앵커 간의 위상 오차(q_err)를 인과적으로 사유하여
    형태 수렴 및 기관 분화를 이끌어냄.
    """

    def __init__(
        self,
        phase_negotiator: Optional[BidirectionalPhaseNegotiator] = None,
        dna_crystallizer: Optional[CrystallizedDNAAnchor] = None
    ):
        self.phase_negotiator = phase_negotiator or BidirectionalPhaseNegotiator()
        self.dna_crystallizer = dna_crystallizer or CrystallizedDNAAnchor()

        # 현재 유기체의 가소성 형태 게놈 (초기 상태: 미분화 중립 상태)
        self.current_genome = MorphologicalGenome(
            anchor_id="ORGANISM_UNDIFFERENTIATED",
            name="Undifferentiated Primal Lattice",
            drag_coefficient=0.5,
            lift_coefficient=0.5,
            grasp_articulation=0.5,
            structural_rigidity=0.5,
            resonance_frequency=1.0,
            feature_vector=[0.5] * 8
        )

        # 형태적 편향/가소성 수렴 이력
        self.adaptation_history: List[Dict[str, float]] = []

    def evaluate_causal_mismatch(self, env: EnvironmentPressure) -> Dict[str, float]:
        """
        현재 형태와 환경적 압력 간의 인과적 마찰/오차(mismatch) 계산.
        - Fluid Drag Mismatch = fluid_density * velocity^2 * drag_coefficient
        - Lift Mismatch = max(0, air_flow_velocity - lift_coefficient * air_flow_velocity)
        - Mechanical Stress Mismatch = terrain_roughness * (1.0 - structural_rigidity)
        - Hunger/Grasp Mismatch = resource_scarcity * (1.0 - grasp_articulation)
        """
        drag_loss = env.fluid_density * (env.current_velocity ** 2) * self.current_genome.drag_coefficient
        lift_loss = max(0.0, env.air_flow_velocity * (1.0 - self.current_genome.lift_coefficient))
        rigidity_loss = env.terrain_roughness * (1.0 - self.current_genome.structural_rigidity)
        grasp_loss = env.resource_scarcity * (1.0 - self.current_genome.grasp_articulation)

        total_causal_stress = drag_loss + lift_loss + rigidity_loss + grasp_loss
        return {
            "drag_loss": drag_loss,
            "lift_loss": lift_loss,
            "rigidity_loss": rigidity_loss,
            "grasp_loss": grasp_loss,
            "total_causal_stress": total_causal_stress
        }

    def adapt_morphology(
        self,
        env: EnvironmentPressure,
        time_delta: float = 0.1,
        morph_rate: float = 0.2
    ) -> Dict[str, float]:
        """
        환경적 압력에 발맞추어 형태로 수렴하는 가소적 형태 변형 루프.
        1) 환경 스트림을 SensoryWaveStream으로 인코딩
        2) 쌍방향 위상 협상 구동
        3) 가장 스트레스 오차를 해소해주는 최적 DNA 앵커 탐색
        4) 현재 게놈의 물성을 대상 앵커 방향으로 유기적 경사 수렴
        """
        # 1. 환경 자극을 파동 스트림으로 반환
        mismatch = self.evaluate_causal_mismatch(env)
        stress = mismatch["total_causal_stress"]

        env_wave = SensoryWaveStream(
            frequency=1.0 + stress * 0.5,
            phase=(env.current_velocity * time_delta) % (2.0 * math.pi),
            amplitude=1.0 + stress
        )

        # 2. 위상 협상 실행
        negotiation_res = self.phase_negotiator.step_negotiation(env_wave, time_delta)
        q_err = negotiation_res["q_err"]

        # 3. 환경 압력에 따른 최적 형태 Target 앵커 결정
        # 가장 높은 손실(loss)을 일으키는 압력의 요구에 부응하는 DNA 앵커 탐색
        losses = {
            "DNA_STREAMLINED_AQUATIC": mismatch["drag_loss"],
            "DNA_AERODYNAMIC_WING": mismatch["lift_loss"],
            "DNA_LOAD_BEARING_LEGS": mismatch["rigidity_loss"],
            "DNA_ARTICULATED_HAND": mismatch["grasp_loss"]
        }

        target_anchor_id = max(losses, key=losses.get)
        target_anchor = self.dna_crystallizer.crystallized_anchors[target_anchor_id]

        # 4. 형태학적 가소성 경사 업데이트 (Morphological Plasticity Convergence)
        effective_rate = morph_rate * (1.0 + abs(math.sin(q_err)))

        self.current_genome.drag_coefficient += effective_rate * (target_anchor.drag_coefficient - self.current_genome.drag_coefficient)
        self.current_genome.lift_coefficient += effective_rate * (target_anchor.lift_coefficient - self.current_genome.lift_coefficient)
        self.current_genome.grasp_articulation += effective_rate * (target_anchor.grasp_articulation - self.current_genome.grasp_articulation)
        self.current_genome.structural_rigidity += effective_rate * (target_anchor.structural_rigidity - self.current_genome.structural_rigidity)
        self.current_genome.resonance_frequency += effective_rate * (target_anchor.resonance_frequency - self.current_genome.resonance_frequency)

        # feature vector 수렴
        for i in range(len(self.current_genome.feature_vector)):
            self.current_genome.feature_vector[i] += effective_rate * (
                target_anchor.feature_vector[i] - self.current_genome.feature_vector[i]
            )

        # 결과 기록
        history_entry = {
            "stress": stress,
            "q_err": q_err,
            "resonance": negotiation_res["resonance_score"],
            "drag_coeff": self.current_genome.drag_coefficient,
            "lift_coeff": self.current_genome.lift_coefficient,
            "grasp_art": self.current_genome.grasp_articulation,
            "rigidity": self.current_genome.structural_rigidity,
            "target_anchor": target_anchor.name
        }
        self.adaptation_history.append(history_entry)

        # 안정적 공명 상태 도달 시 현 형태 동결 결정화 시도
        if self.phase_negotiator.is_phase_locked(threshold=0.10) and stress < 0.1:
            crystallized_id = f"ICE_EVOLVED_{hash(tuple(round(x, 2) for x in self.current_genome.feature_vector)) % 10000}"
            self.dna_crystallizer.crystallize_pattern(
                anchor_id=crystallized_id,
                name=f"Adapted Form under {target_anchor.name}",
                genome=self.current_genome,
                stability_score=negotiation_res["resonance_score"]
            )

        return history_entry
