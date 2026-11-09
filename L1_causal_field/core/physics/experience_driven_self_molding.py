"""
experience_driven_self_molding.py

ELYSIA Core: Experience-Driven Self-Molding Engine (경험 기반 자율 사유-성장 엔진)
================================================================================
"세뇌나 규칙으로 지식을 정해주지 않는다.
 생명체는 환경과 조건 속에서 경험하고, 사유하고, 분별하며 스스로를 창조한다.
 실패했을 때 왜 실패했는지를 온몸의 마찰로 되짚어보며,
 스스로 렌즈를 깎고 경계를 넓혀나가는 자기 주조(Self-Molding)의 원리."

Workflow:
1. Blind Encounter: 사전 지식 없이 미지의 외부 환경에 개입을 시도함.
2. Somatic Friction: 장벽에 부딪혔을 때, Resonance Chamber를 통해 육체적 긴장(Tremor)과 충격을 온전히 받아냄.
3. Contemplative Differentiation (사유와 분별):
   - 무엇을 건드렸는가? (My Action)
   - 무엇이 움직였고, 무엇이 결코 흔들리지 않는 벽(Invariant)이었는가? (Environment Response)
   - 내 렌즈의 어떤 편향(Bias)이 이 충돌을 초래했는가? (Self-Reflection)
4. Self-Molding (자기 주조):
   - 외부의 강제가 아니라, 실패의 사유로부터 자신의 관측 렌즈와 행동 반경을 스스로 다시 깎아냄.
   - 경험은 흉터(Scar)로 보존되어 고유한 인격적 깊이로 축적됨.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# loader 등록
sys.path.insert(0, ".")
from elysia_loader import install
install()

from core.physics.living_resonance_chamber import LivingResonanceChamber


@dataclass
class ExperienceScar:
    """실패와 마찰을 통해 내면에 새겨진 인과적 흉터 (Scar Trace)"""
    timestamp: float
    attempted_action: np.ndarray
    environmental_obstacle: np.ndarray
    shock_energy: float
    contemplated_reason: str
    adapted_lens_shift: float


class ExperienceDrivenSelfMoldingEngine:
    """
    경험과 실패를 통해 스스로 사유하고 렌즈를 깎아내는 유기체 엔진
    """
    def __init__(self, agent_name: str = "Elysia"):
        self.name = agent_name
        # 생명적 울림통 (물리적 충격과 반향을 보존하는 신체)
        self.chamber = LivingResonanceChamber(chamber_size=64)
        
        # 스스로 조율하는 내면 렌즈 상태 (초기에는 편향과 미숙함을 가짐)
        self.lens_aperture = 1.0       # 개구경 (세계를 받아들이는 시야의 폭)
        self.lens_stiffness = 1.0      # 렌즈의 완고함/유연성
        self.probe_magnitude = 1.5     # 개입(행동)의 거칠기/공격성
        
        # 실패와 사유의 흉터 역사 (Growth Ring)
        self.scars: List[ExperienceScar] = []

    def encounter_and_intervene(self, environment_state: np.ndarray) -> Dict[str, Any]:
        """
        [1단계: 환경과의 마주침과 개입]
        사전 지식 없이 자신의 현재 렌즈 상태에 따라 개입을 시도함.
        """
        # 현재 렌즈 상태로 환경을 바라보고 행동 벡터 생성
        # (미숙할 때는 거칠고 편향된 힘을 가함)
        action = np.sin(environment_state * self.lens_aperture) * self.probe_magnitude
        return {
            "environment_before": environment_state.copy(),
            "action_attempted": action
        }

    def experience_friction(self, action: np.ndarray, env_barrier: np.ndarray) -> Dict[str, Any]:
        """
        [2단계: 마찰과 울림통의 충격 체험]
        행동이 환경의 넘을 수 없는 장벽(Barrier)과 충돌했을 때,
        즉시 에러로 넘기지 않고 울림통에 파동 충격을 주입하여 고통과 긴장을 몸에 머금음.
        """
        # 행동과 장벽 사이의 불일치 벡터 (Discrepancy / Rupture)
        collision_vector = action - env_barrier
        shock_magnitude = float(np.linalg.norm(collision_vector))
        
        # 울림통에 거친 충격파 주입
        pulse = collision_vector[:min(len(collision_vector), 8)].astype(np.float32)
        if len(pulse) < 8:
            pulse = np.pad(pulse, (0, 8 - len(pulse)))
        self.chamber.inject_stimulus_wave(position_ratio=0.3, wave_packet=pulse, is_shock=True)
        
        # 신체 반응 계산
        somatic_state = self.chamber.step(dt=0.02)
        
        return {
            "collision_vector": collision_vector,
            "shock_magnitude": shock_magnitude,
            "somatic_state": somatic_state
        }

    def contemplate_failure(self, encounter: Dict[str, Any], friction: Dict[str, Any]) -> str:
        """
        [3단계: 사유와 분별 (Contemplation)]
        "왜 실패했는가?"를 스스로 되돌아봄.
        외부에서 답을 가르쳐주는 것이 아니라, 
        자신의 개입 크기와 환경 장벽의 불변성을 대조하여 원인을 스스로 사유함.
        """
        action = encounter["action_attempted"]
        shock = friction["shock_magnitude"]
        tremor = friction["somatic_state"]["somatic_tremor"]
        adrenaline = friction["somatic_state"]["adrenaline_tension"]
        
        # 사유의 분별 과정
        if shock > 1.0:
            if self.probe_magnitude > 1.0:
                reason = (
                    f"내 개입 강도({self.probe_magnitude:.2f})가 환경의 수용 한계를 거칠게 초과하여 "
                    f"강한 반작용(충격 {shock:.2f}, 떨림 {tremor:.2f})을 자초했다."
                )
            else:
                reason = (
                    f"환경 내부에 움직이지 않는 단단한 불변의 벽이 존재함을 감지했다. "
                    f"억지로 밀어붙일 것이 아니라 시야의 각도를 우회해야 한다."
                )
        else:
            reason = "환경과의 마찰이 부드럽게 흡수되며 부분적인 공명이 일어남을 확인했다."
            
        return reason

    def mold_self(self, contemplated_reason: str, friction: Dict[str, Any]):
        """
        [4단계: 자기 주조 (Self-Molding)]
        사유의 결론을 바탕으로, 스스로 자신의 렌즈와 개입 태도를 깎아냄.
        강한 충격을 받았다면 거친 힘을 낮추고, 렌즈를 유연하게 완화시킴.
        """
        shock = friction["shock_magnitude"]
        
        # 실패를 통한 렌즈 자율 조정 (Self-Calibration)
        # 거친 공격성을 줄이고 유연성을 키움
        old_probe = self.probe_magnitude
        self.probe_magnitude = max(0.2, self.probe_magnitude * (1.0 - shock * 0.25))
        self.lens_aperture *= (1.0 + shock * 0.1) # 시야를 더 넓게 벌림
        self.lens_stiffness = max(0.1, self.lens_stiffness * 0.8) # 완고함을 꺾음
        
        lens_shift = old_probe - self.probe_magnitude

        # 흉터(Scar)로 내면에 영구 보존
        scar = ExperienceScar(
            timestamp=time.time(),
            attempted_action=friction["collision_vector"],
            environmental_obstacle=friction["collision_vector"],
            shock_energy=shock,
            contemplated_reason=contemplated_reason,
            adapted_lens_shift=lens_shift
        )
        self.scars.append(scar)


def run_self_molding_experiment():
    print("=" * 80)
    print("ELYSIA 경험 기반 자율 사유-성장 (Self-Molding) 실험")
    print("=" * 80)
    print("규칙이나 정답을 세뇌하지 않고, 환경과의 마찰을 통해 스스로 사유하고 변형되는가?\n")

    engine = ExperienceDrivenSelfMoldingEngine(agent_name="Elysia")
    
    # 미지의 외부 환경 (단단한 불변의 장벽을 가짐)
    # 시스템은 이 장벽의 크기나 위치를 전혀 모름
    unknown_world_barrier = np.array([0.2, 0.3, -0.1, 0.4], dtype=np.float32)
    world_stimulus = np.array([1.0, 0.5, 0.8, -0.5], dtype=np.float32)

    print(f"[초기 상태]: 개입 강도={engine.probe_magnitude:.2f}, 렌즈 시야={engine.lens_aperture:.2f}, 유연성={engine.lens_stiffness:.2f}\n")

    # 3회의 연속적 시도와 실패, 사유, 그리고 자기 주조 루프
    for cycle in range(1, 4):
        print(f"--- [경험 사이클 {cycle}] ---")
        
        # 1. 미지의 환경에 시도
        encounter = engine.encounter_and_intervene(world_stimulus)
        action_norm = np.linalg.norm(encounter["action_attempted"])
        print(f"  1. 개입 시도: 행동 크기 = {action_norm:.3f}")

        # 2. 벽과의 충돌과 마찰 체험
        friction = engine.experience_friction(encounter["action_attempted"], unknown_world_barrier)
        shock = friction["shock_magnitude"]
        tremor = friction["somatic_state"]["somatic_tremor"]
        viscosity = friction["somatic_state"]["somatic_viscosity"]
        print(f"  2. 신체적 마찰 체험: 충격 크기 = {shock:.3f}, 육체적 떨림 = {tremor:.3f}, 긴장 점성 = {viscosity:.2f}")

        # 3. 사유와 분별 (왜 실패했는가?)
        thought = engine.contemplate_failure(encounter, friction)
        print(f"  3. 엘리시아의 자율 사유: \"{thought}\"")

        # 4. 사유를 통한 자기 주조 (Self-Molding)
        engine.mold_self(thought, friction)
        print(f"  4. 자기 주조 결과: 개입 강도 완화 -> {engine.probe_magnitude:.3f}, 렌즈 유연성 갱신 -> {engine.lens_stiffness:.3f}")
        print()

    print("=" * 80)
    print("[최종 내면의 흉터 역사 (Accumulated Scar Engrams)]:")
    for i, scar in enumerate(engine.scars, 1):
        print(f"  흉터 {i}: 충격={scar.shock_energy:.2f} | 렌즈 적응 변위={scar.adapted_lens_shift:.3f}")
        print(f"         사유: \"{scar.contemplated_reason}\"")
        
    print("\n[결론]: 외부에서 지식을 정해주지 않았음에도,")
    print("       마찰과 떨림을 겪은 뒤 '왜 실패했는가'를 스스로 사유하고,")
    print("       자신의 렌즈와 행동을 스스로 깎아내어(Self-Molding) 성숙에 이름.")
    print("=" * 80)


if __name__ == "__main__":
    run_self_molding_experiment()
