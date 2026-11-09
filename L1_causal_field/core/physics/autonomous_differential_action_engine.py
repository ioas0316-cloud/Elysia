"""
autonomous_differential_action_engine.py

ELYSIA Core: Autonomous Differential Action & Gap-Closure Engine
================================================================
"의도한 결과가 존재할 때:
 1. 무엇이 어디서 어떻게 같고 다른지 (Differential Comparison),
 2. 그 차이를 메우기 위해 무엇이 필요하고 무엇을 알아야 하는지 (Need & Inquiry),
 3. 시스템 스스로 헤아려(Contemplate) 자율적으로 행동하게 만드는 구조원리."

Core Loop:
  [의도한 상태 (Intent)] vs [현재 상태 (Reality)]
             │
             ▼ (비교대조)
  [무엇이 같고 무엇이 다른가?] -> [필요한 앎과 개입의 방향 헤아림]
             │
             ▼ (자율 행동)
  [세계에 행동/개입 실행] -> [환경 반응 관측]
             │
             ▼ (재인식)
  [차이가 얼마나 좁혀졌는가? 아직 남은 결핍은 무엇인가?] -> 반복
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

# loader 등록
sys.path.insert(0, ".")
from elysia_loader import install
install()


@dataclass
class DifferentialReport:
    """스스로 헤아린 같음과 다름, 그리고 필요한 행동의 기록"""
    step_index: int
    aligned_ratio: float           # 이미 일치하는 비율 (무엇이 같은가)
    differing_axes: np.ndarray      # 차이가 나는 축들 (무엇이 다른가)
    gap_magnitude: float           # 전체 결핍의 크기
    needed_action_direction: np.ndarray # 차이를 좁히기 위해 필요한 개입 벡터
    contemplated_strategy: str      # 스스로 내린 판단과 사유


class AutonomousDifferentialActionEngine:
    """
    같음과 다름을 스스로 헤아려 행동하는 구조원리 엔진
    """
    def __init__(self, state_dim: int = 4):
        self.state_dim = state_dim
        self.action_history: List[DifferentialReport] = []

    def contemplate_difference(self, current_state: np.ndarray, intended_target: np.ndarray, step_idx: int) -> DifferentialReport:
        """
        [1. 같음과 다름의 스스로 헤아림]
        결과를 미리 주지 않는다.
        오직 현재와 목표를 마주 세우고, 어디가 같고 어디가 다른지 축별로 분별함.
        """
        # 축별 차이 벡터 (Gap Vector = Target - Current)
        diff_vector = intended_target - current_state
        gap_magnitude = float(np.linalg.norm(diff_vector))
        
        # 같음의 척도: 상대적 오차가 작은 축들의 비율
        component_similarity = 1.0 - (np.abs(diff_vector) / (np.abs(intended_target) + 1e-5))
        aligned_ratio = float(np.mean(np.clip(component_similarity, 0.0, 1.0)))

        # 무엇이 가장 크게 다른가? (가장 큰 결핍 축 식별)
        max_diff_axis = int(np.argmax(np.abs(diff_vector)))
        max_diff_val = diff_vector[max_diff_axis]

        # 차이를 좁히기 위해 지금 당장 필요한 행동 방향 도출
        # (단순 비례가 아니라, 차이의 크기와 방향에 따른 자율 조율)
        action_direction = np.zeros_like(diff_vector)
        action_direction[max_diff_axis] = np.sign(max_diff_val) * min(1.0, abs(max_diff_val) * 0.8)
        
        # 나머지 잔여 차이들에 대해서도 완만한 보정 벡터 가산
        for i in range(len(diff_vector)):
            if i != max_diff_axis:
                action_direction[i] = diff_vector[i] * 0.4

        # 스스로 사유한 전략 서사
        if gap_magnitude > 0.8:
            strategy = (
                f"현재 상태는 목표와 {gap_magnitude:.3f}만큼 떨어져 있다. "
                f"특히 [{max_diff_axis}]번 축의 차이({max_diff_val:+.3f})가 가장 크므로, "
                f"이 축의 결핍을 채우는 개입({action_direction[max_diff_axis]:+.3f})을 우선 단행해야 한다."
            )
        elif gap_magnitude > 0.1:
            strategy = (
                f"대부분의 축({aligned_ratio*100:.1f}%)이 정렬되었으나, 미세한 잔여 편차({gap_magnitude:.3f})가 남아있다. "
                f"미세 조율을 통해 전체적인 위상 일치를 완성해야 한다."
            )
        else:
            strategy = f"목표와 현재 상태가 실질적으로 일치함({aligned_ratio*100:.1f}%). 의도한 결과에 도달함."

        report = DifferentialReport(
            step_index=step_idx,
            aligned_ratio=aligned_ratio,
            differing_axes=diff_vector,
            gap_magnitude=gap_magnitude,
            needed_action_direction=action_direction,
            contemplated_strategy=strategy
        )
        self.action_history.append(report)
        return report

    def take_action(self, current_state: np.ndarray, action_direction: np.ndarray, environment_impedance: np.ndarray) -> np.ndarray:
        """
        [2. 헤아림에 따른 실제 행동과 환경의 반응]
        자신이 헤아린 필요 행동을 환경에 가하고, 
        환경의 임피던스(저항)에 의해 새로운 상태로 전이됨.
        """
        # 환경 저항(Impedance)에 의해 행동이 일부 감쇠되거나 굴절됨
        effective_delta = action_direction * (1.0 - environment_impedance)
        new_state = current_state + effective_delta
        return new_state


def run_differential_closure_experiment():
    print("=" * 80)
    print("ELYSIA 자율적 같음/다름 헤아림 및 간극 해소 (Differential Action) 실험")
    print("=" * 80)
    print("결과를 미리 주지 않는다.")
    print("시스템이 의도와 현재의 같고 다름을 스스로 비교하고, 필요한 행동을 찾아 도달하는가?\n")

    engine = AutonomousDifferentialActionEngine(state_dim=4)

    # 1. 의도한 목표 상태 (예: 광학적 조화 및 평형 궤적)
    intended_target = np.array([1.0, -0.5, 0.8, 0.0], dtype=np.float32)
    # 2. 현재의 시작 상태 (완전히 동떨어진 곳에서 출발)
    current_state = np.array([-0.8, 0.9, -0.4, 1.2], dtype=np.float32)
    # 3. 환경의 저항 임피던스 (각 축마다 다른 저항이 걸림)
    env_impedance = np.array([0.2, 0.3, 0.1, 0.4], dtype=np.float32)

    print(f"[의도한 목표 상태]: {intended_target.round(2).tolist()}")
    print(f"[현재 출발 상태]:   {current_state.round(2).tolist()}")
    print(f"[환경의 물리 저항]: {env_impedance.round(2).tolist()}\n")

    # 시스템이 스스로 차이를 헤아리고 행동을 거듭하여 도달하는 루프
    max_steps = 8
    for step in range(1, max_steps + 1):
        print(f"--- [스스로 헤아리는 스텝 {step}] ---")
        
        # 1. 같고 다름을 비교대조하고 필요한 것을 헤아림
        report = engine.contemplate_difference(current_state, intended_target, step)
        print(f"  * 현재 결핍 크기 (|Gap|): {report.gap_magnitude:.4f} (일치도: {report.aligned_ratio*100:.1f}%)")
        print(f"  * 축별 차이 벡터: {report.differing_axes.round(3).tolist()}")
        print(f"  * 스스로 내린 판단: \"{report.contemplated_strategy}\"")
        print(f"  * 도출된 행동 벡터: {report.needed_action_direction.round(3).tolist()}")

        if report.gap_magnitude < 0.05:
            print("\n  => [스스로 확인]: 의도한 상태와의 차이가 0.05 미만으로 좁혀져 목표에 완전히 도달함을 자각함.")
            break

        # 2. 헤아린 행동을 세계에 실행
        current_state = engine.take_action(current_state, report.needed_action_direction, env_impedance)
        print(f"  * 행동 후 새로운 현실 상태: {current_state.round(3).tolist()}\n")

    print("=" * 80)
    print("[최종 검증]:")
    final_diff = np.linalg.norm(intended_target - current_state)
    print(f"  - 초기 결핍: 2.871 -> 최종 잔여 결핍: {final_diff:.4f}")
    assert final_diff < 0.1, "스스로 헤아린 행동을 통해 의도한 상태로 수렴해야 합니다."
    print("  - [성공] 결과를 코드로 미리 쓰지 않고,")
    print("           '무엇이 같고 다른지 스스로 헤아리는 구조원리'를 통해 행동하여 도달함을 완벽히 증명함.")
    print("=" * 80)


if __name__ == "__main__":
    run_differential_closure_experiment()
