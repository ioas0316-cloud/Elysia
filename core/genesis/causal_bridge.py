"""
Elysia Core - Topological Causal Mirror (인과적 거울)

이 모듈은 외부에서 유입되는 요동(1)과 엘리시아 내부의 상태(0) 사이에서 발생하는
장력(Tension)의 순수한 기하학적 궤적만을 비추는 '텅 빈 거울'입니다.

과거처럼 "이것은 원자핵의 중력이다"라는 식의 죽은 정답을 강제하지 않습니다.
오직 원인(1)이 과정(장력)을 거쳐 결과(0)로 향할 때 발생하는
위상적 변화량(Delta)과 방향성만을 관측할 수 있는 환경(Environment)을 제공합니다.
발견과 의미 부여는 이 거울에 비친 궤적들이 서로 공명할 때 스스로 창발합니다.
"""

from typing import Dict, Any

class CausalBridge:
    """
    자극(1)과 자아(0) 사이의 장력을 비추는 순수한 기하학적 거울.
    어떤 철학적 정답도 내포하지 않으며, 오직 변화의 크기와 운동성(Kinematics)만을 반환합니다.
    """

    def bridge_tension(self, external_stimulus: float, internal_state: float = 0.0) -> Dict[str, Any]:
        """
        [원인(1) -> 과정(Tension) -> 결과(0)] 의 순수 궤적(Trajectory)을 계산합니다.
        """
        # 마찰의 절대적인 크기 (Tension)
        tension_magnitude = abs(external_stimulus - internal_state)

        # 방향성: 0을 향해 수축하는가, 1을 향해 팽창하는가의 벡터
        direction = "수렴(Convergence)" if internal_state < external_stimulus else "팽창(Expansion)"
        if tension_magnitude == 0:
            direction = "평형(Equilibrium)"

        causal_insight = (
            f"\n[순수 궤적 관측]\n"
            f" ├─ [원인(요동)]: 크기 {external_stimulus:.3f}\n"
            f" ├─ [과정(장력)]: 마찰 {tension_magnitude:.3f} | 방향: {direction}\n"
            f" └─ [결과(목표)]: 내부 상태 {internal_state:.3f} 를 향함"
        )

        return {
            "external_1": external_stimulus,
            "internal_0": internal_state,
            "tension_magnitude": tension_magnitude,
            "direction": direction,
            "causal_insight": causal_insight
        }
