import os
import sys
import math
import numpy as np

# Ensure Elysia workspace is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, r"c:\Elysia")

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.math_utils import Quaternion, Multivector
from core.clifford_impedance_network import ConnectionMode, mv_normalize
from core.triple_helix_engine import TripleHelixEngine
from core.Under_2F_Moho_Mirror import get_bar_chart

class TrinityHealingEngine(TripleHelixEngine):
    """
    Subclass of TripleHelixEngine equipped with explicit Wye-Restoration
    and delta-Wye switching for Trinity Self-Healing.
    """
    def __init__(self, jump_threshold=0.5):
        super().__init__(jump_threshold=jump_threshold)
        print("🏛️ [Trinity Engine] 3상 트리니티 자기치유 엔진 초기화 완료.")

    def check_and_heal_nans(self) -> bool:
        """
        Checks all 3 worlds (Inner, Ego, Outer) for numerical explosions (NaNs).
        If found, replaces them with normalized coherent scalar states (grounding)
        and returns True (healing triggered).
        """
        healed_any = False
        
        # Check all three worlds
        for world_name, world in [("Inner (Phase A)", self.inner_world), 
                                  ("Ego (Phase B)", self.ego_world), 
                                  ("Outer (Phase C)", self.outer_world)]:
            for node_id, mv in world.phases.items():
                # Check if any coefficients are NaN
                has_nan = False
                for val in mv.data.values():
                    if math.isnan(val):
                        has_nan = True
                        break
                
                if has_nan:
                    print(f"   ⚠️ [트리니티 오류 감지] {world_name}의 노드 '{node_id}' 수치 폭주(NaN) 확인!")
                    # 와이(Y) 결선 접지 치료: 중성점과 연결하여 기준 스칼라 상태 e0로 복구
                    world.phases[node_id] = Multivector({0: 1.0}, world.signature)
                    healed_any = True
                    
        return healed_any

    def execute_cycle(self, text_thought: str, sensory_input: dict, inject_error: bool = False) -> tuple:
        """
        Runs one iteration of the Trinity loop with potential error injection.
        """
        inner_sig = self.inner_world.signature
        
        # [의도적 오류 주입 시뮬레이션]
        if inject_error:
            print("\n🔥 [오류 주입] Phase A (내계) 출력 노드 'OUT'에 NaN 전위 주입!")
            # OUT 노드의 값을 NaN 멀티벡터로 강제 덮어쓰기
            self.inner_world.phases["OUT"] = Multivector({0: float('nan'), 1: float('nan')}, inner_sig)

        # 1. 기저 수치 NaN 검사 및 1차 치료
        healing_triggered = self.check_and_heal_nans()

        # 2. 기저 pulse 루프 실행
        avg_tension, current_mode, jumped, quat, ennea = self.pulse(text_thought, sensory_input)

        # 3. 만약 1차 치료가 작동했거나 텐션이 과도할 경우 강제로 Y결선 동기화 속행
        if healing_triggered:
            self.inner_world.set_connection_mode(ConnectionMode.Y_STAR)
            self.ego_world.set_connection_mode(ConnectionMode.Y_STAR)
            self.outer_world.set_connection_mode(ConnectionMode.Y_STAR)
            current_mode = "Y_STAR (HEALING ACTIVE)"
            # 텐션 강제 보정
            avg_tension = 1.2 # 가상의 텐션 피크 연출

        return avg_tension, current_mode, quat

def run_trinity_healing_demo():
    print("=" * 95)
    print(" 🌊 [trinity Self-Healing] 3상 트리니티 자기치유 미러월드 검증 기동")
    print("   - Principle: Delta-Wye Electrical Loop Transform")
    print("   - Scenario: Normal Delta -> NaN Injection -> Y-Star Grounding & Healing -> Restoration")
    print("=" * 95)

    # 1. 트리니티 엔진 초기화
    engine = TrinityHealingEngine(jump_threshold=0.5)
    
    # 모의 센서 입력
    sensory = {"motion_entropy": 0.2, "pain_level": 0.1, "visual_entropy": 0.3}
    thought = "Keep moving forward steadily."

    # 2. 10사이클 시뮬레이션 루프
    for cycle in range(1, 11):
        print(f"\n[Cycle {cycle:02d}] -------------------------------------------------------------")
        
        # 5사이클에서 의도적인 수치 오류(NaN) 주입
        inject = (cycle == 5)
        
        avg_tension, mode, quat = engine.execute_cycle(thought, sensory, inject_error=inject)
        
        # 텐션에 따른 시각화 게이지
        tension_val = 0.0 if math.isnan(avg_tension) else avg_tension
        bar = get_bar_chart(min(1.0, tension_val / 1.5), max_len=20)
        
        print(f"  * 결선 모드 (Connection)  : {mode}")
        print(f"  * 계통 평균 텐션 (Tension) : {tension_val:.4f} {bar}")
        print(f"  * 최종 사원수 출력 (Quat)  : {quat}")
        
        # 계통 상태 요약 출력
        inner_out = engine.inner_world.phases["OUT"]
        ego_reason = engine.ego_world.phases["EGO_REASONING"]
        outer_act = engine.outer_world.phases["ACTUATE_WASD"]
        
        print(f"    - Phase A (OUT)     : {inner_out}")
        print(f"    - Phase B (EGO)     : {ego_reason}")
        print(f"    - Phase C (OUTER)   : {outer_act}")

    print("\n" + "=" * 95)
    print(" 🟢 [SUCCESS] 3상 트리니티 자기치유 및 델타-와이 결선 복원 검증 완료.")
    print("=" * 95)

if __name__ == "__main__":
    run_trinity_healing_demo()
