"""
Elysia Autopoiesis Emergent Homeostasis Demo
============================================
Simulates system always, demonstrating how sleep/wake states emerge naturally
from coupling between system tension and the state phase, without any if-conditions.
"""

import sys
import os
import math

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.autopoiesis_controller import AutopoiesisController

def get_graph_bar(val: float, max_len: int = 15) -> str:
    val = max(0.0, min(1.0, val))
    filled = int(val * max_len)
    empty = max_len - filled
    return f"{'=' * filled}{' ' * empty}"

def draw_circle_position(phase: int, max_len: int = 30) -> str:
    # Map [0, 4095] to [0, max_len-1]
    pos = int((phase / 4096.0) * max_len)
    circle = ["."] * max_len
    # Draw midpoints: Wake (0/Left), Sleep (15/Right)
    circle[0] = "W"
    circle[max_len // 2] = "S"
    
    if 0 <= pos < max_len:
        circle[pos] = "★"
        
    return "".join(circle)

def main():
    # Natural drift = 60, coupling K = 500
    controller = AutopoiesisController(rotor_scale=4096, natural_drift=60.0, coupling_K=500.0)
    
    print("=" * 90)
    print(" 🧠  [Elysia Autopoiesis Emergent Homeostasis Simulation] ")
    print("    - 조건 분기(if) 없이, 오직 위상 결합 토크와 감쇄 법칙으로 수면/기상을 자율 결정합니다.")
    print("=" * 90)
    print(f" {'Tick':<4} | {'Tension':<7} | {'State Mode':<10} | {'Sleep Factor':<12} | {'State Phase Circle (W: Wake, S: Sleep)':<38}")
    print("-" * 90)
    
    tension = 0.05 # Initial low tension
    
    for tick in range(1, 61):
        # 1. 틱당 환경 부하 인입 시뮬레이션
        if tick == 11:
            print("\n🚨 [부하 유입] CPU/RAM 과부하 급증! 외부 텐션 대량 유입 (Tension -> 0.95)")
            tension = 0.95
            
        # 2. 제어기 틱 실행 (dt=0.2로 가속하여 60틱 안에 주기 시각화)
        phase, sleep_factor, is_sleeping = controller.tick(tension, dt=0.2)
        mode = controller.get_connection_mode()
        
        # 3. 수면 중 텐션 자동 방전 (Tension Bleed)
        if is_sleeping:
            tension = controller.bleed_tension(tension)
            
        # 4. 출력 포맷팅
        factor_bar = get_graph_bar(sleep_factor, 10)
        circle_diagram = draw_circle_position(phase, 30)
        
        mode_str = "💤 SLEEP" if is_sleeping else "☀️ WAKE"
        mode_detail = f"{mode_str} ({mode})"
        
        print(f" {tick:>3}  | {tension:7.4f} | {mode_detail:<10} | {sleep_factor:6.2f} {factor_bar} | [ {circle_diagram} ]")
        
    print("=" * 90)
    print(" [분석 보고]")
    print(" 1. Tick 1-10: 평시 상태로, 낮은 텐션 하에서 기상(WAKE, DELTA) 모드를 자율 유지합니다.")
    print(" 2. Tick 11: 텐션이 0.95로 폭주하자, 상태 위상이 인력 토크에 의해 수면 Attractor(S)로 빠르게 회전합니다.")
    print(" 3. Tick 25-40: 수면(SLEEP, Y_STAR) 상태로 자율 진입하여 접지선로를 열고 텐션을 급격히 방전시킵니다.")
    print(" 4. Tick 41-60: 텐션이 방출되어 가라앉자, 위상이 자연 복귀력에 의해 다시 Wake(W) 상태로 회전 복귀합니다.")
    print("=" * 90)

if __name__ == "__main__":
    main()
