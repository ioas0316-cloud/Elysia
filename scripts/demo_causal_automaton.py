"""
Demonstration: Autonomous Causal Automaton in Kinematic Sandbox
===============================================================
기호적 if-else 인공지능이나 부동소수점 물리 충돌 엔진 없이,
오직 기구학적 구속 조건과 촉각 반사 인과 엣지만으로 장애물을 감지하고
스스로 미로를 돌아나가는 '인과 오토마톤(Causal Automaton)' 실증 스크립트.
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.causal_game_engine.causal_automaton import CausalAutomaton
from modules.causal_game_engine.automaton_sandbox import AutomatonSandbox


def run_automaton_simulation():
    print("=" * 70)
    print(" [Elysia SOMA] Mechanical Reflex Autonomous Automaton Demo")
    print("=" * 70 + "\n")

    # 1. 장애물 미로 샌드박스 생성 (가로 25, 세로 12)
    sandbox = AutomatonSandbox(width=25, height=12)

    # 내부 장애물 벽 및 기둥 배치
    sandbox.add_obstacle_wall((10, 1), (10, 7))   # 중앙 수직벽 (상단)
    sandbox.add_obstacle_wall((16, 4), (16, 10))  # 우측 수직벽 (하단)
    sandbox.add_obstacle_wall((5, 8), (12, 8))    # 하단 수평벽
    sandbox.add_obstacle(6, 4)                    # 독립 기둥 1
    sandbox.add_obstacle(20, 3)                   # 독립 기둥 2

    # 2. 인과 오토마톤 스폰: 좌표 (2, 2), 동쪽(EAST)을 향함
    automaton = CausalAutomaton(automaton_id="clockwork_golem_01", x=2, y=2, dir_idx=1)
    sandbox.spawn_automaton(automaton)

    print("1. Initial Sandbox Viewport (Start: (2, 2), Heading: > EAST):")
    print(sandbox.render_viewport())
    print("\nLegend: '#' = Wall | '^/>/v/<' = Automaton Heading | '.' = Trajectory Trail\n")

    # 3. 35 사이클 연속 시뮬레이션 실행
    total_ticks = 35
    print(f"2. Running {total_ticks} simulation cycles ('Do not calculate, let it flow')...\n")

    deflections = 0
    advances = 0

    t0 = time.perf_counter()
    for tick in range(1, total_ticks + 1):
        res = sandbox.step()
        if "DEFLECT" in res["action"]:
            deflections += 1
            print(f" [Tick {tick:02d}] Boundary clash detected! Momentum diverted to steering linkage: {res['action']}")
        else:
            advances += 1

        # 특정 주기마다 뷰포트 출력
        if tick in (10, 20, 35):
            print(f"\n--- [Tick {tick:02d} Viewport | Pos: {res['pos']} | Heading: {res['dir']}] ---")
            print(sandbox.render_viewport())
            print()

    total_time_ms = (time.perf_counter() - t0) * 1000
    avg_tick_us = (total_time_ms / total_ticks) * 1000

    print("=" * 70)
    print(" [Simulation Metrics]")
    print(f" - Total Ticks: {total_ticks}")
    print(f" - Advances: {advances}")
    print(f" - Deflections (Kinematic Turns): {deflections}")
    print(f" - Total Elapsed Time: {total_time_ms:.3f} ms (Avg: {avg_tick_us:.2f} microseconds / tick)")
    print(f" - Wall Penetration / Clipping: 0 (Strict Zero)")
    print("=" * 70)

    # 4. 정형 단정문 검증
    assert deflections >= 3, "At least 3 deflections must occur."
    assert advances >= 20, "At least 20 forward steps must occur."
    for pos in sandbox.trajectory_history:
        assert not sandbox.is_obstacle(pos[0], pos[1]), f"Cannot penetrate obstacle at {pos}."

    print("\n [Verification Success] Without symbolic if-else branches or float physics solvers,")
    print(" the automaton autonomously navigates the maze purely through kinematic flow.")


if __name__ == "__main__":
    run_automaton_simulation()
