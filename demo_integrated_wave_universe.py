#!/usr/bin/env python3
"""
Elysia Demo: Integrated Wave Universe Interactive Simulation
============================================================
Simulates a hybrid Fantasy RPG & Dating Sim (미연시) Virtual Universe
powered by Clifford Rotor Spatial Manifolds and Kuramoto Wave Synchronization.

Runs seamlessly on low-end hardware (e.g., GTX 1060 3GB VRAM footprint).
"""

import time
import numpy as np
from elysia_core.integrated_wave_universe import IntegratedWaveUniverse, NPCPhaseNode


def run_interactive_wave_universe_demo():
    print("\n" + "=" * 80)
    print(" ELYSIA :: INTEGRATED WAVE UNIVERSE INTERACTIVE DEMONSTRATION")
    print("     'GTX 1060 3GB VRAM Topological Virtual Universe Runtime'")
    print("=" * 80 + "\n")

    # Initialize Engine
    universe = IntegratedWaveUniverse(primary_zone="fantasy_meadow")

    # Add additional companion NPC: Lucian (던전의 기사)
    universe.register_npc(
        NPCPhaseNode(
            name="lucian",
            title="던전의 기사 (Guardian Knight of Dissonance)",
            natural_frequency=0.8,
            phase=np.pi / 4.0,
            friction_tolerance=0.6,
            personality_vector=np.array([0.4, 0.9, 0.3, 0.2, 0.5], dtype=float)
        )
    )

    simulation_script = [
        ("idle", "안녕하세요, 두 분!", 1.0, "초원에 조용히 서서 정령 엘리시아와 기사 루시안에게 인사를 건넵니다."),
        ("move_forward", "초원 너머 미지의 유적으로 걸어갑니다.", 1.2, "클리포드 로터 회전(Rotor Spin)을 통해 맵을 재로드하지 않고 위상을 수렴합니다."),
        ("explore_meadow", "파동의 결이 고요하고 평화롭네요.", 1.2, "정령 엘리시아와의 깊은 위상 동기화(Phase-Lock) 진행."),
        ("enter_dungeon", "던전 입구로 향합니다.", 1.5, "높은 마찰력 및 불확실성이 마주치는 지점. 루시안 경이 긴장합니다."),
        ("switch_zone_dungeon", "깊은 던전 속으로 들어갑니다!", 0.7, "위상 만홀이 'dungeon_depths'로 토폴로지 점프(Topology Jump)합니다."),
        ("spin_view", "어둠 속에서 무기를 다잡으며 주위를 둘러봅니다.", 0.7, "동적 위상 재조율(Dynamic Phase Re-locking) 가동."),
        ("idle", "엘리시아, 나를 믿고 함께 가자.", 1.2, "동반자 엘리시아와의 감정 주파수가 최고조에 달하며 깊은 공명이 일어납니다.")
    ]

    print(">>> 런타임 구동을 시작합니다...\n")
    time.sleep(0.5)

    for step_idx, (action, speech, voice_freq, narrative) in enumerate(simulation_script, 1):
        print(f"\n[시나리오 스텝 {step_idx}] {narrative}")
        print(f"  • 액션: {action} | 음성 주파수: {voice_freq}Hz | 대사: \"{speech}\"")

        telemetry = universe.step_universe(
            player_action=action,
            player_speech=speech,
            player_voice_freq=voice_freq,
            dt=0.08
        )

        dashboard = universe.render_one_page_universe_dashboard()
        print(dashboard)
        time.sleep(0.3)

    print("\n" + "=" * 80)
    print(" DEMO COMPLETE: Integrated Wave Universe verified successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_interactive_wave_universe_demo()
