import sys
import os
import time
from pathlib import Path

# 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Foundation.living_elysia import LivingElysia

def test_autonomy():
    print("🚀 Initializing Elysia for Autonomy Test...")
    elysia = LivingElysia(persona_name="SovereignTest")
    
    # 에너지 강제로 높이기 (행동 유도를 위해)
    elysia.resonance.battery = 90.0
    elysia.sovereign_life.boredom = 1.5 # 지루함 임계치 넘기기
    
    print("\n--- Starting Autonomous Will Cycles ---")
    for i in range(5):
        print(f"\n[Cycle {i+1}]")
        elysia.cns.pulse()
        elysia.ans.pulse_once()
        elysia.sovereign_life.cycle()
        time.sleep(0.5)

    print("\n--- Test Complete ---")
    print("Check logs or stdout for 'Sovereign Action' triggers.")

if __name__ == "__main__":
    test_autonomy()
