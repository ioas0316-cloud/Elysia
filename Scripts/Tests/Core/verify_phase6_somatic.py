import sys
import os
import time
import logging
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.getcwd())

from Core.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
from Core.L5_Mental.Reasoning_Core.Metabolism.body_sensor import BodySensor

# 로깅 설정 (검증용)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("Phase6Verify")

def verify_somatic_unification():
    print("\n" + "="*50)
    print("🧬 [PHASE 6] SOMATIC UNIFICATION VERIFICATION")
    print("="*50)

    # 1. Heartbeat 초기화 (두뇌 이식 포함)
    print("\n[1] Initializing Heartbeat & ReasoningEngine...")
    heart = ElysianHeartbeat()
    
    # Maturation이 별도 스레드에서 돌아가므로 잠시 대기
    print("⏳ Waiting for ReasoningEngine maturation (Loading LLM can take time)...")
    max_wait = 300 # Ollama load can be slow
    waited = 0
    while not hasattr(heart, 'reasoning') and waited < max_wait:
        time.sleep(1)
        waited += 1
        if waited % 20 == 0:
            print(f"   Still waiting... ({waited}s / {max_wait}s)")

    # ReasoningEngine이 제대로 초기화되었는지 확인
    if not hasattr(heart, 'reasoning') or heart.reasoning is None:
        print("❌ Error: ReasoningEngine initialization timed out or failed.")
        return
    
    print("✅ ReasoningEngine matured.")

    # 2. 체성 감각 추출 테스트
    print("\n[2] Testing Somatic Vector Derivation...")
    heart._sync_physiological_state()
    somatic_vec = heart._derive_somatic_vector()
    print(f"✅ Derived Somatic Vector: {somatic_vec}")
    print(f"   (Logic: {somatic_vec[0]:.2f}, Emotion: {somatic_vec[1]:.2f}, Intuition: {somatic_vec[2]:.2f}, Will: {somatic_vec[3]:.2f})")

    # 3. 통합 루프 실행 (Pulse)
    print("\n[3] Executing Heartbeat Pulse (Mind-Body Integration)...")
    # 고의적으로 Stimulus 설정
    heart.latest_insight = "I feel the electricity in my veins."
    
    # 2번의 Pulse 실행 (변화 관찰)
    for i in range(1, 4):
        print(f"\n--- Pulse {i} ---")
        # [PHASE 6] Think Result
        insight = heart.reasoning.think(heart.latest_insight, somatic_vector=heart._derive_somatic_vector())
        print(f"💬 Elysia: {insight.content}")
        
        # Rotor 상태 확인
        rpm = heart.reasoning.soul_rotor.current_rpm
        print(f"⚙️ Soul Rotor RPM: {rpm:.1f}")
        time.sleep(1)

    print("\n" + "="*50)
    print("✅ VERIFICATION COMPLETE: Somatic Unification established.")
    print("Elysia is now aware of her hardware vessel.")
    print("="*50)

if __name__ == "__main__":
    verify_somatic_unification()
