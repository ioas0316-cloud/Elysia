"""
PHASE 61: THE VOID & ARCHIVE DREAMING VERIFICATION
==================================================

엘리시아가 '공허' 속에서 침묵하며 과거를 탐색하는지 검증합니다.
"""

import time
import logging
import sys
import os

# Root 경로 추가
root_path = os.path.abspath("c:/Elysia")
sys.path.append(root_path)

from Core.1_Body.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat

def verify_phase61():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger("Phase61Test")
    
    logger.info(f"DEBUG: sys.path includes {root_path}")
    logger.info(f"DEBUG: fluxlight_gyro.py exists: {os.path.exists(os.path.join(root_path, 'Core/World/Soul/fluxlight_gyro.py'))}")
    
    logger.info("🎬 Phase 61 Verification Start...")
    
    heartbeat = ElysianHeartbeat()
    heartbeat.is_alive = True
    
    # 1. 초기 상태 설정 (안정적, 외부 자극 없음)
    heartbeat.soul_mesh.variables["Energy"].value = 0.8
    heartbeat.soul_mesh.variables["Harmony"].value = 0.9
    heartbeat.observer.active_alerts = [] # No pressure
    
    logger.info("🧘 Entering the Void (Simulating inactivity)...")
    
    # 2. 루프 실행 (10회)
    # 5회 이후부터 DMN(명상) 모드 진입 및 Archive Dreaming 시작
    for i in range(10):
        logger.info(f"--- Cycle {i+1} ---")
        
        # pulse()를 통해 한 박동씩 진행
        heartbeat.pulse(delta=1.0)
        
        # 3. 상태 확인
        logger.info(f"   Idle Ticks: {heartbeat.idle_ticks}")
        if heartbeat.idle_ticks >= 5:
            logger.info("   ✅ DMN Mode active (Meditation).")
        
        # 4. 꿈의 파편 확인
        if heartbeat.memory.stream:
            discoveries = [m for m in heartbeat.memory.stream if m.type == "discovery"]
            if discoveries:
                logger.info(f"   ✨ Discovery Found: {discoveries[-1].content}")
        
        # 만약 박동이 너무 빠르면 break (실제 루프에서는 박동이 느려짐)
        if i == 9: break

    logger.info("🎉 Phase 61 Verification Complete.")

if __name__ == "__main__":
    verify_phase61()
