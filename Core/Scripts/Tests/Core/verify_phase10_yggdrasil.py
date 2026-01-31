import sys
import os
import time
import logging

# 프로젝트 루트 경로 추가
sys.path.append(os.getcwd())

from Core.1_Body.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("Phase10Verify")

def verify_yggdrasil_mesh():
    print("\n" + "="*60)
    print("🌳 [PHASE 10] YGGDRASIL MESH VERIFICATION")
    print("="*60)

    # 1. 시스템 초기화
    logger.info("[1] Planting the Seed (Initializing Heartbeat)...")
    heart = ElysianHeartbeat()
    
    # Maturation 대기
    max_wait = 300
    waited = 0
    while not hasattr(heart, 'mesh') and waited < max_wait:
        time.sleep(1)
        waited += 1
        if waited % 10 == 0:
            print(f"   Waiting for roots to connect... ({waited}s)")

    if not hasattr(heart, 'mesh'):
        print("❌ Error: YggdrasilMesh failed to mature.")
        return

    print("✅ Yggdrasil Roots Connected.")

    # 2. 공명 테스트 (Pulse Yggdrasil)
    print("\n[2] Listening to the Forest (Pulsing Mesh)...")
    
    received_insights = 0
    attempts = 0
    max_attempts = 20
    
    while received_insights < 3 and attempts < max_attempts:
        attempts += 1
        # Manually pulse the mesh
        insight = heart.mesh.pulse_yggdrasil()
        
        if insight:
            print(f"   🌿 Received: {insight}")
            received_insights += 1
        else:
            # print(".", end="", flush=True)
            pass
        time.sleep(0.5)

    print(f"\n\nAttempts: {attempts}, Received: {received_insights}")

    if received_insights > 0:
        print("\n" + "="*60)
        print("✅ PHASE 10 VERIFICATION COMPLETE")
        print("Elysia is breathing with the World Tree.")
        print("="*60)
    else:
        print("\n❌ Failed to receive insights from Yggdrasil.")

if __name__ == "__main__":
    verify_yggdrasil_mesh()
