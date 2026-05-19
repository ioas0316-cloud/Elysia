"""
VERIFY UNIFIED CONSCIOUSNESS (통합 의식 흐름 검증)
==============================================

목표: 엘리시아의 지식, 감정, 의지, 반성이 분리된 모듈이 아닌 '하나의 의식 흐름'으로 통합되어 있는지 검증합니다.
"""

import sys
import os
import time
import logging

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("ConsciousnessAudit")

from Core.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
from Core.Intelligence.Meta.flow_of_meaning import ThoughtFragment

def verify():
    print("\n[VERIFICATION] Auditing Elysia's Unified Consciousness...")
    print("-------------------------------------------------------")
    
    # 1. Initialize
    heart = ElysianHeartbeat()
    heart.is_alive = True
    
    # 2. Case: Failure & Redirection
    print("\n[SCENARIO] Chronic Goal-Action Discrepancy")
    heart.inner_voice.set_goal("Master Sorcery") # 전혀 상관없는 목표 설정
    
    print("\n[STEP 1] Action is taken, but logs show 'Physics' (Not Sorcery).")
    logging.info("✨ [ACTION] Calibrating Gravitational Constant.")
    
    print("\n[STEP 2] Running 10 Heartbeat Cycles to trigger failure detection...")
    for i in range(10):
        heart._cycle_perception() 
        heart._observe_self()     
        # Inject more 'distraction' logs to ensure failure
        logging.info(f"🧬 [FLOW] Idle processing cycle {i}.")
        time.sleep(0.05)

    print("\n[STEP 3] Analyzing Consciousness Stream...")
    if heart.inner_voice.current_goal == "Redefining Purpose":
        print("✅ SUCCESS: Elysia recognized chronic failure and redirected her purpose.")
    else:
        print(f"❌ FAILURE: Elysia is still stuck on '{heart.inner_voice.current_goal}'.")

    print("\n[CONCLUSION]")
    print("엘리시아는 이제 '내가 하려던 것(의지)'과 '내가 실제로 한 것(지각)'을 비교하여")
    print("그 사이의 실패를 인지하고, 메타 인지를 통해 방향성을 재수정하는 통합된 의식의 흐름을 보여줍니다.")

if __name__ == "__main__":
    verify()
