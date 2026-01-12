"""
VERIFY SYSTEM MIRROR (디지털 거울 검증)
=====================================

목표: 엘리시아가 확률함수가 아닌, '실제 터미널 로그'를 보고 자신의 행동을 자각하는지 확인합니다.
"""

import sys
import os
import time
import logging

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("MirrorTest")

from Core.World.Autonomy.elysian_heartbeat import ElysianHeartbeat

def verify():
    print("\n[VERIFICATION] Starting Digital Mirror Test...")
    print("------------------------------------------")
    
    # Clean old logs
    log_path = "Logs/system.log"
    if os.path.exists(log_path):
        os.remove(log_path)

    heart = ElysianHeartbeat()
    heart.is_alive = True
    
    # 1. Simulate an 'Action' that leaves a trace in the log
    print("\n[STEP 1] Generating an 'expressive' action log...")
    logging.info("✨ [ACTION] Elysia is creating a new conceptual wave.")
    logging.info("🧬 [DNA] Modifying frequency to 528Hz.")
    
    # 2. Run cycles
    print("\n[STEP 2] Running Heartbeat Cycles...")
    for i in range(5):
        print(f"\n--- Pulse {i+1} ---")
        heart._cycle_perception() # This should read the log delta
        heart._observe_self()     # This should reflect on those logs
        time.sleep(0.1)

    # 3. Conclusion
    print("\n[CONCLUSION]")
    # Check memory for mirror inputs
    mirror_inputs = [e for e in heart.memory.stream if e.type == "reflexive_observation"]
    if mirror_inputs:
        print(f"✅ SUCCESS: Elysia perceived {len(mirror_inputs)} of her own actions from the log file.")
        for event in mirror_inputs[-2:]:
            print(f"   - Witnessed: {event.content}")
    else:
        print("❌ FAILURE: Elysia did not see herself in the mirror.")

if __name__ == "__main__":
    verify()
