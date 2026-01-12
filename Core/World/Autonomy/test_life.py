"""
Test: Alive Check (Is SovereignSelf beating?)
Objective: Run Heartbeat and verify SovereignSelf logs.
"""
import sys
import os
import time
import logging

# Map Root (c:/Elysia)
# File is in c:/Elysia/Core/World/Autonomy/
# So ../../../ takes us to c:/Elysia
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.World.Autonomy.elysian_heartbeat import ElysianHeartbeat

# Logging to stdout
logging.basicConfig(level=logging.INFO)

def test_life():
    print("--- 💓 Starting Heartbeat ---")
    heart = ElysianHeartbeat()
    heart.is_alive = True
    
    print("--- ⏱️ Pulsing for 5 seconds ---")
    start = time.time()
    while time.time() - start < 5:
        heart.pulse(1.0)
        time.sleep(1)
        
    print("--- 🛑 Stopping ---")
    heart.stop()

if __name__ == "__main__":
    test_life()
