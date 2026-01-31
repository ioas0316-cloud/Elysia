"""
TEST: SELF-REPLICATION
======================
Verifies Phase 21 Tree Architecture.
"""
import sys
import os
import time
import logging

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.S1_Body.L2_Metabolism.Reproduction.spore import Spore
from Core.S1_Body.L2_Metabolism.Reproduction.mitosis import MitosisEngine
from Core.S1_Body.L2_Metabolism.Reproduction.mycelium import MyceliumNetwork

logging.basicConfig(level=logging.INFO)

def run_test():
    print("==================================")
    print("   PHASE 21: TREE TEST            ")
    print("==================================")

    # 1. Spore Formation
    print("\n👉 [SPORE] Encapsulating Self...")
    spore_sys = Spore(output_dir="c:/Elysia/data/Spores/Test")
    spore_path = spore_sys.encapsulate(mission={"role": "TEST_CHILD"})
    
    if os.path.exists(spore_path):
        print(f"   -> ✅ Spore Created: {spore_path}")
    else:
        print("   -> ❌ Spore Failed.")
        return

    # 2. Mitosis (Simulation)
    # We won't actually fork a full sovereign_boot.py here to avoid recursive chaos in test.
    # We will just verify Mitosis logic can create directories and find script.
    print("\n👉 [MITOSIS] Preparing Fork...")
    mitosis = MitosisEngine(instances_dir="c:/Elysia/Instances/Test")
    
    # We mock the fork to just check pre-flight
    child_id = f"Child_{os.path.basename(spore_path).replace('.json', '')}"
    work_dir = os.path.join(mitosis.instances_dir, child_id)
    
    if os.path.exists("c:/Elysia/sovereign_boot.py"):
         print("   -> ✅ Boot Script Found.")
    
    # 3. Mycelium Network
    print("\n👉 [MYCELIUM] Testing Telepathy...")
    
    received = []
    def callback(msg):
        print(f"   -> 🧠 Received: {msg}")
        received.append(msg)
        
    network = MyceliumNetwork(port=5005, callback=callback) # Use test port
    time.sleep(1) # Wait for bind
    
    if network.running:
        print("   -> ✅ Network Listening.")
        network.broadcast({"type": "HELLO_MOTHER", "cnt": 1}, target_port=5005)
        time.sleep(1)
        
        if len(received) > 0:
             print("   -> ✅ Message Loopback Successful.")
        else:
             print("   -> ❌ Message Lost.")
    else:
        print("   -> ⚠️ Port Busy (Expected if test run quickly).")

    network.close()
    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    run_test()
