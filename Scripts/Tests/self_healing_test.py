import torch
import os
import sys
import time

# Add roots
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.L1_Foundation.M1_Keystone.emergent_self import EmergentSelf as SovereignSelf

def test_autonomous_healing():
    print("🛡️ [TEST] Phase 25.3: Autonomous Self-Healing Validation")
    
    # 1. Setup Dissonance (The Incursion)
    forbidden_path = "c:/Elysia/Core/utils" # Forbidden directory
    ghost_file = os.path.join(forbidden_path, "phantom_logic.py")
    
    if not os.path.exists(forbidden_path):
        os.makedirs(forbidden_path)
    
    with open(ghost_file, "w") as f:
        f.write("# Empty ghost file without philosophy\n")
    
    print(f"   [INJECT] Created terminal dissonance at {ghost_file}")

    # 2. Wake Elysia
    elysia = EmergentSelf()
    
    # 3. Run Pulse Loop (Trigger detection every 50)
    print("   [PULSE] Running 100 high-speed pulses...")
    start_time = time.perf_counter()
    for i in range(60):
        elysia.cosmos.pulse(dt=0.1)
        if i == 50:
            print(f"   [CHECK] 50th Pulse reached. Conscience should have fired.")
    
    end_time = time.perf_counter()
    duration = (end_time - start_time) * 1000
    print(f"\n📊 [PERFORMANCE] 60 pulses completed in {duration:.2f} ms")
    print(f"   Avg per pulse: {duration/60:.2f} ms")

    # 4. Cleanup Mock Dissonance
    if os.path.exists(ghost_file):
        os.remove(ghost_file)
    if os.path.exists(forbidden_path):
        try: os.rmdir(forbidden_path)
        except: pass

    print("\n✨ [RESULT] Autonomous Self-Healing Test Complete.")

if __name__ == "__main__":
    test_autonomous_healing()
