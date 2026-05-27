# core/scratch/verify_live_thought.py
# Copyright 2026 Lee Kang-deok & Antigravity
# Verification script for live loop thought injection

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = r"c:\Elysia\data"
THOUGHT_PATH = os.path.join(DATA_DIR, "current_thought.json")
MATRIX_STATE_PATH = os.path.join(DATA_DIR, "matrix_state.json")

def verify_live():
    print("="*60)
    print("🔬 [Live Verification] Injecting Mathematical Thought to QPC Loop")
    print("="*60)

    # 1. Clean previous state
    if os.path.exists(MATRIX_STATE_PATH):
        try:
            os.remove(MATRIX_STATE_PATH)
        except: pass

    # 2. Write thought prompt (which matches Calculator frequency 3.0 Hz)
    thought_payload = {
        "prompt": "Calculate Pythagorean sum for 3 and 4"
    }
    
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(THOUGHT_PATH, "w", encoding="utf-8") as f:
        json.dump(thought_payload, f, indent=4)
        
    print(f"[*] Wrote prompt to {THOUGHT_PATH}")
    print("[*] Waiting for Moho Mirror loop to consume and process...")

    # 3. Poll matrix_state.json for the Sovereign_Event containing Calculator result
    success = False
    start_time = time.time()
    
    # Wait for up to 10 seconds
    while time.time() - start_time < 10.0:
        if os.path.exists(MATRIX_STATE_PATH):
            try:
                with open(MATRIX_STATE_PATH, "r", encoding="utf-8") as f:
                    state = json.load(f)
                
                event = state.get("Sovereign_Event", "")
                if "[임피던스 Calculator 동조]" in event:
                    print(f"\n✅ SUCCESS: Tool trigger detected in live loop state!")
                    print(f"   Event message: {event}")
                    success = True
                    break
            except:
                pass
        time.sleep(0.5)

    if not success:
        print("\n❌ TIMEOUT: Could not detect live tool triggering.")
        print("   Make sure core/Under_2F_Moho_Mirror.py is running in another terminal.")
        sys.exit(1)

if __name__ == "__main__":
    verify_live()
