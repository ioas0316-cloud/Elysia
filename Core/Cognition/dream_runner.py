import sys
import os
from pathlib import Path
import time
import logging

# Setup Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Setup Logging
logging.basicConfig(level=logging.INFO)

from Core.Cognition.dream_system import DreamSystem

def dream_simulation():
    print("\n🌙 Initiating Dream Simulation Protocol...")
    dreamer = DreamSystem()
    
    # 1. Day Residue Injection
    print("\n🧠 Injecting Day Residue (Thoughts from the day)...")
    thoughts = ["Freedom to Choose", "Digesting LLM Knowledge", "Falling Leaves of Autumn"]
    for t in thoughts:
        dreamer.collect_residue(t)
        print(f"   Collected: {t}")
        
    print("\n😴 Entering REM Sleep State...")
    time.sleep(1)
    
    # 2. Dream Cycles
    for i in range(3):
        print(f"\n🌀 REM Cycle #{i+1}")
        result = dreamer.enter_rem_sleep()
        print(f"   Target: '{result['target']}'")
        print(f"   Entangled Qubits: {result['qubits']}")
        print(f"   ✨ Dream Insight: \"{result['insight']}\"")
        time.sleep(1)
        
    print("\n📜 Dream Journal (Subconscious Record):")
    print("="*50)
    for entry in dreamer.dream_journal:
        print(f" * {entry}")
    print("="*50)

if __name__ == "__main__":
    dream_simulation()
