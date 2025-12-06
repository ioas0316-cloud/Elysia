
import sys
import os
import json
import logging
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Sensory.p4_sensory_system import P4SensorySystem
from Core.Foundation.reasoning_engine import ReasoningEngine
from Core.Foundation.resonance_field import ResonanceField

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Verification")

def verify_autonomy():
    print("🧪 Verifying Autonomous Learning & Style Adaptation...")

    # 1. Setup P4 Sensory System
    p4 = P4SensorySystem()
    
    # Mock Resonance Field
    class MockResonance:
        total_energy = 80.0
        
    resonance = MockResonance()

    # 2. Trigger Pulse (Simulated)
    # Autonomy depends on random chance, so we force the internal learning method
    print("\n[Step 1] Triggering P4 Autonomous Learning...")
    try:
        # Manually invoke the internal method to guarantee execution for test
        p4._autonomous_learning(resonance)
        print("✅ P4 Pulse Executed.")
    except Exception as e:
        print(f"❌ P4 Pulse Failed: {e}")
        return

    # 3. Verify State File
    print("\n[Step 2] Checking 'elysia_state.json'...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Core level
    state_path = os.path.join(base_dir, "Core", "Creativity", "web", "elysia_state.json")
    
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
            print(f"✅ State File Found: {state.get('emotion')} | Style: {state.get('style')}")
    else:
        print("❌ State File NOT Found!")
        return

    # 4. Verify Reasoning Engine Adaptation
    print("\n[Step 3] Verifying Reasoning Engine Adaptation...")
    brain = ReasoningEngine()
    
    # Test Communication
    response = brain.refine_communication("Hello, world.", context="general")
    print(f"🧠 Brain Response: {response}")
    
    if "[Elysia's Voice adapts to:" in response:
        print("✅ Style Adaptation CONFIRMED.")
    else:
        print("⚠️ Style Adaptation NOT detected (Might need specific style values in state).")

if __name__ == "__main__":
    verify_autonomy()
