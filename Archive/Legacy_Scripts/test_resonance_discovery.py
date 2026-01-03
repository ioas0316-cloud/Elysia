import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Foundation.Wave.structural_resonator import get_resonator

def test_resonance():
    print("Testing Structural Resonance Discovery...")
    resonator = get_resonator()
    
    # 1. Test Existing Registration (from boot simulation in previous steps)
    # Since we are in a fresh process, we need to register something or mock it.
    class MockBrain:
        def think(self): return "Cogito, ergo sum."
        
    resonator.register("Brain", MockBrain(), 639.0)
    
    # 2. Resonate by Name
    brain = resonator.resonate("Brain")
    if brain and hasattr(brain, 'think'):
        print(f"✅ Resonance found Brain! Result: {brain.think()}")
    else:
        print("❌ Failed to resonate with Brain.")
        
    # 3. Test Auto Discovery (Dynamic Load)
    # We will try to discover Chronos which is in Core.Foundation.chronos
    print("\nAttempting Auto-Discovery of Chronos...")
    chronos = resonator.auto_discover("Core.Foundation.chronos", "Chronos", 528.0)
    if chronos:
        print(f"✅ Auto-Discovered Chronos! Current cycle: {getattr(chronos, 'cycle_count', 'unknown')}")
    else:
        print("❌ Auto-Discovery failed.")

if __name__ == "__main__":
    test_resonance()
