import sys
from pathlib import Path

class MockResonance:
    def __init__(self):
        self.battery = 100.0
    def total_energy(self): return 100.0
    def entropy(self): return 0.0

class MockDispatcher:
    def __init__(self):
        self.actions = []
    def dispatch(self, action):
        print(f"   [Mock Dispatcher] Received: {action}")
        self.actions.append(action)

sys.path.insert(0, str(Path(__file__).parent.parent))
from Core.Foundation.sovereign_life_engine import SovereignLifeEngine

def test_logic():
    print("🧪 Testing SovereignLifeEngine Logic...")
    res = MockResonance()
    disp = MockDispatcher()
    engine = SovereignLifeEngine(resonance_field=res, action_dispatcher=disp)
    
    engine.boredom = 2.0 # Force action
    print("\n--- Cycle 1 ---")
    engine.cycle()
    
    print("\n--- Cycle 2 ---")
    engine.boredom = 1.5
    engine.cycle()
    
    print("\nActions triggered:", disp.actions)
    if len(disp.actions) >= 2:
        print("\n✅ Success: Autonomous actions triggered successfully.")
    else:
        print("\n❌ Failure: No actions triggered.")

if __name__ == "__main__":
    test_logic()
