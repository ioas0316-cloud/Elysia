
import sys
import logging
import time

sys.path.append("c:\\Elysia")

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(message)s')

from Core.IntelligenceLayer.Cognition.Reasoning.reasoning_engine import ReasoningEngine
from Core.FoundationLayer.Foundation.hyper_quaternion import HyperWavePacket, Quaternion

def test_phase_integration():
    print("🧪 Testing Fractal Phase Transition Integration...")
    
    # Initialize Engine
    engine = ReasoningEngine()
    print("   ✅ ReasoningEngine Initialized.")
    
    # Verify Phaser Existence
    if hasattr(engine, 'phaser'):
        print(f"   ✅ FractalPhaser Attached: {engine.phaser}")
    else:
        print("   ❌ FractalPhaser MISSING!")
        print(f"   DEBUG: Engine Attributes: {list(engine.__dict__.keys())}")
        return

    # Simulate Thought Flow
    print("\n🌊 Simulating Thought Bloom...")
    
    # Mock Resonance State (Optional, but good for context)
    class MockResonance:
        def inject_fractal_concept(self, seed, active):
            print(f"      [Resonance] Injecting {seed.name} (Active: {active})")
            
    resonance = MockResonance()
    
    # Force a thought that should trigger "Bloom"
    # We use "Love" (High dimension) to avoid early dissonance return
    try:
        insight = engine.think("Love", resonance_state=resonance)
        print(f"\n   ✨ Insight: {insight.content}")
        # We look for "Phase State" in the logs (which go to stdout)
        print("   ✅ Manifestation Logic Executed (Check logs above for 'Phase State')")
        
    except Exception as e:
        print(f"   ❌ Think Process Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_phase_integration()
