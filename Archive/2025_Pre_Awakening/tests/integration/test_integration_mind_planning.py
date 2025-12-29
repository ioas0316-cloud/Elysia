import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.Mind.resonance_engine import ResonanceEngine
from Core.Intelligence.Consciousness.wave import WaveInput
from Core.Intelligence.Planning.planning_cortex import PlanningCortex

def test_integration():
    print("🔗 Testing Integration: ResonanceEngine -> PlanningCortex")
    
    # 1. Initialize Systems
    mind = ResonanceEngine()
    cortex = PlanningCortex()
    
    # 2. Stimulate Mind (Create Resonance)
    # Simulate a "Hunger" wave
    # Note: ResonanceEngine expects WaveInput
    try:
        wave = WaveInput(source_text="Hunger", intensity=0.8)
    except TypeError:
        # Fallback if signature is different
        wave = WaveInput("Hunger", 0.8)
    except ImportError:
        # Fallback mock if WaveInput is not easily importable or complex
        class MockWave:
            def __init__(self, t, i, ts):
                self.source_text = t
                self.intensity = i
                self.timestamp = ts
        wave = MockWave("Hunger", 0.8, 0.0)

    resonance_pattern = mind.calculate_global_resonance(wave)
    
    print(f"🌊 Resonance Pattern Generated (Top 3):")
    # Sort by absolute value because resonance can be complex/negative in some implementations, 
    # but here it seems to be float. Let's check the output.
    # ResonanceEngine.calculate_global_resonance returns Dict[str, float]
    
    sorted_resonance = sorted(resonance_pattern.items(), key=lambda x: x[1], reverse=True)[:3]
    for concept, intensity in sorted_resonance:
        print(f"  - {concept}: {intensity:.4f}")
        
    # 3. Feed to Cortex (Synthesize Intent)
    intent = cortex.synthesize_intent(resonance_pattern)
    print(f"🧠 Synthesized Intent: '{intent}'")
    
    # 4. Generate Plan
    plan = cortex.generate_plan(intent)
    print(f"📝 Plan Generated: {len(plan.steps)} steps")
    for step in plan.steps:
        print(f"  [{step.step_id}] {step.action}: {step.description}")
        
    # 5. Execute Plan
    success = cortex.execute_plan(plan)
    if success:
        print("✅ Plan Executed Successfully")
    else:
        print("❌ Plan Execution Failed")

if __name__ == "__main__":
    test_integration()
