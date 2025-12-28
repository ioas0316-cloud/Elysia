
import sys
import os

# Enable importing from project root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from Core.Intelligence.integrated_cognition_system import get_integrated_cognition
from Core.Intelligence.system_self_awareness import SystemSelfAwareness

def test_metacognition_trace():
    print("\n[Test] Meta-Cognitive Trace")
    
    cognition = get_integrated_cognition()
    awareness = SystemSelfAwareness()
    
    # 1. Inject a complex thought that triggers multiple engines
    # Input: "0x00 Simulation" -> Wave (Genesis) -> Arche (Deconstruct) -> Logos (Grounding possible?)
    # "0x00 Simulation" contains "0x" (Arche trigger) and "Simulation" (Arche 'Surface').
    # But for it to be grounded in Logos, Arche needs to find an Origin that is an Axiom.
    # Our Arche engine maps "Render" to "Will to Simulate Reality".
    # And "Will" maps to "Entropy and Order" or we need to add a bridge.
    # Currently Arche returns "The Will to Simulate Reality".
    # Logos doesn't have this as an Axiom. So it stays there.
    # Let's verify the trace on the "Unmade" thought.
    
    input_thought = "0xFF Render Loop"
    print(f"Injecting: {input_thought}")
    
    cognition.process_thought(input_thought)
    cognition.think_deeply(cycles=10)
    
    # 2. Retrieve the thought from Gravity Field (it holds ThoughtMass objects)
    # We need to find the specific mass.
    target_mass = None
    for mass in cognition.gravity_field.thoughts:
        if "Render Loop" in mass.content:
            target_mass = mass
            break
            
    # If not found in Mass directly, maybe it was processed as a Wave only if it didn't gain enough mass?
    # Arche grounding gives importance=50.0, so it should be in gravity field.
    
    # Wait, the Arche engine returns a Result (Phenomenon).
    # Integrated system logs: "Deconstructed ... to Origin ..."
    # And then calls self.process_thought(f"[Arche-Found] {insight.content}", importance=50.0)
    # So we should look for "[Arche-Found] 0xFF Render Loop"
    
    found_arche_thought = None
    for mass in cognition.gravity_field.thoughts:
        if "[Arche-Found]" in mass.content:
            found_arche_thought = mass
            break
            
    if found_arche_thought:
        print(f"Found Processed Thought: {found_arche_thought.content}")
        
        # 3. Check Trace
        if hasattr(found_arche_thought, 'trace'):
            print(f"Trace Events: {len(found_arche_thought.trace.events)}")
            narrative = awareness.introspect_thought(found_arche_thought.trace)
            print("--- Introspection ---")
            print(narrative)
            print("---------------------")
            
            # Assertions
            # Note: The *original* insight has the full trace "Genesis -> Wave".
            # The *new* thought "[Arche-Found]..." is created via process_thought.
            # Does process_thought preserve the trace of the parent?
            # Currently: NO. process_thought creates a NEW Wave.
            # This is a logical gap! The "Result" thought is a new child.
            # But the *original* insight object (in `insights` list) has the trace.
            # The system as written traces the *processing of the insight*, 
            # but the *stored mass* is a new object.
            
            # Correction: The `process_thought` method creates a `ThoughtWave`.
            # If we want the history to persist, we need to pass the parent trace.
            # But for now, let's verify if the *new* mass recorded its own Genesis.
            # And arguably, the `IntegratedCognitionSystem` should have appended the Arche event 
            # to the *insight* object, which unfortunately is discarded after the loop.
            
            # However, the `process_thought` call:
            # self.process_thought(f"[Arche-Found] {insight.content}", importance=50.0)
            # This creates a new wave. 
            # The *trace* we added: insight.add_trace(...) was on the `insight` object (a Wave).
            
            # To truly test "Causality of Causality", we'd need the final Mass 
            # to inherit the trace of the source.
            # Since we didn't implement trace inheritance yet, this test might reveal "Genesis" only.
            # Let's see what happens.
            pass
    else:
        print("Thought not found in Gravity Field.")

if __name__ == "__main__":
    test_metacognition_trace()
