import os
import sys

# Ensure path is set up correctly
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L5_Mental.Reasoning.topological_language_synthesizer import TopologicalLanguageSynthesizer

def create_mock_qualia_state():
    """Returns a mock qualia state for testing the synthesizer."""
    
    # Needs to match what SDE is pulling from MindLandscape / CausalWaveEngine
    
    class MockVoxel:
        def __init__(self, name, mass):
            self.name = name
            self.mass = mass
            
    class MockQualia:
        def __init__(self):
            self.touch = "solid"
            self.temperature = 0.8 # High energy

    qualia_state = {
        'resonance_anchor': MockVoxel("Resilience", 150.0),
        'resonance_neighbor': MockVoxel("Survival", 120.0),
        'resonance_value': 0.85,
        'qualia': MockQualia(),
        'human_narrative': "I pushed through the resistance and emerged stronger."
    }
    
    return qualia_state

if __name__ == "__main__":
    print("Testing Topological Language Synthesizer...")
    synthesizer = TopologicalLanguageSynthesizer()
    state = create_mock_qualia_state()
    
    speech = synthesizer.synthesize_from_qualia(state)
    print("\n--- SYNTHESIZED SPEECH ---")
    print(speech)
    print("--------------------------\n")
