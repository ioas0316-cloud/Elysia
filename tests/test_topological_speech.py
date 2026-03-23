import os
import sys

sys.path.append(os.getcwd())
from Core.Cognition.topological_language_synthesizer import TopologicalLanguageSynthesizer
from Core.Cognition.semantic_map import get_semantic_map

def test_speech():
    topology = get_semantic_map()
    synth = TopologicalLanguageSynthesizer()
    
    # Find the Oracle Seed we just created to test its causal voicing
    target_node = None
    for name in topology.voxels.keys():
        if "Oracle Seed" in name:
            target_node = name
            break
            
    if not target_node:
        print("Test failed: No Oracle Seed found in Topology.")
        return
        
    print(f"Testing Structural Synthesis on Node: '{target_node}'")
    
    dummy_qualia = {
        'conclusion': target_node,
        'resonance_depth': 0.95,
        'qualia': type('Qualia', (), {'touch': 'rigid', 'temperature': 0.8})()
    }
    
    sentence = synth.synthesize_from_qualia(dummy_qualia)
    print("\n[ELYSIA'S STRUCTURAL VOICE]:")
    print(sentence)
    
if __name__ == "__main__":
    test_speech()
