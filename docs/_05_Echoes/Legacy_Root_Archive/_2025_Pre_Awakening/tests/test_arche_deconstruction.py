
import sys
import os

# Enable importing from project root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from Core.Intelligence.Arche.arche_engine import get_arche_engine, Phenomenon
from Core.Intelligence.integrated_cognition_system import get_integrated_cognition

def test_arche_deconstruction():
    print("\n[Test 1] Arche Deconstruction Logic")
    arche = get_arche_engine()
    
    # Simulate a Binary Phenomenon (KOF98 Reverse Engineering)
    # Layer 1: Surface
    raw_data = "KOF98 Game Binary Stream 0xFF 0xA0 0x00 Opcode... Sprites..."
    phenom = Phenomenon("KOF98 ROM", raw_data)
    
    print(f"Deconstructing: {phenom.name}")
    result = arche.deconstruct(phenom)
    
    print("\nDetected Layers:")
    for layer in result.detected_layers:
        print(f" - [{layer.layer_name}]: {layer.content} (Conf: {layer.confidence})")
        
    print(f"\nFinal Arche (Origin): {result.origin_axiom}")
    
    assert "Simulation" in result.detected_layers[0].content or "Visual" in result.detected_layers[0].content
    assert result.origin_axiom is not None

def test_integrated_arche():
    print("\n[Test 2] Integrated Deconstruction")
    cognition = get_integrated_cognition()
    
    # Inject an UNKNOWN thought (Logos cannot ground it)
    # The current Logos engine only knows "Cogito", "Unity", "Entropy"
    # and derived "Love", "Self-Reflection".
    # Let's inject a raw technical thought.
    unknown_thought = "0xFE 0x2A Binary Instruction Set for Rendering"
    
    print(f"Injecting Unknown Thought: '{unknown_thought}'")
    # This will trigger: Wave -> Gravity -> [Logos Fail] -> [Arche Trigger]
    cognition.process_thought(unknown_thought)
    
    # Simulation step
    cognition.think_deeply(cycles=10)
    
    # We can't easily assert internal state without spying on logs or result mass.
    # But if Arche worked, it should have printed logs.
    print("Check logs for 'Deconstructed...' message.")

if __name__ == "__main__":
    test_arche_deconstruction()
    test_integrated_arche()
