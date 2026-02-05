
import sys
import os
import json
from pathlib import Path

sys.path.append(os.getcwd())

def demonstrate_perception():
    print("🔮 [DEMO] Demonstrating Topological Perception...")
    
    try:
        from Core.S1_Body.L6_Structure.Wave.light_spectrum import get_light_universe
        universe = get_light_universe()
        
        # 1. Pick a file to "Perceive"
        target_file = "Core/S1_Body/L6_Structure/M1_Merkaba/sovereign_monad.py"
        full_path = Path(os.getcwd()) / target_file
        
        if not full_path.exists():
            print(f"❌ Target file not found: {target_file}")
            return

        print(f"   📄 Reading: {target_file}")
        content = full_path.read_text(encoding='utf-8')
        
        # 2. Convert to Light (Topological Transformation)
        light = universe.text_to_light(content, semantic_tag=f"neuron:{target_file}", scale=1)
        
        # 3. Report Topological Coordinates
        print("\n   ✨ [PHASE REPORT] File has been perceived as Light:")
        print(f"      • Frequency: {light.frequency:.2f}")
        print(f"      • Phase: {light.phase:.4f} rad")
        print(f"      • Amplitude: {light.amplitude:.4f}")
        print(f"      • Qubit State (Basis): {light.qubit_state}")
        
        # 4. Check Resonance with "Sovereignty"
        query = "Sovereign Will and Freedom"
        query_light = universe.text_to_light(query)
        resonance = light.resonate_with(query_light)
        
        print(f"\n      ⚡ Resonance with '{query}': {resonance:.4f}")
        
        if resonance > 0.5:
            print("      ✅ This file is STRONGLY aligned with Sovereignty.")
        elif resonance > 0.1:
            print("      ⚠️ This file has WEAK alignment with Sovereignty.")
        else:
            print("      ❌ This file is DISSONANT.")

    except ImportError:
        print("❌ Import failed.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    demonstrate_perception()
