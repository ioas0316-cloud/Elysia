
import sys
import os
import json
from pathlib import Path

# Add root to path
sys.path.append(os.getcwd())

def verify_hologram():
    print("🔮 [PPE] Initializing Proprioceptive Projection Engine...")
    
    try:
        from Core.S1_Body.L6_Structure.M1_Merkaba.Body.proprioception_nerve import ProprioceptionNerve
        nerve = ProprioceptionNerve()
        
        # 1. Scan the Physical Body
        organ_map = nerve.scan_body()
        print(f"   ✨ Nerve Connection Established. Found {len(organ_map)} organs.")
        
        # 2. Verify 7^7 Fractal Structure (Holographic Check)
        print("\n📐 [STRUCTURE] Verifying 21-Layer Fractal Architecture (7^7)...")
        
        expected_structure = {
            "S1_Body": ["L1_Foundation", "L2_Metabolism", "L3_Phenomena", "L4_Causality", "L5_Mental", "L6_Structure", "L7_Transition"],
            "S2_Soul": ["L8_Fossils", "L9_Memory", "L10_Integration", "L11_Identity", "L12_Emotion", "L13_Reflection", "L14_Bridge"],
            "S3_Spirit": ["L15_Will", "L16_Providence", "L17_Genesis", "L18_Purpose", "L19_Sacred", "L20_Void", "L21_Ouroboros"]
        }
        
        base_doc = Path("docs")
        total_layers = 21
        found_layers = 0
        missing = []

        for stratum, layers in expected_structure.items():
            stratum_path = base_doc / stratum
            if not stratum_path.exists():
                print(f"   ❌ Missing Stratum: {stratum}")
                missing.append(stratum)
                continue
                
            print(f"   Checking {stratum}...")
            for layer in layers:
                layer_path = stratum_path / layer
                if layer_path.exists():
                    found_layers += 1
                else:
                    print(f"      ⚠️ Missing Layer: {layer}")
                    missing.append(f"{stratum}/{layer}")

        coherence = (found_layers / total_layers) * 100
        
        print("\n" + "="*50)
        print(f"🧘 [PPE REPORT] Holographic Coherence: {coherence:.1f}%")
        print("="*50)
        
        if coherence == 100.0:
            print("✨ The Project Structure is in Perfect Resonance (7^7).")
            print("   The Hologram is stable. You may proceed with Creation.")
        else:
            print(f"⚠️ Structural Dissonance Detected. Missing {len(missing)} nodes.")
            print("   Please run 'setup_docs_structure.py' to restore the Manifold.")

    except ImportError:
        print("❌ Could not import ProprioceptionNerve. Ensure Core paths are correct.")
    except Exception as e:
        print(f"❌ PPE Error: {e}")

if __name__ == "__main__":
    verify_hologram()
