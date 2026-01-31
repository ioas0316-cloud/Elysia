import os
import sys

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.1_Body.L6_Structure.Engine.code_field_engine import CODER_ENGINE

def test_engine_directly():
    print("🚀 [Engine Test] Testing Code-Field Sensing & Induction...")
    
    # 1. Sense Neural Mass
    print("\n--- [Step 1: Sensing Active Neural Circuits] ---")
    mass = CODER_ENGINE.sense_neural_mass()
    for component, size in mass.items():
        print(f"   📊 {component}: {size:.2f} MB")
        
    total_mb = sum(mass.values())
    print(f"   ✨ Active Neural mass sensed: {total_mb:.2f} MB.")

    # 2. Induce Code
    print("\n--- [Step 2: Inducing a Monad Script] ---")
    file_path = CODER_ENGINE.induce_monad_code("Self-Healing-Memory-Buffer")
    
    if os.path.exists(file_path):
        print(f"✅ Success: Monad induced at {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            print("\n--- [Generated Code Snippet] ---")
            print(f.read())
            print("--------------------------------")
    else:
        print("❌ Failure: Induction failed.")

if __name__ == "__main__":
    test_engine_directly()
