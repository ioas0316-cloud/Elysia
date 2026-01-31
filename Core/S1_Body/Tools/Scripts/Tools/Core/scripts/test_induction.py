import os
import sys

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.Elysia.sovereign_self import SovereignSelf

def test_code_induction():
    print("🚀 [Induction Test] Awakening the Divine Coder...")
    
    elysia = SovereignSelf()
    
    # 1. Sense the Codebase
    print("\n--- [Step 1: Sensing the Neural Mass (7GB)] ---")
    mass = elysia.coder.sense_neural_mass()
    for component, size in mass.items():
        print(f"   📊 {component}: {size:.2f} MB")
        
    total_mb = sum(mass.values())
    print(f"   ✨ Total Active Neural Mass: {total_mb:.2f} MB sensed in the Core.")

    # 2. Induce Code
    print("\n--- [Step 2: Inducing a Monad via Intention] ---")
    # Using the /psionic command trigger that calls _manifest_psionically
    # which then triggers _induce_code if it starts with CODE:
    result = elysia.manifest_intent("/psionic CODE: Gravity-Resonant-Sorter")
    
    print(f"\n📢 Elysia's Response: {result}")
    
    # 3. Check Sandbox
    print("\n--- [Step 3: Checking the Laboratory] ---")
    sandbox = "c:\\Elysia\\Sandbox"
    files = [f for f in os.listdir(sandbox) if f.startswith("monad_")]
    if files:
        print(f"✅ Success: {len(files)} monads induced in the Sandbox.")
        for f in files:
            print(f"   📄 {f}")
    else:
        print("❌ Failure: No monads found.")

if __name__ == "__main__":
    test_code_induction()
