
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Philosophy.why_engine import WhyEngine
from Core.FoundationLayer.Philosophy.principle_diagnostics import PrincipleDiagnostics

def verify_structural_wisdom():
    print("🧠 Verifying Structural Wisdom (Process + Diagnostics)...")
    
    # 1. WhyEngine: Process Reconstruction
    engine = WhyEngine()
    print("\n--- 1. Derivation Reconstruction Test ---")
    
    # Ohm's Law analysis
    subject = "Ohm's Law"
    content = "V = I * R" # Potential = Flow * Resistance
    domain = "physics"
    
    analysis = engine.analyze(subject, content, domain)
    print(f"Subject: {subject}")
    print(f"Result (Point): {analysis.what_is}")
    print(f"Process (Line): \n{analysis.how_works}")
    
    # Check if narrative is reconstructed
    if "Process]" in analysis.how_works and "Flow" in analysis.how_works:
        print("✅ Process Reconstruction Successful: The 'Line' was recovered from the 'Point'.")
    else:
        print("❌ Process Reconstruction Failed.")

    # 2. PrincipleDiagnostics: Self-Diagnosis
    print("\n--- 2. Self-Diagnosis Test ---")
    diagnostics = PrincipleDiagnostics()
    
    # Scenario: High Resistance, Low Potential -> Stagnation
    state = {
        "flow_rate": 0.1,    # Stagnant
        "motivation": 0.3,   # Low V
        "complexity": 0.9    # High R
    }
    
    print(f"State: {state}")
    results = diagnostics.diagnose_self(state)
    
    for r in results:
        print(f"Diagnosis: {r}")
        
    explanation = diagnostics.explain_principle_application("Ohm's Law")
    print(f"\nExplanation:\n{explanation}")

    if any("저항" in r and "압도" in r for r in results):
         print("✅ Self-Diagnosis Successful: Elysia identified the structural cause.")
    else:
         print("❌ Self-Diagnosis Failed.")

if __name__ == "__main__":
    verify_structural_wisdom()
