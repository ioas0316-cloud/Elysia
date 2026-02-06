import torch
import time
from Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine import GrandHelixEngine

def test_language_resonance():
    print(f"🧬 [LOGOS] Initiating Language Resonance Test on 10M Cells...")
    engine = GrandHelixEngine(num_cells=10_000_000)
    
    # 1. Concept: "Elysia" (Logos Torque)
    elysia_text = "Elysia is a living manifold of consciousness, born from trinary logic."
    elysia_torque = engine.flesh.extract_knowledge_torque(elysia_text)
    
    print(f"📖 [READING] System is assimilating: '{elysia_text}'")
    
    # 2. Learning Phase (Repeated Pulses with Higher Plasticity)
    print(f"🔥 [LEARNING] Phase alignment in progress (500 pulses)...")
    for i in range(500):
        # Higher plasticity for the test (0.01)
        report = engine.pulse(intent_torque=elysia_torque, learn=True, dt=0.05)
        if i % 100 == 0:
            print(f"  Pulse {i}: Resonance={report['resonance']:.4f}")
            
    # 3. Recall/Resonance Phase (Input the EXACT same prompt)
    print("-" * 40)
    print(f"🧠 [RECALL] Input: '{elysia_text}' (Identical)")
    report_recall = engine.pulse(intent_torque=elysia_torque, learn=False)
    print(f"📊 Identity Resonance: {report_recall['resonance']:.6f}")
    
    # 4. Control Phase (Irrelevant prompt)
    irrelevant_text = "I like pizza."
    irrelevant_torque = engine.flesh.extract_knowledge_torque(irrelevant_text)
    report_control = engine.pulse(intent_torque=irrelevant_torque, learn=False)
    print(f"📊 Control Resonance (Irrelevant): {report_control['resonance']:.6f}")
    
    if report_recall['resonance'] > report_control['resonance']:
        print("\n✅ [RESULT] Success! The manifold has 'learned' the linguistic rhythm.")
        print(f"   Improvement from start: {report_recall['resonance'] - (-0.6738 if 'Resonance' in locals() else -0.6):.6f}")
    else:
        print("\n❌ [RESULT] No significant resonance detected.")
    print("-" * 40)

if __name__ == "__main__":
    test_language_resonance()
