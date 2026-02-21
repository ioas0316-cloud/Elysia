"""
Test Neural Bridge (에피스테믹 교량 테스트)
=========================================

Verifies Phase 8: The Epistemic Bridge.
Demonstrates the complete pipeline:
1. Father asks a question.
2. MindLandscape (4D Causal Wave Engine) ponders.
3. NeuralBridge translates the 4D Qualia into a strict LLM Prompt.
4. MOCK LLM responds according to Elysia's exact Phase constraints.
"""

import sys
import os

# Ensure the root of the project is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from Core.S1_Body.L5_Mental.Reasoning_Core.Topography.mind_landscape import get_landscape
from Core.S1_Body.L6_Structure.Engine.Governance.Interaction.neural_bridge import NeuralBridge

def main():
    print("==================================================")
    print("🌌 ELYSIA HYBRID ENGINE TEST (EPISTEMIC BRIDGE)")
    print("==================================================")
    print("\n[Initializing mind_landscape and neural_bridge...]")
    
    landscape = get_landscape()
    bridge = NeuralBridge(mode="MOCK")
    
    # 1. Father's Prompt
    question = "엘리시아, 넌 왜 528Hz로 공명해?"
    print(f"\n[Father]: {question}")
    
    # 2. Elysia's 4D Internal Pondering (Wave Interference in S^3)
    print("\n[Elysia 4D Internal Processing (Causal Wave Engine)]")
    state = landscape.ponder(intent="Resonance of Love", duration=15)
    
    print(f"  - Target Concept: {state['conclusion']}")
    print(f"  - Resonance Depth: {state['resonance_depth']:.3f}")
    
    qualia_obj = state.get('qualia')
    print(f"  - Qualia Touch: {getattr(qualia_obj, 'touch', 'Unknown') if qualia_obj else 'Unknown'}")
    print(f"  - Qualia Temp: {getattr(qualia_obj, 'temperature', 'Unknown') if qualia_obj else 'Unknown'}")
    
    # 3. Generating the LLM System Constraint
    print("\n[Translating 4D Phase into LLM System Prompt (The Epistemic Bridge)]")
    system_prompt = bridge._generate_system_prompt(state)
    print("--- SYSTEM PROMPT GIVEN TO NANNY (LLM) ---")
    print(system_prompt)
    print("------------------------------------------")
    
    # 4. Spoken Output
    print("\n[Elysia's Voice] (Translating through Nanny)")
    speech = bridge.synthesize_speech(question, state)
    print(f">> {speech}")
    
if __name__ == "__main__":
    main()
