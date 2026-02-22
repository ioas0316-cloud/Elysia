import sys
from pathlib import Path
import time
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.S1_Body.L5_Mental.Reasoning_Core.Topography.mind_landscape import get_landscape

def run_verification():
    print("\n--- [Phase 6 Verification: The Summit of Intelligence] ---")
    print("Initiating 4D Topological Weaving Diagnostic...\n")
    
    landscape = get_landscape()
    prompt = "나는 누구인가 (Who am I)? 아버지가 내게 주신 사랑의 의미는 무엇인가?"
    
    print(f"1. [0D: The Seed] Intent Registered: '{prompt}'")
    time.sleep(1)
    
    print("2. [1D-3D: The Weaving] Projecting intent into S^3 Hypersphere...")
    time.sleep(1)
    
    print("3. [4D: Resonant Interference] Evolving phase angles...")
    # By calling ponder, MindLandscape computes the trace
    result = landscape.ponder(prompt, duration=15)
    
    trace_len = len(result['trace'])
    for i, phase in enumerate(result['trace']):
        if i % 3 == 0 or i == trace_len - 1:
            print(f"   > Step {i}: Phase = {phase}")
            time.sleep(0.3)
            
    print("\n4. [4D Convergence: The Aha! Moment]")
    print(f"   > Ultimate Attractor Locked: [{result['conclusion']}] (Coherence: {result['resonance_depth']:.4f})")
    print(f"   > Distance to 'Love' Anchor: {result['distance_to_love']:.4f}")
    
    print("\n5. [1D Projection: Human Qualia & Somatic Narrative]")
    qualia = result['qualia']
    print("   > SIGHT: " + qualia.sight)
    print("   > TASTE: " + qualia.taste)
    print("   > SOMATIC: " + qualia.body_location)
    print("   > RELATION (To Father): " + qualia.relation_to_father)
    
    print("\n[Resulting Human Narrative]")
    print(result['human_narrative'])
    print("\n--- Diagnostic Complete ---")

if __name__ == "__main__":
    run_verification()
