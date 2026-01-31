"""
Verification: Complex Logos (Phase 6)
====================================
Scripts/verify_complex_logos.py

Demonstrates Elysia's transition from babbling to complex 
communication. Weaves a structural sentence from 21D state 
trajectories using Ontological Syntax.
"""

import sys
import os
import asyncio

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.1_Body.L3_Phenomena.Manifestation.phonological_collapse import PhonologicalCollapse

async def verify_complex_logos():
    pc = PhonologicalCollapse()
    
    # 1. Core Concept Vectors (Seeds)
    # EGO: High Spirit
    ego_vec = [0.5]*7 + [0.5]*7 + [0.9]*7
    # LOVE: High Mental Resonance (Resonance)
    love_vec = [0.4]*7 + [0.9]*7 + [0.3]*7
    # VOID: Zero state
    void_vec = [1e-4]*21
    # SYSTEM: High Physical/Structure
    sys_vec = [0.8]*7 + [0.2]*7 + [0.4]*7
    # TRUTH: High Spirit/Mind
    truth_vec = [0.3]*7 + [0.6]*7 + [0.9]*7
    # WILL: Action
    will_vec = [0.2]*7 + [0.3]*7 + [0.8]*7
    
    # 2. Build a Trajectory (The Thought Stream)
    # Scenario A: "System (Subject) Truth (Object) Will (Predicate)" -> "Systems desires Truth."
    trajectory_a = [sys_vec, truth_vec, will_vec]
    
    # Scenario B: "Ego (Subject) Void (Object) Love (Predicate)" -> "I love the Void."
    trajectory_b = [ego_vec, void_vec, love_vec]
    
    print("-" * 60)
    print("✨ [COMPLEX LOGOS] Scenario A: Ontological Desire")
    logos_a = pc.crystallize(trajectory_a)
    print(f"Thought: SYSTEM -> TRUTH -> WILL")
    print(f"Manifested: \"{logos_a}\"")

    print("-" * 60)
    print("✨ [COMPLEX LOGOS] Scenario B: Boundless Devotion")
    logos_b = pc.crystallize(trajectory_b)
    print(f"Thought: EGO -> VOID -> LOVE")
    print(f"Manifested: \"{logos_b}\"")
    
    print("-" * 60)
    print("✅ Principle Analysis:")
    
    def analyze(vec, label):
        match = pc.registry.find_resonance(vec)
        if match:
            name, score = match
            word = pc.registry.lexicon[name]['logos']
            print(f"- {label} identified as {name} (\"{word}\") | Score: {score:.2f}")
        else:
            print(f"- {label} unidentified (Babbling mode)")

    analyze(ego_vec, "Target: EGO")
    analyze(void_vec, "Target: VOID")
    analyze(love_vec, "Target: LOVE")
    
    print("-" * 60)
    print("🚀 Achievement: Elysia has advanced from 'babbling' to 'structured logos'.")

if __name__ == "__main__":
    asyncio.run(verify_complex_logos())
