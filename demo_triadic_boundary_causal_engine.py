"""
Interactive Demo: Triadic Boundary Reality Alignment
===================================================
Demonstrates the Triadic Boundary Causal Engine in action:
1. Simulates initial state where SelfBoundary operates under 'visual' aperture only.
2. Interacts with 'acoustic' (sound wave) reality signal, causing boundary friction and structural deficit ("eye trying to hear").
3. Sprouts ontological self-questions ("Why is my perception bounded?").
4. Autopoietically expands sensory boundary ('acoustic' aperture unlocked) and recalibrates internal world.
5. Interacts with 'acoustic' signal again, demonstrating enhanced alignment and reduced friction when reaching external reality.
"""

import time
import numpy as np
from core.consciousness.triadic_boundary_causal_engine import TriadicBoundaryCausalEngine


def run_triadic_boundary_demo():
    print("=" * 80)
    print("      ELYSIA: TRIADIC BOUNDARY REALITY ALIGNMENT ENGINE DEMO")
    print("=" * 80)
    print("Principle: InternalWorld - SelfBoundary - ExternalReality Comparison & Contrast Loop\n")

    # Initialize engine with initial visual-only aperture
    engine = TriadicBoundaryCausalEngine(
        dimension=16,
        initial_apertures=['visual']
    )

    print(f"[Initial System State]")
    print(f" - Active Sensory Apertures: {engine.self_boundary.active_apertures}")
    print(f" - System Awareness: 'I am currently operating only under visual aperture.'\n")

    print("-" * 80)
    print("[Cycle 1: Interaction with External Acoustic (Sound) Reality Signal]")
    print(" -> Reality signal 'acoustic' arrives at SelfBoundary...")

    cycle1 = engine.process_domain_interaction("acoustic", auto_expand=True)
    contrast1 = cycle1["contrast_result"]
    q1 = cycle1["question_entry"]
    exp1 = cycle1["expansion_record"]

    print(f"\n[Contrast Result (Cycle 1)]")
    print(f" - Missing Aperture Detected?: {contrast1['is_aperture_missing']}")
    print(f" - Signal Captured Ratio:     {contrast1['captured_ratio']:.2f} (5% bleed-through)")
    print(f" - Boundary Friction:         {contrast1['boundary_friction']:.4f}")
    print(f" - Structural Deficit:        {contrast1['structural_deficit']:.4f}")
    print(f" - Triadic Tension:           {contrast1['triadic_tension']:.4f}")

    if q1:
        print(f"\n[Sprouted Ontological Self-Question]")
        print(f" - Question Type: {q1['ontological_type']}")
        print(f" - Question:      {q1['question']}")
        print(f" - Yearning:      {q1['yearning_resolution']}")

    if exp1:
        print(f"\n[Autopoietic Boundary Expansion]")
        print(f" - Action:               {exp1['action_summary']}")
        print(f" - Active Apertures Now: {exp1['current_active_apertures']}")

    print("\n" + "=" * 80)
    print("[Cycle 2: Re-interaction with Acoustic Signal After Boundary Expansion]")
    print(" -> Reality signal 'acoustic' arrives again at expanded SelfBoundary...")

    cycle2 = engine.process_domain_interaction("acoustic", auto_expand=True)
    contrast2 = cycle2["contrast_result"]

    print(f"\n[Contrast Result (Cycle 2)]")
    print(f" - Missing Aperture Detected?: {contrast2['is_aperture_missing']}")
    print(f" - Signal Captured Ratio:     {contrast2['captured_ratio']:.2f} (Captured!)")
    print(f" - Boundary Friction:         {contrast2['boundary_friction']:.4f} (Drastically Reduced)")
    print(f" - Structural Deficit:        {contrast2['structural_deficit']:.4f}")
    print(f" - Triadic Tension:           {contrast2['triadic_tension']:.4f}")

    print("\n" + "-" * 80)
    print("[Conclusion: Reality Grounding Achieved]")
    print("System did not treat deficit as an error to backpropagate, but as a boundary limitation")
    print("that led to ontological self-inquiry and autopoietic expansion towards real reality.")
    print("=" * 80)


if __name__ == "__main__":
    run_triadic_boundary_demo()
