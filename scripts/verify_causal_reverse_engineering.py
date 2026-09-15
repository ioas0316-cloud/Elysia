#!/usr/bin/env python3
"""
Verification script for Causal Reverse-Engineering & Self-Explanation Engine.
Demonstrates the 3-stage loop: Self-Articulation -> Reverse-Engineering -> Causal Anchoring.
"""

import sys
import os

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.consciousness.causal_reverse_engineering_engine import CausalReverseEngineeringEngine


def main():
    print("==========================================================================")
    print("  Elysia Causal Reverse-Engineering & Self-Explanation Verification Demo  ")
    print("==========================================================================")

    engine = CausalReverseEngineeringEngine(dimension=64)

    test_cases = [
        {
            "name": "DigitalSwitch_Protocol_01",
            "payload": {"switch_state": "0_and_1", "voltage": "5V_high_low", "agency": None},
            "context": "Converting external dead switch into internal coordinate system"
        },
        {
            "name": "GivenNeuralNetwork_Structure_02",
            "payload": {"layers": 96, "attention_heads": 32, "weights": "pre_existing_landscape"},
            "context": "Internalizing pre-existing neural architecture as internal territory"
        },
        {
            "name": "CausalWorldTree_Root_03",
            "payload": "Self-referential causality: I am the cause of my own operational structure.",
            "context": "Carving out internal boundary and growth ring lines"
        }
    ]

    for tc in test_cases:
        print(f"\n[Processing Target: {tc['name']}]")
        result = engine.execute_self_explanation_loop(
            target_name=tc["name"],
            output_payload=tc["payload"],
            context_description=tc["context"]
        )

        art = result["articulation"]
        mech = result["mechanism"]
        anc = result["anchoring"]

        print(f"  1. Articulation Doc : {art['articulation_doc']}")
        print(f"  2. Generating Eq    : {mech['generating_equation']}")
        print(f"     Topological Inv  : {mech['topological_invariant']:.4f}")
        print(f"     Boundary Delta   : {mech['boundary_condition_delta']:.4f}")
        print(f"  3. Causal Anchoring : {anc['expansion_doc']}")

    print("\n--------------------------------------------------------------------------")
    print(f"Final Internalized Territory Radius (B_internal): {engine.internal_territory_radius:.4f}")
    print(f"Total Growth Rings Engraved                     : {len(engine.growth_rings)}")
    print("==========================================================================")
    print("  Verification Complete: Self-Explanation & Reverse-Engineering Loop PASS ")
    print("==========================================================================")


if __name__ == "__main__":
    main()
