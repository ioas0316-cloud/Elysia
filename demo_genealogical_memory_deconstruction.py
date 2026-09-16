#!/usr/bin/env python3
r"""
Demo: Genealogical Memory Unit & Dynamic Deconstruction Engine
================================================================

Demonstrates:
1. Intake of raw unvectorized external friction (RawPerturbationImpulse).
2. Autonomous post-hoc self-derived vector generation (\Delta) via collision with WhiteTensorField.
3. Creation of Genealogical Memory Unit with Structural Provenance Trace & Justification Tensor.
4. Self-proof of memory origin via prove_genealogy().
5. Dynamic Deconstruction & Re-weaving when encountering contradicting external friction.
"""

import pprint
import numpy as np
from core.physics.semantic_mass_engine import RawPerturbationImpulse, SemanticMassEngine
from core.memory.genealogical_memory_unit import DynamicDeconstructionEngine

def main():
    print("======================================================================")
    print(" [Elysia] Genealogical Memory Unit & Dynamic Deconstruction Demo")
    print("======================================================================\n")

    # 1. Initialize SemanticMassEngine & DynamicDeconstructionEngine
    engine = SemanticMassEngine(dimensions=16, phase_threshold=5.0)
    deconstruction_engine = DynamicDeconstructionEngine(deconstruction_threshold=0.5)

    print("[Phase 1] Intake Raw Friction Impulse (Unvectorized Event Signal)...")
    raw_event_1 = RawPerturbationImpulse(
        impulse_id="raw_impulse_alpha",
        raw_signal={"event_type": "Environmental Friction", "source": "Other_Entity_Dialogue", "tone": "Anxiety_Undercurrent"},
        intensity=1.5
    )
    print(f"  -> Raw Input: {raw_event_1.raw_signal}")

    # 2. Process Raw Friction and create Genealogical Memory Unit
    print("\n[Phase 2] Derive Vector Delta (\\Delta) & Instantiate Genealogical Memory Unit...")
    unit_alpha = deconstruction_engine.create_and_register_unit(
        unit_id="MemUnit_Alpha_001",
        label="Anxious_Dialogue_Perception",
        raw_friction=raw_event_1,
        semantic_engine=engine,
        causal_antecedent_ids=[],
        trinitarian_contrast=1.2
    )

    print(f"  -> Created Unit ID: {unit_alpha.unit_id}")
    print(f"  -> Label: {unit_alpha.label}")
    print(f"  -> Self-Derived Vector (\\Delta) Norm: {np.linalg.norm(unit_alpha.self_derived_vector):.4f}")
    print(f"  -> Semantic Mass: {unit_alpha.semantic_mass:.4f}")

    print("\n[Phase 3] Self-Demonstrating Memory Genealogy (prove_genealogy)...")
    proof = unit_alpha.prove_genealogy()
    pprint.pprint(proof)

    # 3. Introduce Contradicting External Friction to trigger Dynamic Deconstruction
    print("\n[Phase 4] Introduce Opposing Contradicting Raw Friction Shockwave...")
    raw_event_contradict = RawPerturbationImpulse(
        impulse_id="raw_impulse_beta",
        raw_signal="Calm Affirmation & Topological Invariant Counter-Force",
        intensity=2.2,
        frequency_signature=-1.2 * unit_alpha.provenance_trace.refraction_delta_vector
    )

    print("  -> Evaluating contradiction against MemUnit_Alpha_001...")
    is_deconstructed, rewoven_unit, report = deconstruction_engine.evaluate_and_deconstruct(
        target_unit_id="MemUnit_Alpha_001",
        new_raw_friction=raw_event_contradict,
        semantic_engine=engine,
        new_label_if_rewoven="Rewoven_Serenity_Attractor"
    )

    print(f"\n[Phase 5] Deconstruction Report:")
    print(f"  -> Deconstruction Triggered: {is_deconstructed}")
    print(f"  -> Contradiction Tension: {report['contradiction_tension']:.4f}")
    print(f"  -> Target Unit Deconstructed State: {unit_alpha.is_deconstructed}")
    print(f"  -> Target Unit Residual Validity: {unit_alpha.active_validity:.4f}")

    if rewoven_unit is not None:
        print(f"\n[Phase 6] Re-woven Memory Unit Created!")
        print(f"  -> Rewoven Unit ID: {rewoven_unit.unit_id}")
        print(f"  -> Rewoven Label: {rewoven_unit.label}")
        print(f"  -> Ancestry Chain (causal_antecedent_ids): {rewoven_unit.provenance_trace.causal_antecedent_ids}")

        print("\n[Proof of Rewoven Unit Genealogy]:")
        pprint.pprint(rewoven_unit.prove_genealogy())

    print("\n======================================================================")
    print(" [Elysia] Demo Completed Successfully!")
    print("======================================================================")

if __name__ == "__main__":
    main()
