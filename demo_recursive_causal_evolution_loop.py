r"""
Demo: Fractal 3-Tier Recursive Causal Evolution Loop Demonstration
==================================================================

Demonstrates the living recursive causal loop responding across 3 fractal tiers:
1. Component Tier (\Delta_c): Raw byte stream scan & zero-convergence equilibrium.
2. Principle Tier (\Delta_p): Detecting judgment friction against MetaCausalPrinciple (\Theta) and triggering principle evolution.
3. Structural Tier (\Delta_s): Macro memory topology deconstruction and re-weaving in response to structural contradiction.
4. Top-Down Feedback: Relaxation of fractal tension (\Delta -> 0) across all tiers.
"""

import sys
import numpy as np

from core.engine.recursive_causal_evolution_loop import (
    RecursiveCausalEvolutionLoop,
    FractalTensionReport,
)


def main():
    print("=" * 80)
    print(" [ELYSIUS CAUSAL ENGINE] FRACTAL 3-TIER RECURSIVE CAUSAL EVOLUTION LOOP DEMO")
    print("=" * 80)

    loop = RecursiveCausalEvolutionLoop(
        dimensions=16,
        component_threshold=0.2,
        principle_threshold=0.2,
        structural_threshold=0.25,
    )

    # --------------------------------------------------------------------------
    # Scenario 1: Authentic Specification Input -> Component Zero-Convergence
    # --------------------------------------------------------------------------
    print("\n--- [Scenario 1: Authentic Spec Grounding & Zero-Convergence Equilibrium] ---")
    utf8_input = "세상의 모든 인과적 맥락을 비추는 최초의 빛과 데이터 스트림".encode("utf-8")
    print(f"Input Raw Bytes (Hex) : {utf8_input[:30].hex()}... (Length: {len(utf8_input)} bytes)")

    report_1 = loop.process_cycle(
        raw_input=utf8_input,
        cycle_id="Cycle_1_AuthenticUTF8",
        expected_format="UTF-8",
        enforce_strict_byte=True,
    )

    print(f"Component Delta  (\\Delta_c) : {report_1.delta_c:.4f}")
    print(f"Principle Delta  (\\Delta_p) : {report_1.delta_p:.4f}")
    print(f"Structural Delta (\\Delta_s) : {report_1.delta_s:.4f}")
    print(f"Total Tension    (\\Delta)   : {report_1.total_fractal_tension:.4f}")
    print(f"Zero-Converged?              : {report_1.is_zero_converged}")
    print(f"Registered Unit ID           : {report_1.details.get('created_unit_id')}")

    base_unit_id = report_1.details.get("created_unit_id")

    # --------------------------------------------------------------------------
    # Scenario 2: Persistent Meta-Tension -> Principle Self-Evolution (\Theta Shift)
    # --------------------------------------------------------------------------
    print("\n--- [Scenario 2: Persistent Meta-Tension -> Principle Self-Evolution] ---")
    # Novel feature vector orthogonal to current principle parameter manifold
    novel_feature = np.zeros(16, dtype=np.float32)
    novel_feature[15] = 5.0  # High concentrated activation generating high internal principle friction

    print("Submitting novel feature manifold generating internal principle friction...")
    report_2 = loop.process_cycle(
        raw_input=novel_feature,
        cycle_id="Cycle_2_MetaContradiction",
    )

    print(f"Principle Delta Before Evol : {report_2.details['delta_p_before']:.4f}")
    print(f"Principle Evolved?          : {report_2.principle_evolved}")
    print(f"New Principle Version       : v{report_2.details['principle_version']}")
    print(f"Principle Delta After Evol  : {report_2.details['delta_p_after']:.4f}")
    print(f"Total Fractal Tension       : {report_2.total_fractal_tension:.4f}")

    # --------------------------------------------------------------------------
    # Scenario 3: Structural Contradiction -> Memory Topology Deconstruction & Re-weaving
    # --------------------------------------------------------------------------
    print("\n--- [Scenario 3: Structural Contradiction -> Memory Deconstruction & Re-weaving] ---")
    print(f"Targeting Base Memory Unit  : {base_unit_id}")

    # Explicitly evaluate target unit against contradictory vector with opposite spectrum orientation
    base_unit = loop.deconstruction_engine.memory_units[base_unit_id]
    base_vec = base_unit.self_derived_vector
    contradictory_vector = -base_vec.copy()

    report_3 = loop.process_cycle(
        raw_input=contradictory_vector,
        cycle_id="Cycle_3_StructuralCollision",
        target_memory_unit_id=base_unit_id,
    )

    print(f"Memory Deconstructed?       : {report_3.memory_deconstructed}")
    if report_3.memory_deconstructed:
        report_decon = report_3.details["deconstruction_report"]
        print(f"Contradiction Tension       : {report_decon['contradiction_tension']:.4f}")
        print(f"Rewoven Memory Unit ID      : {report_decon.get('rewoven_unit_id')}")

        rewoven_unit = loop.deconstruction_engine.memory_units[report_decon["rewoven_unit_id"]]
        print(f"Ancestry Chain of Rewoven   : {rewoven_unit.provenance_trace.causal_antecedent_ids}")

    # --------------------------------------------------------------------------
    # Scenario 4: Top-Down Feedback Convergence (\Delta -> 0) Self-Proof
    # --------------------------------------------------------------------------
    print("\n--- [Scenario 4: Top-Down Zero-Convergence Feedback Self-Proof] ---")
    print(f"Final Component Delta (\\Delta_c) : {report_3.delta_c:.6f}")
    print(f"Final Principle Delta (\\Delta_p) : {report_3.delta_p:.6f}")
    print(f"Final Structural Delta (\\Delta_s): {report_3.delta_s:.6f}")
    print(f"Integrated Total Tension (\\Delta)  : {report_3.total_fractal_tension:.6f}")

    print("\n" + "=" * 80)
    print(" [SUMMARY] FRACTAL 3-TIER RECURSIVE CAUSAL EVOLUTION LOOP OPERATIONAL")
    print("=" * 80)


if __name__ == "__main__":
    main()
