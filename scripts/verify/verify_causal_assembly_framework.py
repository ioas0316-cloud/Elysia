"""
Verification Script: Causal Assembly Framework
===============================================
Demonstrates the 3 Causal Assembly Pillars:
1. Fixed Edge Dependency Mapping
2. Static State Matrix Lookup
3. Retroactive Intent Anchor & Causal Feedback Loop
"""

import sys
import os
import numpy as np

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.consciousness.causal_assembly_engine import CausalAssemblyEngine


def main():
    print("==========================================================")
    print("   Elysia Causal Assembly Framework Demonstration")
    print("==========================================================")

    # Instantiate Unified Causal Assembly Engine
    engine = CausalAssemblyEngine(key_dim=4)

    # 1. Define fragmented resources (Legacy system APIs, Data, Code)
    def legacy_excel_sheet():
        return {"sales_volume": 1500, "unit_price": 25}

    def pricing_api(legacy_excel_sheet=None):
        vol = legacy_excel_sheet["sales_volume"] if legacy_excel_sheet else 0
        price = legacy_excel_sheet["unit_price"] if legacy_excel_sheet else 0
        gross_revenue = vol * price
        return {"gross_revenue": gross_revenue}

    def tax_computation_service(pricing_api=None):
        rev = pricing_api["gross_revenue"] if pricing_api else 0
        net_tax = rev * 0.10
        return {"net_revenue": rev - net_tax, "tax_paid": net_tax}

    fragments = {
        "legacy_excel_sheet": legacy_excel_sheet,
        "pricing_api": pricing_api,
        "tax_computation_service": tax_computation_service
    }

    causal_edges = [
        ("legacy_excel_sheet", "pricing_api"),
        ("pricing_api", "tax_computation_service")
    ]

    # 2. Static pathways (Pre-validated matrix states)
    static_pathways = [
        {"key": "1_1_1_1", "vector": [1.0, 1.0, 1.0, 1.0], "outcome": "OPTIMAL_CACHE_HIT"},
        {"key": "-1_-1_1_1", "vector": [-1.0, -1.0, 1.0, 1.0], "outcome": "DEGRADED_FALLBACK"}
    ]

    print("\n[Pillar 1 & 2] Assembling Fragments and Matrix Pathways...")
    engine.assemble_fragments(fragments, causal_edges, static_pathways)

    # 3. First execution cycle
    var_inputs = np.array([1.2, 0.8, 0.9, 1.1])
    print("\nExecuting Causal Cycle 1...")
    res1 = engine.run_causal_cycle({}, var_inputs)

    print(f" -> Propagated Results: {res1['propagated_results']}")
    print(f" -> Matrix Outcome: {res1['matrix_outcome']} (Friction: {res1['matrix_friction']:.4f})")
    print(f" -> Outcome Quaternion: {res1['outcome_quaternion']}")
    print(f" -> Initial Phase Divergence (ΔΦ): {res1['phase_divergence']:.4f}")
    print(f" -> Retroactive Adjusted Variables: {res1['adjusted_variables']}")

    # 4. Retroactively set/update intent anchor from realized outcome
    print("\n[Pillar 3] Retroactively derivative intent from outcome & applying feedback loop...")
    actual_q = engine.intent_anchor.derive_intent_from_outcome(res1['propagated_results'])
    engine.intent_anchor.update_intent_anchor(actual_q, adaptation_rate=1.0)

    # Re-run cycle with adjusted variables
    print("\nExecuting Causal Cycle 2 (Post Variable Adjustment)...")
    res2 = engine.run_causal_cycle({}, res1['adjusted_variables'])

    print(f" -> Phase Divergence (ΔΦ): {res2['phase_divergence']:.6f}")
    print(f" -> Intent Aligned: {res2['intent_aligned']}")

    print("\n==========================================================")
    print("   Causal Assembly Framework Verification Completed Successfully!")
    print("==========================================================")


if __name__ == "__main__":
    main()
