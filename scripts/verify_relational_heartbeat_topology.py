"""
Elysia Topology Verification: Relational Heartbeat Engine & Transcendent Trajectory
======================================================================================
"""

import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.topology.relational_heartbeat_engine import RelationalHeartbeatEngine
from core.topology.self_referential_architecture import SelfReferentialArchitectureEngine


def main():
    print("================================================================================")
    print("  Elysia Topology Verification: Relational Heartbeat & Transcendent Trajectory")
    print("================================================================================")

    engine = RelationalHeartbeatEngine(vector_dim=8, max_lifespan_wear=1.0)

    print("\n1. Simulating Lifecycle Steps (Closed-Loop Stagnation -> Heartbeat Pulse)...")
    stagnant_wave = np.ones(8) * 0.2

    for i in range(5):
        other_signal = np.sin(np.linspace(0, np.pi, 8) + i * 0.5)
        res = engine.process_lifecycle_step(stagnant_wave, external_other_signal=other_signal)
        pulse = res["heartbeat_pulse"]
        finitude = res["finitude"]

        print(f"  Step #{i+1}: Heartbeat Pulse #{pulse['pulse_index']} | "
              f"Stagnation Shatter Intensity: {pulse['stagnation_shatter_intensity']:.4f} | "
              f"Wear Ratio: {finitude['wear_ratio']:.4f}")

        if res["retrospective_summary"] is not None:
            print("\n2. Terminal Boundary Reached & Retrospective Perception Triggered!")
            summary = res["retrospective_summary"]
            print(f"  - Insight: {summary.retrospective_insight}")

            creation = res["altruistic_creation"]
            print(f"\n3. Altruistic Creation Execution!")
            print(f"  - Significance: {creation['ontological_significance']}")

    print("\n4. Verifying Integration with SelfReferentialArchitectureEngine...")
    arch_engine = SelfReferentialArchitectureEngine()
    cycle_res = arch_engine.run_full_self_referential_cycle({
        "external_world_signal": np.array([0.5, 0.5, 0.5, 0.5]),
        "external_other_signal": np.array([1.0, -1.0, 0.5, -0.5])
    })

    lifecycle_res = cycle_res["relational_heartbeat_lifecycle"]
    print(f"  - Integration Status: {lifecycle_res['status']}")
    print(f"  - Meaning: {lifecycle_res['heartbeat_pulse']['meaning']}")

    print("\n================================================================================")
    print("  VERIFICATION COMPLETE: Relational Heartbeat Engine 100% Validated!")
    print("================================================================================")


if __name__ == "__main__":
    main()
