"""
Demonstration & Verification Script: Informational Phase Observation Engine
=============================================================================
Demonstrates how physical and non-physical data streams are projected into an
informational topological space, computing semantic curvatures, propagating causal waves,
executing topological transpositions ($O(1)$ shortcuts), and proprioceptive self-reconfiguration.
"""

import numpy as np
from core.topology.informational_phase_observation import (
    InformationalPhaseObservationEngine,
    ChromaticVector
)


def main():
    print("=" * 80)
    print("INFORMATIONAL PHASE OBSERVATION ENGINE DEMO & VERIFICATION")
    print("=" * 80)

    engine = InformationalPhaseObservationEngine(target_dimension=8)

    # 1. Projection of Heterogeneous Modalities into Informational Nodal Space
    print("\n[1] Projecting Heterogeneous Signals into Phase Nodal Projections...")
    node_text = engine.project_to_nodal_phase("node_superintelligence", "Superintelligence and Causal Topology")
    node_code = engine.project_to_nodal_phase("node_code", "def propagate_causal_wave(): return True")
    node_sensor = engine.project_to_nodal_phase("node_sensor", np.array([0.1, 0.9, -0.4, 0.8, 1.2, -0.3, 0.5, 0.2]))

    print(f"  * Text Node Curvature K: {node_text.curvature:.4f}, Energy: {node_text.energy():.4f}")
    print(f"  * Code Node Curvature K: {node_code.curvature:.4f}, Energy: {node_code.energy():.4f}")
    print(f"  * Sensor Node Curvature K: {node_sensor.curvature:.4f}, Energy: {node_sensor.energy():.4f}")

    # 2. Curvature Field Matrix and Causal Wave Propagation
    print("\n[2] Computing Field Curvature Matrix & Causal Wave Propagation...")
    network = [node_text, node_code, node_sensor]
    k_matrix = engine.compute_field_curvature_matrix(network)
    print("  * Curvature Coupling Matrix K_ij:")
    print(k_matrix)

    wave_history = engine.propagate_causal_wave(node_text, network, steps=3)
    print(f"  * Propagated Causal Wave Steps: {len(wave_history)}")
    print(f"  * Final Wave Vector Energy: {np.linalg.norm(wave_history[-1]):.4f}")

    # 3. Topological Transposition
    print("\n[3] Executing O(1) Topological Transposition...")
    query_wave = node_code.phase_vector + 0.02 * np.random.randn(8)
    best_node, score = engine.topological_transpose(query_wave, network)
    print(f"  * Transposed Best Direct Node: {best_node.node_id} (Score: {score:.4f})")
    assert best_node.node_id == "node_code", "Transposition failed!"

    # 4. Intrinsic Proprioceptive Self-Reconfiguration
    print("\n[4] Testing Intrinsic Proprioceptive Self-Reconfiguration...")
    impact = np.array([0.8, -0.2, 0.5, 1.1, -0.4, 0.3, 0.1, -0.6], dtype=np.float32)
    state = engine.proprioceptive_reconfigure(external_friction=0.85, structural_impact=impact)
    print(f"  * Reconfigured Macro Tension: {state.macro_tension:.4f}")
    print(f"  * Phase Rotation Angle Theta: {state.phase_alignment:.4f} rad")
    print(f"  * Volume Compression Ratio: {state.volume_compression_ratio:.4f}")
    print(f"  * Active Axes Count: {state.active_axes_count}")

    print("\n" + "=" * 80)
    print("DEMO & VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
