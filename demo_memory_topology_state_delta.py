"""
Demonstration: State-Delta Knowledge Graph & Fluid Phase Transition Engine
===========================================================================

Demonstrates:
1. Decoupled Execution with State-Delta Knowledge Graph recording What, Where, How, Why metadata.
2. Inverse Causal Tracing from target node back to origin.
3. Bit Modulation & Dynamic Resolution Demodulation based on observer distance.
4. Continuous Fluid Phase Dynamics <-> Crystallization (hat{C}) <-> Lattice Phase <-> Melting Phase (hat{M}).
"""

import numpy as np
from core.topology.memory_topology_isomorphism import (
    StateDeltaKnowledgeGraph,
    BitModulatorDemodulator,
    FluidPhaseTransitionEngine,
)


def sample_quaternion_rotation(state: np.ndarray, angle: float, axis: np.ndarray) -> np.ndarray:
    """
    Decoupled tool operator: applies 3D/4D rotation vector transformation on 6D state.
    """
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    # Rotation transformation on state vector
    res = np.copy(state)
    res[:3] = state[:3] * cos_a + np.cross(axis[:3], state[:3]) * sin_a
    return res


def main():
    print("=== Demo 1: Decoupled Execution & State-Delta Knowledge Graph ===")
    graph = StateDeltaKnowledgeGraph()
    initial_state = np.array([1.0, 0.0, 0.0, 0.5, -0.2, 0.8])
    graph.add_node("node_t0", initial_state)

    # Apply operator and record delta edge
    axis = np.array([0.0, 0.0, 1.0])
    target_state = graph.apply_operator_and_record_edge(
        source_id="node_t0",
        target_id="node_t1",
        operator_fn=sample_quaternion_rotation,
        operator_id="quaternion_rotate_z",
        operator_params={"angle": np.pi / 2, "axis": axis},
        memory_offset=0x1004,
        intent_id="intent_align_orientation"
    )

    print(f"Source State (t0): {initial_state}")
    print(f"Target State (t1): {target_state}")

    # Inverse causal reasoning trace
    edges = graph.inverse_trace("node_t1")
    print("\n--- Inverse Causal Trace for node_t1 ---")
    for edge in edges:
        print(f"  Source Node: {edge.source_node_id}")
        print(f"  What Delta (dS): {edge.what_delta}")
        print(f"  Where Memory Offset: 0x{edge.where_memory_offset:X}")
        print(f"  How Operator ID: {edge.how_operator_id}")
        print(f"  Why Intent ID: {edge.why_intent_id}")

    print("\n=== Demo 2: Bit Modulation & Dynamic Resolution Demodulation ===")
    modulator = BitModulatorDemodulator(buffer_size=2048)
    tensor_data = np.array([[1.5, -2.3, 3.1], [4.0, 0.0, -1.2]], dtype=np.float32)
    bytes_written = modulator.modulate(tensor_data, offset=0)
    print(f"Modulated Tensor Data into {bytes_written} primitive bytes.")

    # Demodulation from close observer (high resolution)
    obs_close = np.array([0.0, 0.0, 0.0])
    target_pos = np.array([1.0, 0.0, 0.0])
    demod_close = modulator.demodulate(offset=0, shape=(2, 3), observer_position=obs_close, target_position=target_pos)
    print("Close Demodulation (Full Precision):\n", demod_close)

    # Demodulation from distant observer (compressed potential)
    obs_far = np.array([50.0, 50.0, 0.0])
    demod_far = modulator.demodulate(offset=0, shape=(2, 3), observer_position=obs_far, target_position=target_pos)
    print("Distant Demodulation (Compressed Potential):\n", demod_far)

    print("\n=== Demo 3: Fluid Information Dynamics & 4-Phase Transition Pipeline ===")
    fluid_engine = FluidPhaseTransitionEngine(grid_size=8)
    print(f"Initial Phase: {fluid_engine.phase}")

    # Step fluid dynamics
    for _ in range(5):
        fluid_engine.step_fluid_dynamics(source_s=0.2)
    print(f"Fluid Density Field Mean: {np.mean(fluid_engine.rho):.4f}")

    # Crystallization hat{C}
    fluid_engine.crystallize()
    print(f"Post-Crystallization Phase: {fluid_engine.phase}")
    print(f"Crystallized Lattice Nodes Count: {len(fluid_engine.lattice_nodes)}")
    print(f"Crystallized Lattice Edges Count: {len(fluid_engine.lattice_edges)}")

    # Melting hat{M}
    fluid_engine.melt(thermal_energy_delta_e=1.5)
    print(f"Post-Melting Phase: {fluid_engine.phase}")
    print(f"Melted Fluid Density Field Mean: {np.mean(fluid_engine.rho):.4f}")

    print("\nState-Delta Memory Topology Demo Completed Successfully!")


if __name__ == "__main__":
    main()
