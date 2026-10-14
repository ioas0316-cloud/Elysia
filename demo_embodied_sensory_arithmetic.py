"""
[Demo Script: Embodied Sensory Arithmetic & Retrocausal Tracing]
(체득형 감각 수량 인지 및 역방향 인과 역추적 시연 데모)

Demonstrates the 4 continuous stages of embodied arithmetic cognition:
1. Stage 1: Sensory Unit Grounding (Apple 1 & Apple 2 continuous waveform sensing & rectification)
2. Stage 2: Physical Transformation & Sequence (Phase fusion/addition & chronological sequence logging)
3. Stage 3: Symbolic Mapping & Abstraction (Invariant topological pattern & harmony extraction)
4. Stage 4: Higher Causality & Retrocausal Reversible Circuit (100% back-tracing with Delta S ≈ 0)
"""

import numpy as np
from core.sensory.embodied_sensory_arithmetic import EmbodiedSensoryArithmeticEngine


def run_embodied_sensory_arithmetic_demo():
    print("==========================================================================")
    print("   ELYASIA COGNITIVE ENGINE: EMBODIED SENSORY ARITHMETIC DEMO            ")
    print("==========================================================================\n")

    engine = EmbodiedSensoryArithmeticEngine(noise_threshold=0.1, resolution=128)

    # ----------------------------------------------------
    # STAGE 1: Sensory Unit Grounding (Apple 1 & Apple 2)
    # ----------------------------------------------------
    print("[STAGE 1] Sensory Unit Grounding (Sensory Perception of Apples)")
    print("Simulating continuous sensory waveforms (Apple 1 & Apple 2)...")

    t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
    raw_apple_1 = np.sin(2 * t) + 0.15 * np.random.randn(128).astype(np.float32)
    raw_apple_2 = np.sin(2 * t + 0.2) + 0.12 * np.random.randn(128).astype(np.float32)

    apple_1 = engine.register_sensory_unit("apple_1", raw_apple_1, amplitude=1.0)
    apple_2 = engine.register_sensory_unit("apple_2", raw_apple_2, amplitude=1.0)

    print(f"  - Apple 1 registered: Rectified Wave Mean = {np.mean(apple_1.spectrum_waveform):.4f}")
    print(f"    Chromatic Vector [Flux, Order, Entropy] = {apple_1.chromatic_vector.round(3).tolist()}")
    print(f"  - Apple 2 registered: Rectified Wave Mean = {np.mean(apple_2.spectrum_waveform):.4f}")
    print(f"    Chromatic Vector [Flux, Order, Entropy] = {apple_2.chromatic_vector.round(3).tolist()}")
    print(f"  - Current Emergent Field Energy: {np.sum(np.square(engine.emergent_field_state)):.4f}\n")

    # ----------------------------------------------------
    # STAGE 2: Physical Transformation & Sequence (1 + 1)
    # ----------------------------------------------------
    print("[STAGE 2] Physical Transformation & Sequence Logging (1 Apple + 1 Apple)")
    print("Applying physical phase superposition/fusion (Addition operation)...")

    seq_step = engine.execute_physical_transformation(
        operation="ADDITION",
        unit_ids=["apple_1", "apple_2"],
        timestamp=1.0
    )

    print(f"  - Causal Sequence Step #{seq_step.step_index}: Operation = {seq_step.operation_type}")
    print(f"  - Inputs: {seq_step.input_unit_ids}")
    print(f"  - Phase Shift Induced: {seq_step.phase_shift:.4f} rad")
    print(f"  - Energy Transferred: {seq_step.energy_transferred:.4f}")
    print(f"  - Sequence History Depth: {len(engine.sequence_history)}\n")

    # ----------------------------------------------------
    # STAGE 3: Symbolic Mapping & Abstraction
    # ----------------------------------------------------
    print("[STAGE 3] Symbolic Mapping & Invariant Topological Pattern Abstraction")
    print("Extracting invariant topological signature and abstract quantity...")

    abstract_pattern = engine.extract_symbolic_mapping()

    print(f"  - Abstract Pattern ID: {abstract_pattern.pattern_id}")
    print(f"  - Perceived Physical Quantity: {abstract_pattern.perceived_quantity:.4f} (Approx 2.0 Units)")
    print(f"  - Symbolic Mapping: '{abstract_pattern.symbolic_representation}'")
    print(f"  - Harmony Coherence Score: {abstract_pattern.harmony_coherence:.4f}\n")

    # ----------------------------------------------------
    # STAGE 4: Higher Causality & Retrocausal Tracing
    # ----------------------------------------------------
    print("[STAGE 4] Higher Causality & Retrocausal Reversible Circuit")
    print("Back-tracing from emergent state back to initial sensory units...")

    retro_result = engine.trace_retrocausal_trajectory()

    print(f"  - Emergent Quantity Measured: {retro_result.emergent_quantity:.4f}")
    print(f"  - Reconstructed Units Count: {len(retro_result.reconstructed_initial_units)}")
    print(f"  - Reconstructed Sequence Steps: {len(retro_result.reconstructed_sequence)}")
    print(f"  - Reconstruction Fidelity: {retro_result.reconstruction_fidelity * 100:.2f}%")
    print(f"  - Reversibility Dissipation Index (Delta S): {retro_result.reversibility_dissipation_index:.6f} (≈ 0)")

    print("\n==========================================================================")
    print("   EMBODIED SENSORY ARITHMETIC DEMO COMPLETE: SUCCESS                     ")
    print("==========================================================================")


if __name__ == "__main__":
    run_embodied_sensory_arithmetic_demo()
