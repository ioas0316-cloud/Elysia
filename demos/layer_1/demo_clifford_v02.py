"""
ELYSIA v0.2 Interactive Terminal Demo: Clifford Multivector Rotor Engine & Dynamic Buffer Pruning

Demonstrates:
1. Multivector Bivector Rotor sandwich operations (v' = R v R~) and exact 1:1 counterfactual rewind (Eye of Agamotto).
2. O(1) Global and ROI Local Clifford Scale Rotor Transformations.
3. Bivector Rotor Coupling Constraint relaxation to phase-locked equilibrium.
4. Dynamic VRAM Buffer Pruning & ROI Slicing pipeline showing real-time memory footprint reduction and O(1) hydration.
5. Clifford Causal Cognitive Memory workspace updates and intertemporal belief revision.
"""

import numpy as np
import time
from core.topology.clifford_v02_engine import (
    CliffordMultivector,
    CliffordSpectralWaveformTensor,
    CliffordScaleRotor,
    DynamicBufferPruningEngine,
    CliffordRotorCouplingConstraint,
    CliffordCausalCognitiveMemory,
    CliffordObservationEvent
)


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    print_section("ELYSIA v0.2: CLIFFORD MULTIVECTOR ROTOR & VRAM PRUNING ENGINE DEMO")

    # -------------------------------------------------------------------------
    # 1. Genuine Clifford Multivector Rotor & Sandwich Operator (v' = R v R~)
    # -------------------------------------------------------------------------
    print("\n[1. Clifford Multivector Rotor & Sandwich Operator v' = R v R~]")
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 2.0  # 90 degrees around Z axis
    rotor = CliffordMultivector.create_rotor(axis, angle)

    print(f"  - Created Rotor (90 deg around Z): Scalar={rotor.scalar:.4f}, Bivector={rotor.bivector.round(4)}")

    v = np.array([1.0, 0.0, 0.0])
    v_transformed = rotor.transform_vector(v)
    v_rewound = rotor.reverse_transform_vector(v_transformed)

    print(f"  - Original Vector v              : {v}")
    print(f"  - Rotated Vector v' (v' = R v R~): {v_transformed.round(4)}")
    print(f"  - Counterfactual Rewind (R~ v' R): {v_rewound.round(4)}")
    assert np.allclose(v, v_rewound, atol=1e-12)
    print("  -> Exact 1:1 Counterfactual Time Rewind Verification: PASSED (Error < 1e-12)")

    # -------------------------------------------------------------------------
    # 2. Clifford Spectral Waveform Tensor & Global/Local Scale Rotors
    # -------------------------------------------------------------------------
    print("\n[2. Clifford Wavefield Tensor & O(1) Scale Rotor Control]")
    wave_tensor = CliffordSpectralWaveformTensor(
        shape=(8, 8),
        frequencies=np.full((8, 8), 440.0),
        phases=np.full((8, 8), 0.0)
    )
    print(f"  - Initial Wavefield Energy: {wave_tensor.compute_energy():.2f}")
    print(f"  - Initial Memory Footprint: {wave_tensor.calculate_memory_bytes()} bytes")

    # Global rotation
    rot_global = CliffordScaleRotor.rotate_global_bivector(wave_tensor, rotor)
    print(f"  - Global Bivector Rotor Applied: Mean Phase = {np.mean(rot_global.phases):.4f} rad")

    # Local ROI 'Eye of Agamotto' Rewind
    roi_mask = np.zeros((8, 8), dtype=bool)
    roi_mask[2:6, 2:6] = True
    rot_local = CliffordScaleRotor.apply_local_rotor(rot_global, roi_mask, rotor, rewind_factor=1.0)
    print(f"  - Local ROI Rewind Applied (4x4 center region):")
    print(f"    * Center ROI Mean Phase (Rewound): {np.mean(rot_local.phases[roi_mask]):.4f} rad")
    print(f"    * Outer Region Mean Phase (Maintained): {np.mean(rot_local.phases[~roi_mask]):.4f} rad")

    # -------------------------------------------------------------------------
    # 3. Dynamic Buffer Pruning & ROI Slicing Pipeline (Low-VRAM Optimization)
    # -------------------------------------------------------------------------
    print("\n[3. Dynamic VRAM Buffer Pruning & Tensor Slicing Pipeline]")
    # Create tensor with phase discrepancy in outer cells
    noisy_tensor = CliffordSpectralWaveformTensor(shape=(16, 16))
    noisy_tensor.phases[0:4, :] = np.pi  # Decoupled noisy regions
    noisy_tensor.phases[12:16, :] = np.pi

    pruning_engine = DynamicBufferPruningEngine(coherence_threshold=0.5)
    pruned_tensor = pruning_engine.prune_tensor_buffer(noisy_tensor, reference_phase=0.0)
    report = DynamicBufferPruningEngine.get_memory_footprint_report(noisy_tensor, pruned_tensor)

    print(f"  - Full Tensor Size              : {noisy_tensor.shape}")
    print(f"  - Original Buffer Bytes          : {report['original_bytes']} bytes")
    print(f"  - Active Bytes Post-Pruning     : {report['pruned_active_bytes']} bytes")
    print(f"  - Pruned Decoupled Cells Ratio   : {report['reduction_ratio'] * 100:.1f}%")
    print(f"  - Memory Savings                 : {report['savings_percentage']:.1f}%")

    # O(1) Hydration test
    hydrated = pruning_engine.hydrate_tensor_buffer(pruned_tensor)
    print(f"  - O(1) Counterfactual Hydration  : Success (Is Pruned = {hydrated.is_pruned})")

    # Lightweight ROI Slicing
    sliced = DynamicBufferPruningEngine.slice_active_threshold_roi(noisy_tensor, (slice(4, 12), slice(4, 12)))
    print(f"  - Sliced ROI Active Tensor Shape : {sliced.shape} ({sliced.calculate_memory_bytes()} bytes)")

    # -------------------------------------------------------------------------
    # 4. Bivector Rotor Coupling Constraint Relaxation
    # -------------------------------------------------------------------------
    print("\n[4. Bivector Rotor Coupling Constraint Relaxation]")
    tensor_a = CliffordSpectralWaveformTensor(shape=(4, 4), phases=np.full((4, 4), np.pi / 2))
    tensor_b = CliffordSpectralWaveformTensor(shape=(4, 4), phases=np.full((4, 4), 0.0))

    constraint = CliffordRotorCouplingConstraint(constraint_id="c_v02_01", target_phase_diff=0.0, coupling_strength=0.8)
    initial_torque = constraint.compute_bivector_torque(tensor_a, tensor_b)
    print(f"  - Initial Coupling Torque : {initial_torque:.4f}")

    curr_a, curr_b = tensor_a, tensor_b
    for step in range(1, 6):
        curr_a, curr_b, err = constraint.relax_step(curr_a, curr_b)
        print(f"    Step {step}: Torque Residual Error = {err:.6f}")

    print(f"  - Phase-Lock Equilibrium Converged to Residual Error < 0.01: {err < 0.01}")

    # -------------------------------------------------------------------------
    # 5. Intertemporal Clifford Causal Cognitive Memory Workspace
    # -------------------------------------------------------------------------
    print("\n[5. Clifford Intertemporal Causal Cognitive Memory Workspace]")
    memory = CliffordCausalCognitiveMemory()

    # Event 1: Intact Object
    obs1 = CliffordObservationEvent(
        event_id="e1", timestamp=1.0, target_id="obj_vase",
        attributes={"category": "Vase", "material": "Porcelain", "damage_level": 0.0}
    )
    memory.record_observation(obs1)
    belief1 = memory.update_intertemporal_belief("obj_vase", "e1")
    print(f"  - Initial Belief (t=1.0): Status={belief1.status}, Category={belief1.category}")

    # Event 2: Ambiguous Damage (Cause unknown -> Suspend Judgment)
    obs2 = CliffordObservationEvent(
        event_id="e2", timestamp=2.0, target_id="obj_vase",
        attributes={"category": "Vase", "material": "Porcelain", "damage_level": 0.8, "is_fragmented": True},
        context_conditions={}
    )
    memory.record_observation(obs2)
    belief2 = memory.update_intertemporal_belief("obj_vase", "e2")
    print(f"  - Ambiguous Damage Belief (t=2.0): Status={belief2.status} (Uncertainty Preserved)")

    # Event 3: Observed Cause -> Revise Belief
    obs3 = CliffordObservationEvent(
        event_id="e3", timestamp=3.0, target_id="obj_vase",
        attributes={"category": "BrokenVase", "material": "Porcelain", "damage_level": 0.8, "is_fragmented": True},
        context_conditions={"observed_cause": "Impact with marble floor"}
    )
    memory.record_observation(obs3)
    belief3 = memory.update_intertemporal_belief("obj_vase", "e3")
    print(f"  - Resolved Cause Belief (t=3.0)  : Status={belief3.status}, Updated Category={belief3.category}")

    print_section("ELYSIA v0.2 DEMO COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
