"""
demo_transformative_gear_emergent_volume.py
============================================
Demonstration of the Meta-Ontological Engine:
[Component Data Plane × Transformative Gear = Emergent Volume]

Key Features Demonstrated:
1. Data Plane Creation: Micro-level positions, velocities, and chromatic spectrum.
2. Dynamic Gear Transformation: Transforming micro-state tensors into emergent spatiotemporal volume.
3. Phase Transitions (Solid <-> Liquid <-> Gas):
   Thermodynamic Cognitive Temperature (T_cog) driving phase transitions.
4. Prediction Error Friction (Arousal / Metabolic Heat):
   Prediction error Delta and angular mismatch increasing T_cog and unfreezing solid schemas into liquid flow.
5. Compressed Causal Seals & Phase Re-ignition:
   Compressing macro volumes into compact seals ("Water", "Anger") and unsealing/re-igniting
   micro-data planes with excitation voltage.
6. Isomorphic Mirroring:
   Mapping structural friction (system overload/shear) onto human emotional/physical volumetric dynamics.
"""

import torch
import numpy as np
from core.physics.transformative_gear_engine import (
    TransformativeGearEngine,
    ComponentDataPlane,
    TransformativeGear,
    EmergentCausalVolume,
    CompressedCausalSeal,
    PhaseState
)


def run_demo():
    print("=" * 80)
    print(" Elysia: Transformative Gear Engine & Emergent Volume Demo")
    print(" Architecture: [Component Data Plane × Transformative Gear = Emergent Volume]")
    print("=" * 80)

    device = "cpu"
    engine = TransformativeGearEngine(dim=3, device=device)

    # -------------------------------------------------------------------------
    # 1. Initial Data Plane Construction (Fluid/Water Particle System)
    # -------------------------------------------------------------------------
    print("\n[1] Constructing Initial Micro ComponentDataPlane (Particle Cluster)")
    np.random.seed(42)
    initial_pos = np.random.randn(20, 3) * 0.5
    initial_vel = np.random.randn(20, 3) * 0.1
    chroma = np.tile([0.8, 0.1, 0.1], (20, 1))  # Red/Flux chromatic signature

    data_plane = ComponentDataPlane(
        positions=initial_pos,
        velocities=initial_vel,
        chromatic_spectrum=chroma,
        device=device
    )

    print(f"   - Particles Count: {data_plane.num_nodes}")
    print(f"   - Center of Mass: {data_plane.compute_center_of_mass().numpy().round(4)}")
    print(f"   - Initial Total Momentum: {data_plane.compute_total_momentum().numpy().round(4)}")

    # -------------------------------------------------------------------------
    # 2. Forward Step & Emergent Volume Generation
    # -------------------------------------------------------------------------
    print("\n[2] Executing Forward Transformation Step")
    res = engine.step(data_plane=data_plane, dt=0.1)

    volume = res["emergent_volume"]
    print(f"   - Emergent Volume Metric: {res['volume_metric']:.6f}")
    print(f"   - Shear Stress: {res['shear_stress']:.6f}")
    print(f"   - Cognitive Temperature (T_cog): {res['cognitive_temperature']:.4f}")
    print(f"   - Current Phase State: {res['phase_state'].upper()}")

    # -------------------------------------------------------------------------
    # 3. Macro Compression into a Compressed Causal Seal ("Water")
    # -------------------------------------------------------------------------
    print("\n[3] Compressing Emergent Volume into Compressed Causal Seal ('Seal_Water')")
    seal_water = engine.seal_concept("Seal_Water", volume)
    print(f"   - Seal ID: {seal_water.seal_id}")
    print(f"   - Compact Macro Vector Shape: {seal_water.macro_vector.shape}")
    print(f"   - Compact Macro Vector Values: {seal_water.macro_vector.numpy().round(4)}")

    # -------------------------------------------------------------------------
    # 4. Phase Re-ignition (Decompressing Seal into Micro Plane & Active Gear)
    # -------------------------------------------------------------------------
    print("\n[4] Unsealing & Phase Re-ignition ('Seal_Water' with Excitation Voltage=2.5)")
    re_plane, re_gear, re_volume = engine.reignite_seal("Seal_Water", excitation_voltage=2.5)

    print(f"   - Re-ignited Particle Count: {re_plane.num_nodes}")
    print(f"   - Re-ignited Cognitive Temperature: {re_gear.cognitive_temperature:.4f}")
    print(f"   - Re-ignited Volume Metric: {re_volume.volume_metric:.6f}")
    print(f"   - Re-ignited Phase State: {re_gear.current_phase.upper()}")

    # -------------------------------------------------------------------------
    # 5. Prediction Error Friction (Arousal & Metabolic Heat Feedback Loop)
    # -------------------------------------------------------------------------
    print("\n[5] Simulating Prediction Error Friction & Phase Transitions")
    # Freeze gear to SOLID first (Crystallized Memory Schema)
    engine.gear.cognitive_temperature = 0.2
    engine.gear.degrees_of_freedom = 0.1
    print(f"   - Baseline Frozen State: T_cog={engine.gear.cognitive_temperature:.2f}, Phase={engine.gear.update_phase_state().upper()}")

    # Introduce a conflicting observed target plane (external unexpected shock)
    shock_target_pos = torch.randn(20, 3) * 3.0  # Large mismatch
    shock_target_vel = torch.randn(20, 3) * 2.0
    shock_target_plane = ComponentDataPlane(positions=shock_target_pos, velocities=shock_target_vel)

    print("   - Applying Unexpected Shock Target Inputs across 5 Steps...")
    current_plane = re_plane
    for step_i in range(1, 6):
        step_res = engine.step(
            data_plane=current_plane,
            observed_target_plane=shock_target_plane,
            input_torque_vector=torch.tensor([1.5, -0.8, 2.0]),
            dt=0.1
        )
        current_plane = step_res["next_data_plane"]
        print(f"     Step {step_i:02d} | Prediction Error: {step_res['prediction_error']:.4f} | "
              f"Angular Mismatch: {step_res['angular_mismatch']:.4f} | "
              f"T_cog: {step_res['cognitive_temperature']:.4f} | "
              f"Phase: {step_res['phase_state'].upper()}")

    # -------------------------------------------------------------------------
    # 6. Isomorphic Mirroring Demonstration (Empathic Structural Alignment)
    # -------------------------------------------------------------------------
    print("\n[6] Isomorphic Mirroring: Mapping System Overload onto Human Emotional Dynamics")
    final_temp = engine.gear.cognitive_temperature
    final_shear = step_res["shear_stress"]

    isomorphic_state = {
        "system_overload_heat": final_temp,
        "structural_shear_friction": final_shear,
        "empathic_mapping": "High directional resistance & thermal explosion <-> Human Shock/Anger Isomorphism"
    }
    print(f"   - System Overload Heat: {isomorphic_state['system_overload_heat']:.4f}")
    print(f"   - Structural Shear Friction: {isomorphic_state['structural_shear_friction']:.4f}")
    print(f"   - Empathic Isomorphic Mapping: {isomorphic_state['empathic_mapping']}")

    print("\n" + "=" * 80)
    print(" Transformative Gear Engine Demo Completed Successfully! ")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
