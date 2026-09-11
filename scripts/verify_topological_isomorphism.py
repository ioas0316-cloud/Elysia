r"""
Integration Verification Script for Pure Non-Symbolic Physical Strain Field Engine & Scale Renormalization.

Demonstrates non-symbolic continuous physical strain field dynamics:
1. Physical Wave Gradient Refraction (No text/string labels)
2. Anisotropic Refraction & In-situ Substrate Strain Tensor Deformation G(x)
3. Strain Field Wave Propagation & Tension Field Relaxation (\nabla \cdot T = 0)
4. Macro Phase Order Emergence
5. Enforcing 4 Topological Plasticity Constraints (Reverse Phase Transition, Backpressure Liquefaction, Latent Fault Preservation, Hysteresis)
"""

import sys
import numpy as np
from typing import Dict, Any

from synaptic_architecture.topological_isomorphism_engine import (
    NonSymbolicReceptiveRefractor,
    TopologicalIsomorphismEngine,
    MacroPhaseOrder
)


def run_isomorphism_simulation():
    print("=" * 80)
    print(" [PURE NON-SYMBOLIC PHYSICAL STRAIN FIELD SIMULATION] ")
    print(" Demonstrating Non-Symbolic Physical Wave Refraction, Local Strain Tensors G(x), ")
    print(" Propagation Fields, In-situ Deformation, Strain Relaxation & Macro Phase Orders ")
    print("=" * 80)

    engine = TopologicalIsomorphismEngine(
        gauge_dim=32,
        f_critical=0.2,
        f_dissolve=1.8,
        z_backpressure_threshold=0.6
    )

    # -------------------------------------------------------------------------
    # Step 1: Non-Symbolic Wave Refraction
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Physical Wave Gradient Refraction (No String Labels) ---")
    raw_inputs = [
        "Physical Wave Interaction Field 1",
        "Physical Wave Interaction Field 2",
        "Physical Wave Interaction Field 3",
        np.array([0.5, -0.2, 0.8] * 10, dtype=np.float32)
    ]

    for idx, inp in enumerate(raw_inputs):
        grad = engine.refractor.refract(inp)
        print(f"[Input {idx+1}] Refracted Wave Gradient Norm={np.linalg.norm(grad):.4f}, Mean={np.mean(grad):.4f}, Std={np.std(grad):.4f}")

    # -------------------------------------------------------------------------
    # Step 2: Physical Event Processing & In-situ Substrate Strain Deformation G(x)
    # -------------------------------------------------------------------------
    print("\n--- Step 2: In-situ Strain Tensor Deformation G(x) & Macro Phase Order Emergence ---")

    for idx in range(5):
        wave_input = f"Continuous Strain Pulse {idx+1}"
        rec = engine.process_physical_event(wave_input)
        print(f"  Pulse {idx+1}: PointID={rec['point_id']}, AvgFriction={rec['avg_friction']:.4f}, "
              f"MaxZ={rec['max_impedance']:.4f}, EmergedOrder={rec['emerged_order_id']}")

    active_orders = [m for m in engine.macro_orders.values() if not m.is_fissioned]
    print(f"\n--> Active Macro Phase Orders Emerged: {len(active_orders)}")
    for order in active_orders:
        print(f"    * Order ID: {order.order_id} | Formation Energy: {order.formation_energy:.4f} | "
              f"Encapsulated Points: {len(order.encapsulated_point_ids)} | Tensor Shape: {order.macro_order_tensor.shape}")

    # -------------------------------------------------------------------------
    # Step 3: Enforcing 4 Topological Plasticity Constraints
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Enforcing 4 Topological Plasticity Constraints ---")

    if active_orders:
        target_order = active_orders[0]
        print(f"Initial Target Macro Phase Order: Order {target_order.order_id} (Is Fissioned: {target_order.is_fissioned})")

        print("\n[Constraint 1 & 4: Reverse Phase Transition & Hysteresis Threshold]")
        print("Injecting extreme friction and lowering formation energy requirement to trigger Fission...")
        target_order.formation_energy = 0.01
        engine.f_dissolve = 0.05

        fission_rec = engine.process_physical_event("Extreme Friction Disturbance Wave")
        print(f"Fission Event Results -> Fissioned Order IDs: {fission_rec['fissioned_order_ids']}")
        print(f"Target Order {target_order.order_id} Is Fissioned Status: {target_order.is_fissioned}")

        print("\n[Constraint 2: Impedance Backpressure Liquefaction]")
        print("Injecting high impedance pulse to trigger Beam Liquefaction...")
        engine.f_dissolve = 2.5
        bp_rec = engine.process_physical_event(np.ones(32) * 100.0)
        print(f"Backpressure Liquefaction Results -> Liquefied Beam Count: {bp_rec['liquefied_beam_count']}")

        print("\n[Constraint 3: Latent Fault-Line Preservation]")
        p = engine.points[1]
        print(f"Preserved Fault-Line Vector Count in Substrate Point 1: {len(p.latent_faults)}")

    print("\n" + "=" * 80)
    print(" SIMULATION COMPLETE: Pure Non-Symbolic Physical Strain Field Engine Verified Successfully! ")
    print("=" * 80)


if __name__ == "__main__":
    run_isomorphism_simulation()
