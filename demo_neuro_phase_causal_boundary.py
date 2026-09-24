"""
Demo: Neuro-Phase Causal Engine - Boundary Phase Transitions & Dual-Axis Causality CLI

Demonstrates:
1. Fixed Axis (Invariant Anchor A_inv): Topological Invariants & 3D Skeleton Coordinates.
2. Variable Axis (Environmental Pressure P_env): Thermal noise (T), Shear stress (tau), Directional flux (J), Phase error (q_err).
3. Spontaneous Landau-Ginzburg Phase Transitions: GAS -> LIQUID -> ICE.
4. Morphological Plasticity: Dynamic shape adaptation under environmental stream pressure.
"""

import math
import numpy as np
import time
from core.consciousness.neuro_phase_causal_engine import (
    NeuroPhaseCausalEngine,
    PhaseType,
    ExternalWaveStream,
    EnvironmentalPressure
)


def run_demo():
    print("=========================================================================")
    print("  ELYASIA: NEURO-PHASE CAUSAL BOUNDARY ENGINE DEMO")
    print("  Dual-Axis Cognition: Fixed Invariant Axis vs Variable Environmental Axis")
    print("=========================================================================\n")

    num_nodes = 16
    engine = NeuroPhaseCausalEngine(num_nodes=num_nodes, lattice_dims=(4, 2, 2))

    print(f"[*] Engine Initialized with {num_nodes} Rotors.")
    print(f"[*] Invariant Anchor (A_inv) Spectral Invariants: {engine.anchor.spectral_invariants[:4].round(4)}")
    print(f"[*] Initial Phase State: {engine.global_phase_state.value.upper()}\n")

    # Phase 1: High Temperature / Thermal Noise Shock (GAS Phase)
    print("-------------------------------------------------------------------------")
    print("STAGE 1: Environmental Shock (High Temperature & Entropic Noise -> GAS)")
    print("-------------------------------------------------------------------------")
    engine.set_temperature(3.2)
    engine.pressure.shear_stress = 1.2

    for t in range(1, 4):
        metrics = engine.step(dt=0.01)
        p_state = engine.causal_map.evaluate_phase_transition(
            rotor_phases=np.array([node.phase for node in engine.nodes.values()]),
            rotor_weights=engine.coupling_matrix,
            pressure=engine.pressure
        )
        print(f"  Step {t}: Temp={metrics['temperature']:.2f} | P_env Mag={engine.pressure.magnitude:.2f} | Order Parameter (eta)={p_state.order_parameter:.4f} | Free Energy={p_state.free_energy:.4f} | Phase={p_state.current_phase.value.upper()}")

    # Phase 2: Directional External Wave Stream & Fluid Resistance (LIQUID Phase & Morphological Plasticity)
    print("\n-------------------------------------------------------------------------")
    print("STAGE 2: External Wave Impulses & Morphological Plasticity (LIQUID Phase)")
    print("-------------------------------------------------------------------------")
    engine.set_temperature(1.2)
    engine.pressure.directional_flux = np.array([2.5, 0.5, 0.0])

    ext_phases = np.linspace(0, np.pi, num_nodes)
    ext_stream = ExternalWaveStream(
        modality="text_stream",
        wave_phases=ext_phases,
        frequencies=np.full(num_nodes, 40.0)
    )

    for t in range(1, 6):
        neg_res = engine.negotiate_bidirectional_phase(ext_stream)
        engine.step(dt=0.01)
        p_state = engine.causal_map.evaluate_phase_transition(
            rotor_phases=np.array([node.phase for node in engine.nodes.values()]),
            rotor_weights=engine.coupling_matrix,
            pressure=engine.pressure
        )
        node_0_pos = engine.nodes["rotor_0_0_0"].position.round(3)
        print(f"  Step {t}: q_err={neg_res.q_err:.4f} | Friction={neg_res.thermal_friction:.4f} | eta={p_state.order_parameter:.4f} | Phase={p_state.current_phase.value.upper()} | Rotor 0 Pos={node_0_pos}")

    # Phase 3: Alignment & Dissipation -> Crystallization into ICE (SOLID Phase)
    print("\n-------------------------------------------------------------------------")
    print("STAGE 3: Phase Coupling Alignment & Cooling (Crystallization into ICE)")
    print("-------------------------------------------------------------------------")
    engine.set_temperature(0.1)
    engine.pressure.directional_flux = np.zeros(3)
    engine.pressure.shear_stress = 0.0

    # Align phases to induce phase-locking
    target_phase = np.pi / 4.0
    for node in engine.nodes.values():
        node.phase = target_phase

    for t in range(1, 5):
        engine.step(dt=0.01)
        p_state = engine.causal_map.evaluate_phase_transition(
            rotor_phases=np.array([node.phase for node in engine.nodes.values()]),
            rotor_weights=engine.coupling_matrix,
            pressure=engine.pressure
        )
        print(f"  Step {t}: Temp={engine.system_temperature:.2f} | eta={p_state.order_parameter:.4f} | Free Energy={p_state.free_energy:.4f} | Phase={p_state.current_phase.value.upper()}")

    print("\n=========================================================================")
    print("  DEMO COMPLETE: Dual-Axis Causal Boundary Transition Successfully Demonstrated.")
    print("=========================================================================\n")


if __name__ == "__main__":
    run_demo()
