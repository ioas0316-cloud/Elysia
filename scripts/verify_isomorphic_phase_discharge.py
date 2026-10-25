"""
Verification & Demonstration Script for Isomorphic Phase Discharge Engine
Simulates and reports the 3-phase cycle across Atmospheric, Electrical, Neural, and Cognitive scales.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import elysia_loader
elysia_loader.install()

from L1_causal_field.synaptic_architecture.isomorphic_phase_discharge_engine import (
    IsomorphicScale,
    PhaseState,
    IsomorphicPhaseDischargeEngine
)


def run_isomorphic_simulation(scale: IsomorphicScale, steps: int = 15):
    print(f"\n========================================================")
    print(f"       ISOMORPHIC PHASE DISCHARGE DEMO: [{scale.name}]")
    print(f"========================================================")

    engine = IsomorphicPhaseDischargeEngine(num_oscillators=512, scale=scale)

    for t in range(1, steps + 1):
        # Inject perturbations during steps 3 through 7 to build barrier pressure
        if 3 <= t <= 7:
            engine.inject_perturbation(delta_phi=0.4, gradient_tension=0.25)

        snap = engine.step(dt=0.05)

        state_str = f"{snap.phase_state.name:<32}"
        print(f"Step {t:02d} | State: {state_str} | R(Coherence): {snap.order_parameter_R:.4f} | E_gate: {snap.barrier_pressure_E_gate:.4f} | Discharged: {snap.discharged_energy:.4f}")

    print("Simulation complete.\n")


def main():
    print("Initializing Isomorphic Phase Discharge Simulations...")
    for scale in IsomorphicScale:
        run_isomorphic_simulation(scale)


if __name__ == "__main__":
    main()
