import numpy as np
import time
from synaptic_architecture.causal_observer import CausalObserver

def main():
    print("==================================================================")
    print(" [Elysia Synaptic Core] Universal Causal Backtracking Observation")
    print("==================================================================\n")

    observer = CausalObserver(resolution=256)

    # Pre-seed the field with 'Universal Causes' (Potential Wells)
    print("[System] Seeding Universal Causal Potential Wells...")
    # Seed 1: Atomic Mass Constant
    observer.field.deposit_engram(np.array([50, 50]), np.random.randn(64) * 0.5)
    # Seed 2: Methylation Pattern
    observer.field.deposit_engram(np.array([150, 150]), np.random.randn(64) * 0.5)
    # Seed 3: Biological Life (Consumption Axis)
    observer.field.deposit_engram(np.array([200, 200]), np.random.randn(64) * 0.5)

    # Define the Unified Causal Step
    def unified_cognitive_step(elapsed, params):
        temp = params['temperature']
        print(f"\n--- [T={temp:.2f}] Frequency: {params['frequency']:.1f}Hz ---")

        # In a real run, this would be a single scenario.
        # Here we demonstrate the unified logic across all three.

        # 1. CHEMISTRY
        stim_chem = np.sin(np.linspace(0, 10, 64))
        reac_chem = stim_chem * 0.5 # Mass loss
        deficit_chem = stim_chem - reac_chem
        v_chem = observer.vortex.converge_to_vortex(deficit_chem)
        print(f" [Chemistry] Mass Deficit -> Vortex: {v_chem}")

        # 2. EPIGENETICS
        stim_epi = np.cos(np.linspace(0, 5, 64))
        reac_epi = np.roll(stim_epi, 5) # Stress-induced shift
        deficit_epi = stim_epi - reac_epi
        v_epi = observer.vortex.converge_to_vortex(deficit_epi)
        print(f" [Epigenetics] Phase Shift -> Vortex: {v_epi}")

        # 3. JAJANGMYEON
        stim_jajang = np.ones(64)
        reac_jajang = np.zeros(64) # Total consumption
        deficit_jajang = stim_jajang - reac_jajang
        v_jajang = observer.vortex.converge_to_vortex(deficit_jajang)
        print(f" [Jajangmyeon] Consumption -> Vortex: {v_jajang}")

        # Strengthen the paths (Memristivity)
        for v in [v_chem, v_epi, v_jajang]:
            observer.field.propagate_signal(v, 1.0)

    # Execution Loop
    print("\n[Phase 1] High Temperature Exploration (Sampling Micro-Deficits)")
    observer.scheduler.set_temperature(2.5)
    observer.scheduler.cognitive_loop(duration=1.0, step_func=unified_cognitive_step)

    print("\n[Phase 2] Low Temperature Hardening (Macro-Causal Stabilization)")
    observer.scheduler.set_temperature(0.2)
    observer.scheduler.cognitive_loop(duration=1.0, step_func=unified_cognitive_step)

    print("\n==================================================================")
    print(" [Observation Complete] All scenarios unified by Vortex Potential Field.")
    print("==================================================================")

if __name__ == "__main__":
    main()
