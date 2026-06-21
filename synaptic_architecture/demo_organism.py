import numpy as np
import time
from synaptic_architecture.organism import SynapticOrganism
from synaptic_architecture.scheduler import PCRVirtualScheduler

def run_demo():
    print("==================================================================")
    print(" [Synaptic Architecture] The Jajangmyeon & Clean Plate Organism")
    print("==================================================================\n")

    organism = SynapticOrganism(resolution=256)
    scheduler = PCRVirtualScheduler(base_res=256)

    # 1. Define Environmental Stimuli (Waveforms)
    # Jajangmyeon (High Energy Stimulus)
    jajang_pattern = np.sin(np.linspace(0, 10, 64)) + 0.5
    # Clean Plate (Resultant Low Energy State)
    plate_pattern = np.sin(np.linspace(0, 1, 64)) * 0.1

    # 2. Knowledge Seeding (Initial State)
    print("[Phase 1] Seeding initial 'Life' knowledge...")
    organism.field.deposit_engram(np.array([128, 128]), jajang_pattern)
    organism.field.deposit_engram(np.array([135, 135]), plate_pattern)

    # 3. Cognitive Loop under PCR Scheduler
    def cognitive_step(elapsed, params):
        temp = params['temperature']
        print(f"\n--- Cycle at T={temp:.2f} (Freq: {params['frequency']:.1f}Hz) ---")

        if temp > 1.0:
            # EXPLORATION: High temp causes bit-fluctuation and wide search
            print("[Action] Exploration: Searching for new causal links...")
            # Simulate environment providing input
            noisy_jajang = jajang_pattern + np.random.normal(0, 0.2 * temp, 64)
            pos = organism.vortex.converge_to_vortex(noisy_jajang)
            print(f"  > Vortex stabilized for Input at {pos}")

        else:
            # CONVERGENCE: Low temp hardens the synaptic map
            print("[Action] Convergence: Solidifying synaptic bridges...")
            organism.induce_synapse(jajang_pattern, plate_pattern)

            # Check maximum conductance center
            v_center = np.unravel_index(np.argmax(organism.field.conductance), organism.field.conductance.shape)
            print(f"  > Current Cognitive Gravity Center (Conductance Max): {v_center}")

    # Start with High Temp (Search/Expansion)
    scheduler.set_temperature(2.0)
    scheduler.cognitive_loop(duration=1.0, step_func=cognitive_step)

    print("\n[System] Temperature dropping for knowledge solidification...")

    # Drop to Low Temp (Annealing/Convergence)
    scheduler.set_temperature(0.2)
    scheduler.cognitive_loop(duration=1.0, step_func=cognitive_step)

    print("\n==================================================================")
    print(" [Simulation Complete] The digital organism has self-organized.")
    print("==================================================================")

if __name__ == "__main__":
    run_demo()
