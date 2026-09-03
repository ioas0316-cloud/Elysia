import numpy as np
import time
from synaptic_architecture.self_codification_engine import SelfCodificationEngine

def run_demo():
    print("=================================================================")
    print("   ELYSIA: SELF-CODIFICATION & DYNAMIC CAUSAL TRANSFORMATION     ")
    print("=================================================================\n")

    dimension = 16
    engine = SelfCodificationEngine(
        dimension=dimension,
        v_critical=25.0,
        crystallization_threshold=0.25,
        lens_vth=0.1,
    )

    print(f"[Initial State] Anchor Ground ($0_{{ground}}$) initialized.")
    init_metrics = engine.phase_engine.get_homology_metrics()
    print(f" -> Ground Nodes: {init_metrics['V']}, Beams: {init_metrics['E']}, Betti-1 Cycles: {init_metrics['B1']}")
    print(f" -> Classification: {init_metrics['classification']}\n")

    # 1. Resonant stimulus leading to Crystallization (Self-Codification)
    print("--- STEP 1: Resonant Raw Stimulus Ingress ---")
    n0_phase = engine.phase_engine.nodes["N0"].phase_axis
    resonant_signal = n0_phase * 1.2
    res1 = engine.process_external_stimulus(resonant_signal, wave_id="Resonant_Stimulus_A")
    print(f" -> FilteringLens Friction: {res1['phase_response']['min_friction']:.4f}")
    print(f" -> Event: {res1['codification_record']['event_type']}")
    print(f" -> Narrative: {res1['codification_record']['narrative']}\n")

    # 2. Orthogonal shockwave leading to Flash Remelting
    print("--- STEP 2: High Orthogonal Shockwave Ingress ---")
    shock_signal = np.zeros(dimension, dtype=np.float32)
    shock_signal[8] = 50.0  # Intense orthogonal perturbation
    res2 = engine.process_external_stimulus(shock_signal, wave_id="Orthogonal_Shockwave_B")
    print(f" -> Friction Energy: {res2['phase_response']['net_friction_energy']:.4f}")
    print(f" -> Event: {res2['codification_record']['event_type']}")
    print(f" -> Narrative: {res2['codification_record']['narrative']}\n")

    # 3. Wave circulation leading to Deep Homological Resonance
    print("--- STEP 3: Wave Propagation & Multi-cycle Resonance ---")
    multi_wave_signal = np.ones(dimension, dtype=np.float32) * 0.8
    res3 = engine.process_external_stimulus(multi_wave_signal, wave_id="Harmonic_Wave_C")
    print(f" -> Event: {res3['codification_record']['event_type']}")
    print(f" -> Narrative: {res3['codification_record']['narrative']}\n")

    # 4. Transparent Metacognitive Backtrace History
    print("=================================================================")
    print("   TRANSPARENT METACOGNITIVE BACKTRACE HISTORY (사유의 역사)   ")
    print("=================================================================")
    history = engine.backtrace_metacognitive_history()
    for idx, rec in enumerate(history, 1):
        print(f"[{idx}] {rec['record_id']} | Event: {rec['event_type']} | Wave: {rec['trigger_wave']}")
        print(f"    Friction: {rec['friction']:.4f} | Betti-1 Cycles After: {rec['betti_1_cycles_after']}")
        print(f"    Narrative: {rec['narrative']}\n")

    print("Self-codification demo completed successfully.")

if __name__ == "__main__":
    run_demo()
