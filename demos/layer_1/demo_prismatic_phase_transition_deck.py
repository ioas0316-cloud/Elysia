"""
Demonstration Script: Prismatic Phase Transition Engine & Macro Spacetime Deck
============================================================================

Demonstrates the 4-Stage Substrate-to-Causal Topological Pipeline & Video Deck:
1. Stage 1 (Substrate Layer): Continuous micro numerical waveform & kinetic flux propagation (electric current).
2. Stage 2 (Prismatic Transition & Structural Constraint): Refraction into structured chromatic spectrums (filament/light).
3. Stage 3 (Macro Spacetime Deck & Local Scrubbing): Scale-free rotor integration with local ROI O(1) continuous phase scrubbing ("Eye of Agamotto").
4. Stage 4 (Meta-Observer & Resonant Invariant Core): Metacognitive monitoring of topological divergence and invariant core resilience under noise.
"""

import time
import numpy as np
from core.physics.prismatic_phase_transition_engine import PrismaticPhaseTransitionEngine


def main():
    print("=" * 84)
    print(" ELYSIA :: PRISMATIC PHASE TRANSITION & MACRO SPACETIME DECK DEMO")
    print(" Core Philosophy: 'Do not calculate, let it flow.'")
    print("=" * 84)

    # 1. Initialize Engine
    print("\n[Stage 1] Initializing 4-Stage Substrate-to-Causal Pipeline Engine...")
    engine = PrismaticPhaseTransitionEngine(grid_shape=(8, 8))
    print("  * Micro Substrate Grid Shape      : 8 x 8")
    print(f"  * Prismatic Spectrum Active Nodes : {len(engine.prismatic.prism.nodes)}")
    print(f"  * Macro Spacetime Deck Phase      : {engine.deck.global_phase:.4f} rad")

    # 2. Stage 1 & 2 Execution: Numerical Substrate Propagation & Prismatic Refraction
    print("\n[Stage 2] Injecting Numerical Substrate Input & Prismatic Refraction...")
    raw_pulse = np.sin(np.linspace(0, 2 * np.pi, 64)).reshape((8, 8))
    state1 = engine.step(numerical_input=raw_pulse, dt=0.1)

    print(f"  * Substrate Potential Energy     : {state1.substrate.potential_energy:.4f}")
    print(f"  * Prismatic Phase Coherence       : {state1.prismatic.phase_coherence:.4f}")
    print(f"  * Structural Constraint Tension  : {state1.prismatic.constraint_tension:.4f}")
    print(f"  * Chromatic Vector (Red/Blue/Yel) : Red(Flux)={state1.prismatic.chromatic_field[0,0,0]:.4f}, Blue(Order)={state1.prismatic.chromatic_field[0,0,1]:.4f}, Yellow(Entropy)={state1.prismatic.chromatic_field[0,0,2]:.4f}")

    # 3. Stage 3 Execution: Macro Spacetime Deck & O(1) Local ROI Phase Scrubbing
    print("\n[Stage 3] Spacetime Deck Playback & Local ROI Phase Scrubbing ('Eye of Agamotto')...")
    # Define ROI Mask in middle of grid
    roi_mask = np.zeros((8, 8), dtype=bool)
    roi_mask[2:6, 2:6] = True
    engine.deck.set_roi(roi_mask)

    print(f"  * Set Local Scrubbing ROI Mask   : {np.sum(roi_mask)} cells selected (center 4x4 region)")
    initial_roi_phase = float(np.mean(engine.deck.waveform_tensor.phases[roi_mask]))
    initial_bg_phase = float(np.mean(engine.deck.waveform_tensor.phases[~roi_mask]))

    print(f"  * Initial ROI Mean Phase         : {initial_roi_phase:.4f} rad")
    print(f"  * Initial Background Mean Phase  : {initial_bg_phase:.4f} rad")

    print("\n  --> Executing O(1) Local ROI Time Rewind (-1.57 rad / -90 deg)...")
    state2 = engine.step(dt=0.1, roi_scrub_delta=-1.57)

    new_roi_phase = float(np.mean(state2.deck.waveform_tensor.phases[roi_mask]))
    new_bg_phase = float(np.mean(state2.deck.waveform_tensor.phases[~roi_mask]))

    print(f"  * Post-Scrub ROI Mean Phase      : {new_roi_phase:.4f} rad (Shifted local timeline)")
    print(f"  * Post-Scrub Background Mean Phase: {new_bg_phase:.4f} rad (Maintained global timeline)")

    # 4. Stage 4 Execution: Meta-Observer & Invariant Core Resilience
    print("\n[Stage 4] Meta-Observer Metacognition & Invariant Core Resilience under Noise...")
    print("  --> Injecting Noise Perturbations into Substrate over 5 cycles...")

    for i in range(5):
        noise = np.random.randn(8, 8) * 0.5
        st = engine.step(numerical_input=noise, dt=0.1)
        meta = st.meta_feedback
        print(f"   Cycle {i+1}: Invariant Coherence = {meta.invariant_coherence:.4f} | Divergence = {meta.topological_divergence:.4f} | Core Intact = {meta.invariant_intact} | Stiffness Adj = {meta.feedback_adjustment['constraint_stiffness']:+.4f}")

    print("\n" + "=" * 84)
    print(" ELYSIA :: PRISMATIC PHASE TRANSITION & SPACETIME DECK DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 84)


if __name__ == "__main__":
    main()
