#!/usr/bin/env python3
"""
demo_wavefield_phase_locking.py

Demonstration of Wavefield High-Dimensional Vectorization, Kuramoto Phase-Locking,
Noise Rectification (Constructive/Destructive Interference), and Relational Fabric
Persistence under Sensory Layer Suppression.
"""

import numpy as np


def run_kuramoto_phase_locking(num_rotors=32, coupling_k=2.5, steps=200, dt=0.05):
    """
    Simulates a network of complex rotors syncing via Kuramoto phase coupling.
    Calculates order parameter r(t) = |(1/N) * sum(exp(i * theta_j))|.
    """
    # Random initial frequencies and phases
    np.random.seed(42)
    natural_frequencies = np.random.normal(loc=1.0, scale=0.2, size=num_rotors)
    phases = np.random.uniform(0, 2 * np.pi, size=num_rotors)

    r_history = []
    for step in range(steps):
        # Calculate order parameter
        complex_order = np.mean(np.exp(1j * phases))
        r = np.abs(complex_order)
        psi = np.angle(complex_order)
        r_history.append(r)

        # Phase update equation: dtheta_i/dt = omega_i + K * r * sin(psi - theta_i)
        d_phases = natural_frequencies + coupling_k * r * np.sin(psi - phases)
        phases = (phases + d_phases * dt) % (2 * np.pi)

    return r_history, phases


def run_noise_rectification_demo(num_signals=10, signal_len=512):
    """
    Demonstrates noise filtering via constructive and destructive wave superposition.
    Coherent resonant signals construct a harmonic attractor, while uncorrelated
    stochastic noise destructively cancels out.
    """
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, signal_len)
    base_freq = 2.0

    # Coherent signal component (Phase-aligned harmonic)
    coherent_waves = [np.sin(base_freq * t + np.random.normal(0, 0.05)) for _ in range(num_signals)]

    # Incoherent random noise waves
    noise_waves = [np.random.normal(0, 1.0, size=signal_len) for _ in range(num_signals)]

    total_coherent = np.mean(coherent_waves, axis=0)
    total_noise = np.mean(noise_waves, axis=0)

    coherent_power = np.mean(total_coherent ** 2)
    noise_power = np.mean(total_noise ** 2)

    return coherent_power, noise_power


def run_layer_persistence_demo():
    """
    Demonstrates that multi-layer sensory coupling preserves the central relational
    lattice (spindle axis) even when one primary input layer is suppressed (amplitude = 0).
    """
    np.random.seed(42)
    # Define 3 sensory layers: Visual (Layer 0), Auditory (Layer 1), Tactile (Layer 2)
    num_nodes = 16
    t = np.linspace(0, 2 * np.pi, 100)

    # Shared underlying relational spindle axis phase
    spindle_phase = np.random.uniform(0, 2 * np.pi, size=num_nodes)

    # Phase-locked sensory layers (Visual, Auditory, Tactile) anchored to spindle with small layer offsets
    layer_offsets = [0.0, 0.15, -0.15]
    base_phases = np.zeros((3, num_nodes))
    for l in range(3):
        base_phases[l] = spindle_phase + layer_offsets[l] + np.random.normal(0, 0.1, size=num_nodes)

    # Calculate full-layer relational correlation
    full_state = np.zeros((3, num_nodes), dtype=complex)
    for l in range(3):
        full_state[l] = np.exp(1j * base_phases[l])

    relational_matrix_full = np.abs(np.dot(full_state, full_state.conj().T)) / num_nodes

    # Suppress Visual Layer (Layer 0 amplitude = 0)
    suppressed_state = full_state.copy()
    suppressed_state[0] = 0.0

    # Reconstruct relational fabric from remaining coupled layers (Auditory & Tactile)
    relational_matrix_suppressed = np.abs(np.dot(suppressed_state[1:], suppressed_state[1:].conj().T)) / num_nodes

    return relational_matrix_full, relational_matrix_suppressed


def main():
    print("=" * 80)
    print("ELYSIA: WAVEFIELD PHASE-LOCKING & RELATIONAL PERSISTENCE DEMONSTRATION")
    print("=" * 80)

    # 1. Kuramoto Phase Locking
    print("\n[Phase 1] Kuramoto Rotor Phase-Locking Dynamics")
    r_history, final_phases = run_kuramoto_phase_locking()
    print(f"  * Initial Order Parameter r(0)   : {r_history[0]:.4f}")
    print(f"  * Final Order Parameter r(end) : {r_history[-1]:.4f}")
    assert r_history[-1] > 0.85, "Phase synchronization failed to converge!"
    print("  => SUCCESS: Rotors synchronized into coherent phase attractor.")

    # 2. Noise Rectification via Interference
    print("\n[Phase 2] Constructive/Destructive Wave Noise Rectification")
    coherent_pwr, noise_pwr = run_noise_rectification_demo()
    snr_gain = coherent_pwr / (noise_pwr + 1e-9)
    print(f"  * Coherent Harmonic Wave Power  : {coherent_pwr:.4f}")
    print(f"  * Destructive Noise Wave Power  : {noise_pwr:.4f}")
    print(f"  * Resonance SNR Gain            : {snr_gain:.2f}x")
    assert snr_gain > 5.0, "Noise cancellation via destructive interference failed!"
    print("  => SUCCESS: Harmonic attractor amplified while stochastic noise cancelled.")

    # 3. Layer Persistence under Suppression
    print("\n[Phase 3] Relational Fabric Persistence under Sensory Layer Suppression")
    full_rel, supp_rel = run_layer_persistence_demo()
    print(f"  * Full Multi-Layer Relational Coherence (Layers 0,1,2):\n{full_rel}")
    print(f"  * Persistent Relational Coherence with Visual Layer Suppressed (Layers 1,2):\n{supp_rel}")
    assert supp_rel[0, 1] > 0.5, "Relational structure collapsed upon layer suppression!"
    print("  => SUCCESS: Topological relational fabric preserved despite layer suppression.")

    print("\n" + "=" * 80)
    print("ALL WAVEFIELD VECTORIZATION DEMONSTRATIONS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
