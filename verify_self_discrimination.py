"""
Verification script for Self-Discriminative Mechanism and Domain-Transcendent Mass Isomorphism.
Validates label-free self-discrimination via residual impedance Delta Z.
"""

import numpy as np
from core.physics.constructive_causal_spacetime import ConstructiveLogicDiscriminator


def verify_self_discrimination():
    print("=== [2/3] Verifying Self-Discriminative Mechanism & Mass Isomorphism ===")
    discriminator = ConstructiveLogicDiscriminator(feature_dim=8)

    # Archetypal Mass Mechanism (Inertia + Tension Compression Curve)
    # Common homological stem mechanism: Exponential tension accumulation and relaxation
    stem_mechanism = np.exp(-np.linspace(0, 3, 8)) * np.cos(np.linspace(0, np.pi, 8))

    # Domain 1: Physical Mass (Inertia + Gravitational Resistance)
    physical_mass_wave = stem_mechanism * 1.5 + np.array([0.01, -0.01, 0.02, 0.0, -0.01, 0.01, 0.0, 0.01])

    # Domain 2: Informational Mass (Connectivity Density + Attractor Gravity)
    info_mass_wave = stem_mechanism * 4.2 + np.array([-0.01, 0.01, 0.0, 0.01, 0.01, -0.01, 0.02, 0.0])

    # Domain 3: Semantic Mass (Context Compression Tension + Meaning Inertia)
    semantic_mass_wave = stem_mechanism * 0.8 + np.array([0.0, 0.02, -0.01, 0.01, -0.01, 0.0, 0.01, -0.01])

    # Domain 4: Non-Isomorphic Noise (Random unaligned wave)
    random_noise_wave = np.random.randn(8)

    # Perform discrimination
    archetype_res = discriminator.discriminate_mass_archetype(
        physical_mass_wave, info_mass_wave, semantic_mass_wave
    )

    print(f"Common Archetype Avg Delta Z: {archetype_res['common_archetype_delta_z']:.6f}")
    print(f"Phys-Info Isomorphism: {archetype_res['phys_info_isomorphism']}")
    print(f"Phys-Sem Isomorphism: {archetype_res['phys_sem_isomorphism']}")
    print(f"Info-Sem Isomorphism: {archetype_res['info_sem_isomorphism']}")
    print(f"Archetype Isomorphism Verified: {archetype_res['archetype_verified']}")

    assert archetype_res["archetype_verified"], "Mass archetype isomorphism across 3 domains must be verified."

    # Test noise discrimination
    noise_res = discriminator.measure_residual_impedance(random_noise_wave, stem_mechanism)
    print(f"Random Noise Residual Impedance Delta Z: {noise_res['residual_impedance_delta_z']:.6f}")
    assert noise_res["residual_impedance_delta_z"] > 0.1, "Non-isomorphic noise must yield high residual impedance Delta Z."

    print("✓ Self-Discriminative Mechanism & Mass Isomorphism Verified Successfully!\n")


if __name__ == "__main__":
    verify_self_discrimination()
