"""
Unit Tests for IrrationalityTranscendentalSpectrumEngine
======================================================
Tests all 5 pillars:
1. Denominator scale tension (q) and irrationality exponent (\\mu) error bound calculation.
2. Recursive fractionation (fraction of fraction) fractal hierarchy generation.
3. Transcendental super-spectrum (\\pi of \\pi) multi-rotor phase interaction and resonance.
4. Trigonometric phase-locking (\\sin / \\cos) continuous Kuramoto-style dynamics.
5. CausalBlockDNA field resonance propagation and modular Lego/DNA self-assembly.
"""

import pytest
import numpy as np
from core.physics.irrationality_spectrum_engine import (
    IrrationalityTranscendentalSpectrumEngine,
    CausalBlockDNA,
    SuperSpectrumResonance
)


def test_engine_initialization():
    engine = IrrationalityTranscendentalSpectrumEngine(seed=42)
    assert "pi_root" in engine.blocks
    assert "pi_of_pi" in engine.blocks
    assert "rotor_sincos" in engine.blocks
    assert len(engine.blocks) == 3


def test_irrationality_scale_tension_evaluation():
    engine = IrrationalityTranscendentalSpectrumEngine(seed=42)
    res = engine.evaluate_irrationality_scale_tension("pi_root", target_val=np.pi)

    assert res["scale_tension_q"] == 10
    assert res["best_numerator_p"] == 31  # round(3.14159... * 10) = 31 -> 31/10 = 3.1
    assert abs(res["actual_error"] - abs(np.pi - 3.1)) < 1e-5
    assert res["bound_satisfied"] is True
    assert res["tension_energy"] > 0.0
    assert engine.causal_gravity_curvature > 0.0


def test_recursive_fractionation():
    engine = IrrationalityTranscendentalSpectrumEngine(seed=42)
    sub_blocks = engine.execute_recursive_fractionation("pi_root", num_sub_fractions=2)

    assert len(sub_blocks) == 2
    assert "pi_root_frac_1" in engine.blocks
    assert "pi_root_frac_2" in engine.blocks

    frac1 = engine.blocks["pi_root_frac_1"]
    assert frac1.parent_id == "pi_root"
    assert frac1.depth == 1
    assert frac1.scale_tension == 20.0  # 10.0 * (1 + 1)
    assert "pi_root_frac_1" in engine.blocks["pi_root"].children_ids


def test_transcendental_super_spectrum():
    engine = IrrationalityTranscendentalSpectrumEngine(seed=42)
    resonance = engine.evaluate_transcendental_super_spectrum("pi_root", "pi_of_pi")

    assert isinstance(resonance, SuperSpectrumResonance)
    assert resonance.primary_block_id == "pi_root"
    assert resonance.secondary_block_id == "pi_of_pi"
    assert -1.0 <= resonance.phase_coherence <= 1.0
    assert resonance.field_energy >= 0.0
    assert resonance.harmonic_ratio > 0.0


def test_trigonometric_phase_locking():
    engine = IrrationalityTranscendentalSpectrumEngine(seed=42)
    initial_phase_pi = engine.blocks["pi_root"].phase

    # Step continuous time
    engine.apply_trigonometric_phase_locking(dt=0.05, coupling_strength=0.8)

    updated_phase_pi = engine.blocks["pi_root"].phase
    assert updated_phase_pi != initial_phase_pi
    assert 0.0 <= updated_phase_pi < 2 * np.pi
    assert engine.time == 0.05


def test_field_resonance_and_self_assembly():
    engine = IrrationalityTranscendentalSpectrumEngine(seed=42)

    # Force identical phases and chromatic signatures to test coupling bond formation
    block_a = CausalBlockDNA(
        block_id="dna_node_a",
        phase=0.0,
        frequency=10.0,
        amplitude=1.0,
        scale_tension=5.0,
        irrationality_exponent=2.0,
        chromatic_signature=np.array([0.5, 0.5, 0.0], dtype=np.float64)
    )
    block_b = CausalBlockDNA(
        block_id="dna_node_b",
        phase=0.0,
        frequency=10.0,
        amplitude=1.0,
        scale_tension=5.0,
        irrationality_exponent=2.0,
        chromatic_signature=np.array([0.5, 0.5, 0.0], dtype=np.float64)
    )
    engine.add_causal_block(block_a)
    engine.add_causal_block(block_b)

    assembled = engine.propagate_field_resonance_and_self_assemble(coherence_threshold=0.9)

    pair_found = any(
        (a == "dna_node_a" and b == "dna_node_b") or (a == "dna_node_b" and b == "dna_node_a")
        for a, b, _ in assembled
    )
    assert pair_found is True
    assert "dna_node_b" in block_a.coupling_affinity
    assert "dna_node_a" in block_b.coupling_affinity
