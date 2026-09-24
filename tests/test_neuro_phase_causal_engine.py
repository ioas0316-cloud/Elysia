"""
Unit Tests for Neuro-Phase Causal Engine.
Tests phase transitions (Gas, Liquid, Solid) and the 4 conscious mechanisms:
1. Attention (Energy Lens)
2. Intent / Teleology (Attractor Vector)
3. Sensory Grounding (Cross-Resonance)
4. Plasticity (Dynamic Coupling Rewiring)
"""

import math
import numpy as np
import pytest

from core.consciousness.neuro_phase_causal_engine import (
    NeuroPhaseCausalEngine,
    NeuroPhaseState,
    NeuroRotorNode
)


def test_engine_initialization():
    engine = NeuroPhaseCausalEngine(num_nodes=8, lattice_dims=(2, 2, 2))
    assert len(engine.nodes) == 8
    assert engine.global_phase_state == NeuroPhaseState.GAS
    assert engine.coupling_matrix.shape == (8, 8)


def test_phase_transitions():
    engine = NeuroPhaseCausalEngine(num_nodes=8, lattice_dims=(2, 2, 2))

    # 1. High Temperature -> GAS
    engine.set_temperature(3.0)
    assert engine.global_phase_state == NeuroPhaseState.GAS

    # 2. Medium Temperature -> LIQUID
    engine.set_temperature(1.2)
    assert engine.global_phase_state == NeuroPhaseState.LIQUID

    # 3. Low Temperature / High Phase-Locking -> SOLID
    engine.set_temperature(0.1)
    # Force all phases to be synchronized to simulate phase lock
    for node in engine.nodes.values():
        node.phase = 1.0
    engine.update_phase_state()
    assert engine.global_phase_state == NeuroPhaseState.SOLID
    assert engine.calculate_global_coherence() > 0.95


def test_attention_lens():
    engine = NeuroPhaseCausalEngine(num_nodes=8, lattice_dims=(2, 2, 2))
    target = ["rotor_0_0_0", "rotor_0_0_1"]

    engine.apply_attention_lens(target_nodes=target, gain=4.0)

    assert engine.nodes["rotor_0_0_0"].energy > engine.nodes["rotor_1_1_1"].energy
    assert engine.attention_focus["rotor_0_0_0"] == 4.0
    assert engine.attention_focus["rotor_1_1_1"] == 0.2


def test_teleological_intent():
    engine = NeuroPhaseCausalEngine(num_nodes=4, lattice_dims=(2, 2, 1))
    target_phases = {
        "rotor_0_0_0": 1.0,
        "rotor_0_1_0": 1.0,
        "rotor_1_0_0": 1.0,
        "rotor_1_1_0": 1.0
    }
    # Equalize intrinsic frequencies to isolate intent attractor pull
    for node in engine.nodes.values():
        node.intrinsic_frequency = 40.0

    engine.set_teleological_intent(target_phases=target_phases, strength=200.0)
    assert engine.active_intent is not None

    # Step simulation multiple times; phases should pull towards target
    engine.set_temperature(0.1)
    for _ in range(100):
        engine.step(dt=0.001)

    coherence = engine.calculate_global_coherence()
    assert coherence > 0.85  # Intent pulled system towards phase synchronization


def test_sensory_grounding():
    engine = NeuroPhaseCausalEngine(num_nodes=4, lattice_dims=(2, 2, 1))
    external_wave = {"rotor_0_0_0": math.pi / 2.0}

    initial_phase = engine.nodes["rotor_0_0_0"].phase
    engine.inject_sensory_grounding(external_wave=external_wave, coupling_gain=2.0)

    new_phase = engine.nodes["rotor_0_0_0"].phase
    # Phase should have shifted towards pi/2
    assert new_phase != initial_phase


def test_plasticity_rewiring():
    engine = NeuroPhaseCausalEngine(num_nodes=4, lattice_dims=(2, 2, 1))
    engine.set_temperature(1.0)  # Liquid phase allows plasticity

    idx0 = engine.node_id_map["rotor_0_0_0"]
    idx1 = engine.node_id_map["rotor_0_1_0"]
    initial_coupling = engine.coupling_matrix[idx0, idx1]

    # Synchronize node 0 and node 1 phases continuously
    for _ in range(50):
        engine.nodes["rotor_0_0_0"].phase = 0.5
        engine.nodes["rotor_0_1_0"].phase = 0.5
        engine.step(dt=0.01)

    # Coupling strength between co-firing nodes should increase
    assert engine.coupling_matrix[idx0, idx1] > initial_coupling
