"""
Unit Tests for Neuro-Phase Causal Engine.
Tests phase transitions (Gas, Liquid, Solid/ICE), dual-axis causality, invariant anchors,
environmental pressure tensors, Landau-Ginzburg free energy, morphological plasticity, and the 4 conscious mechanisms:
1. Attention (Energy Lens)
2. Intent / Teleology (Attractor Vector)
3. Sensory Grounding & Bidirectional Phase Coupling (Cross-Resonance, q_err resource)
4. Morphological Plasticity (Dynamic Coupling Rewiring & Coordinate Adaptation)
"""

import math
import numpy as np
import pytest

from core.consciousness.neuro_phase_causal_engine import (
    NeuroPhaseCausalEngine,
    NeuroPhaseState,
    PhaseType,
    NeuroRotorNode,
    InvariantAnchor,
    EnvironmentalPressure,
    PhaseState,
    ExternalWaveStream
)


def test_engine_initialization():
    engine = NeuroPhaseCausalEngine(num_nodes=8, lattice_dims=(2, 2, 2))
    assert len(engine.nodes) == 8
    assert engine.global_phase_state == PhaseType.GAS
    assert engine.coupling_matrix.shape == (8, 8)
    assert engine.anchor is not None
    assert engine.anchor.spectral_invariants.shape[0] == 8
    assert engine.anchor.geometric_skeleton.shape == (8, 3)


def test_invariant_anchor_and_pressure_tensor():
    num_nodes = 4
    topo = np.eye(num_nodes, dtype=complex)
    spect = np.array([0.0, 1.0, 2.0, 3.0])
    skeleton = np.zeros((num_nodes, 3))

    anchor = InvariantAnchor(
        num_nodes=num_nodes,
        topo_matrix=topo,
        spectral_invariants=spect,
        geometric_skeleton=skeleton
    )
    assert anchor.num_nodes == 4

    pressure = EnvironmentalPressure(
        thermal_noise=1.0,
        shear_stress=0.5,
        directional_flux=np.array([1.0, 0.0, 0.0]),
        phase_error=0.25
    )
    # magnitude = 1.0 + 0.5 + 1.0 + (0.25 * 2.0) = 3.0
    assert abs(pressure.magnitude - 3.0) < 1e-6


def test_free_energy_and_phase_map_eval():
    engine = NeuroPhaseCausalEngine(num_nodes=8, lattice_dims=(2, 2, 2))
    pressure = EnvironmentalPressure(thermal_noise=0.2, phase_error=0.01)

    # Coherent phases -> high order parameter eta
    rotor_phases = np.zeros(8)
    p_state = engine.causal_map.evaluate_phase_transition(
        rotor_phases=rotor_phases,
        rotor_weights=engine.coupling_matrix,
        pressure=pressure
    )

    assert p_state.order_parameter > 0.95
    assert p_state.current_phase == PhaseType.ICE
    assert isinstance(p_state.free_energy, float)


def test_morphological_plasticity_adaptation():
    engine = NeuroPhaseCausalEngine(num_nodes=8, lattice_dims=(2, 2, 2))

    # 1. Liquid state adaptation under directional flux
    engine.set_temperature(1.0) # Liquid phase
    engine.pressure.directional_flux = np.array([2.0, 0.0, 0.0])

    initial_pos = np.copy(engine.nodes["rotor_0_0_0"].position)

    for _ in range(10):
        engine.adapt_morphology()

    new_pos = engine.nodes["rotor_0_0_0"].position
    # Positions should shift in response to environmental directional flux and anchor restoration
    assert not np.array_equal(initial_pos, new_pos)


def test_phase_transitions():
    engine = NeuroPhaseCausalEngine(num_nodes=8, lattice_dims=(2, 2, 2))

    # 1. High Temperature -> GAS
    engine.set_temperature(3.0)
    assert engine.global_phase_state == PhaseType.GAS

    # 2. Medium Temperature -> LIQUID
    engine.set_temperature(1.2)
    assert engine.global_phase_state == PhaseType.LIQUID

    # 3. Low Temperature / High Phase-Locking -> ICE / SOLID
    engine.set_temperature(0.1)
    # Force all phases to be synchronized to simulate phase lock
    for node in engine.nodes.values():
        node.phase = 1.0
    engine.update_phase_state()
    assert engine.global_phase_state in (PhaseType.ICE, PhaseType.SOLID)
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


def test_sensory_grounding_and_bidirectional_negotiation():
    engine = NeuroPhaseCausalEngine(num_nodes=4, lattice_dims=(2, 2, 1))

    ext_phases = np.array([0.5, 0.5, 0.5, 0.5])
    ext_stream = ExternalWaveStream(
        modality="text",
        wave_phases=ext_phases,
        frequencies=np.array([40.0, 40.0, 40.0, 40.0])
    )

    result = engine.negotiate_bidirectional_phase(external_stream=ext_stream, coupling_gain=2.0)

    assert isinstance(result.q_err, float)
    assert 0.0 <= result.resonance_level <= 1.0
    assert engine.last_q_err == result.q_err


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
