r"""
Unit Tests for State-Delta Knowledge Graph, Bit Modulation, and Fluid Phase Transitions
========================================================================================
"""

import pytest
import numpy as np
from core.topology.memory_topology_isomorphism import (
    StateDeltaKnowledgeGraph,
    BitModulatorDemodulator,
    FluidPhaseTransitionEngine,
)


def sample_operator(state: np.ndarray, scale: float) -> np.ndarray:
    return state * scale


def test_state_delta_knowledge_graph_tracing():
    graph = StateDeltaKnowledgeGraph()
    init_s = np.array([1.0, 2.0, 3.0])
    graph.add_node("s0", init_s)

    target_s = graph.apply_operator_and_record_edge(
        source_id="s0",
        target_id="s1",
        operator_fn=sample_operator,
        operator_id="scale_op",
        operator_params={"scale": 2.0},
        memory_offset=0x10,
        intent_id="double_values"
    )

    assert np.allclose(target_s, np.array([2.0, 4.0, 6.0]))
    edges = graph.inverse_trace("s1")
    assert len(edges) == 1
    assert edges[0].source_node_id == "s0"
    assert np.allclose(edges[0].what_delta, np.array([1.0, 2.0, 3.0]))
    assert edges[0].how_operator_id == "scale_op"


def test_bit_modulation_demodulation():
    modulator = BitModulatorDemodulator(buffer_size=128)
    data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    bytes_num = modulator.modulate(data, offset=0)
    assert bytes_num == 12

    obs_pos = np.array([0.0, 0.0, 0.0])
    target_pos = np.array([1.0, 0.0, 0.0])
    full_res = modulator.demodulate(offset=0, shape=(3,), observer_position=obs_pos, target_position=target_pos)
    assert np.allclose(full_res, data)

    obs_far = np.array([100.0, 0.0, 0.0])
    low_res = modulator.demodulate(offset=0, shape=(3,), observer_position=obs_far, target_position=target_pos)
    assert np.allclose(low_res, data * 0.1)


def test_fluid_phase_transitions():
    fluid = FluidPhaseTransitionEngine(grid_size=8)
    assert fluid.phase == "FLUID"

    fluid.step_fluid_dynamics(source_s=0.5)

    cryst_ok = fluid.crystallize()
    assert cryst_ok is True
    assert fluid.phase == "LATTICE"
    assert len(fluid.lattice_nodes) > 0

    melt_ok = fluid.melt(thermal_energy_delta_e=2.0)
    assert melt_ok is True
    assert fluid.phase == "FLUID"
    assert len(fluid.lattice_nodes) == 0
