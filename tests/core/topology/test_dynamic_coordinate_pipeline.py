import pytest
from core.topology.dynamic_coordinate_pipeline import PyCausalTracePool, PyDynamicCoordinateEngine

def test_py_causal_trace_pool():
    pool = PyCausalTracePool(3)
    assert pool.count == 3

    idx0 = pool.push_back(val=1.5, delta=0.2, op_id=1, ctx_mask=0x1, inv_score=0.9, parent_idx=-1, res_x=0.001, is_axis=1)
    assert idx0 == 3
    assert pool.is_axis[idx0] == 1

    idx1 = pool.push_back(val=2.0, delta=0.3, op_id=2, ctx_mask=0x1, inv_score=0.95, parent_idx=idx0, res_x=0.001, is_axis=1)
    chain = pool.trace_back(idx1)
    assert chain == [idx1, idx0]

def test_py_dynamic_coordinate_engine():
    engine = PyDynamicCoordinateEngine(condensation_threshold=0.8, relativization_friction=0.5)
    pool = PyCausalTracePool(2)

    # Node 0: Variable x
    pool.values[0] = 0.0
    pool.resistor_x[0] = 4.0
    pool.is_axis[0] = 0
    pool.invariance_scores[0] = 0.75

    # Node 1: Fixed Axis (1,2)
    pool.values[1] = 10.0
    pool.resistor_x[1] = 0.001
    pool.is_axis[1] = 1
    pool.invariance_scores[1] = 0.95

    # Forward Step
    engine.forward_step(pool, [8.0, 8.0])
    assert pool.values[0] > 0.0
    assert pool.values[1] > 10.0

    # Low Error -> Condensation (x -> Axis)
    engine.reflect_and_mutate(pool, [0.01, 0.01])
    assert pool.invariance_scores[0] >= 0.8
    assert pool.is_axis[0] == 1

    # High Error on Node 1 -> Relativization (Axis -> x)
    engine.reflect_and_mutate(pool, [0.01, 0.7])
    assert pool.is_axis[1] == 0
    assert pool.resistor_x[1] > engine.min_resistor_x
