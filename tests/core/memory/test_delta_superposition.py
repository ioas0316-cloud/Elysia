import pytest
import numpy as np
import threading
from core.memory.delta_superposition import (
    ImmutableBaseSlab,
    LockFreeDeltaRingBuffer,
    ObserverView,
    DeltaSuperpositionEngine
)
from core.memory.state_dag import StateDAGManager
from core.memory.causal_gc import CausalAwareGC

def test_immutable_base_slab():
    # Test Dict Base Slab
    dict_slab = ImmutableBaseSlab({"temp": 20.0, "status": "INIT"})
    assert not dict_slab.is_vector
    assert dict_slab.dict_data == {"temp": 20.0, "status": "INIT"}

    # Test Vector Base Slab
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vec_slab = ImmutableBaseSlab(vec)
    assert vec_slab.is_vector
    assert np.array_equal(vec_slab.vector_data, vec)

    # Immutability check
    with pytest.raises(ValueError):
        vec_slab.vector_data[0] = 999.0

def test_lock_free_delta_ring_buffer_kv():
    ring = LockFreeDeltaRingBuffer(capacity=4)
    idx0 = ring.push_kv_delta("a", 1)
    idx1 = ring.push_kv_delta("b", 2)
    idx2 = ring.push_kv_delta("c", 3)
    idx3 = ring.push_kv_delta("d", 4)

    assert idx0 == 0 and idx3 == 3
    assert ring.get_kv_delta(idx0) == ("a", 1)

    # Wrap around push to expire idx0
    idx4 = ring.push_kv_delta("e", 5)
    assert idx4 == 4
    assert ring.get_kv_delta(idx0) is None  # Expired
    assert ring.get_kv_delta(idx4) == ("e", 5)

def test_lock_free_delta_ring_buffer_vector_simd():
    dim = 4
    base_vec = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    ring = LockFreeDeltaRingBuffer(capacity=16, vector_dim=dim)

    d1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    d2 = np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32)

    idx1 = ring.push_vector_delta(d1)
    idx2 = ring.push_vector_delta(d2)

    composite = ring.composite_vector_superposition(base_vec, [idx1, idx2])
    expected = np.array([11.0, 22.0, 30.0, 40.0], dtype=np.float32)
    assert np.allclose(composite, expected)

def test_observer_view_zero_copy_branching():
    base_dict = {"temp": 20.0, "pressure": 1.0, "status": "INIT"}
    engine = DeltaSuperpositionEngine(base_dict, ring_capacity=100)

    root_view = engine.create_root_view()
    branch_a_1 = root_view.branch_and_apply_kv("temp", 100.0)
    branch_a_2 = branch_a_1.branch_and_apply_kv("status", "BOILING")

    branch_b_1 = root_view.branch_and_apply_kv("pressure", 2.5)

    # Base state in engine remains untouched
    assert engine.base_slab.dict_data["temp"] == 20.0

    # Views observe distinct superimposed states without full data copies
    obs_a = branch_a_2.observe()
    obs_b = branch_b_1.observe()

    assert obs_a == {"temp": 100.0, "pressure": 1.0, "status": "BOILING"}
    assert obs_b == {"temp": 20.0, "pressure": 2.5, "status": "INIT"}

def test_large_scale_zero_copy_branching_tree():
    engine = DeltaSuperpositionEngine({"val": 0}, ring_capacity=100000)
    root = engine.create_root_view()

    # Generate 10,000 virtual branches in O(1) memory overhead
    num_branches = 10000
    views = []
    for i in range(num_branches):
        views.append(root.branch_and_apply_kv("val", i))

    assert len(views) == 10000
    # Randomly sample and verify correctness
    assert views[42].observe()["val"] == 42
    assert views[9999].observe()["val"] == 9999

def test_concurrent_superposition_access():
    base_vec = np.zeros(64, dtype=np.float32)
    engine = DeltaSuperpositionEngine(base_vec, ring_capacity=10000)
    root = engine.create_root_view()

    def worker_thread(thread_id: int):
        v = root
        for step in range(100):
            d = np.zeros(64, dtype=np.float32)
            d[thread_id % 64] = float(step + 1)
            v = v.branch_and_apply_vector(d)
        res = v.observe()
        assert res[thread_id % 64] > 0

    threads = [threading.Thread(target=worker_thread, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

def test_dag_manager_and_causal_gc_integration():
    dag = StateDAGManager({"temp": 20.0, "status": "INIT"}, ring_capacity=16)
    node1 = dag.step({"temp": 30.0})
    node2 = dag.step({"status": "HEATING"})

    assert dag.current_node.get_state_chain() == {"temp": 30.0, "status": "HEATING"}

    gc = CausalAwareGC(dag)
    pruned = gc.run_cgc()
    # Current path is protected, so no active nodes pruned
    assert pruned == 0
