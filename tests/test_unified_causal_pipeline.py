import sys
import os
import pytest

# Add parent directory to path to import synaptic_architecture
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from synaptic_architecture.unified_causal_pipeline import (
    SignalCartridge,
    SignalBufferComponent,
    LockFreeTripleBuffer,
    partition_domain_chunks,
    CausalDAG,
    SubspaceConstraintSolver,
    SnapshotRingBuffer,
    DeadReckoningInertializer,
    ZeroCopyTensorView,
    VulkanTimelineSemaphore,
    NetworkSignalPacket,
    UnifiedCausalPipeline,
    ENTITY_STATE_DTYPE
)


def test_entity_state_alignment():
    assert ENTITY_STATE_DTYPE.itemsize == 64, "EntityState struct size must be exactly 64 bytes"


def test_signal_cartridge_bitpacking():
    proto_id = 0xAB
    phase = 0x03
    param = 0x8000  # 0.5 normalized
    seed = 12345678

    cartridge = SignalCartridge.pack(proto_id, phase, param, seed)

    assert cartridge.protocol_id == proto_id
    assert cartridge.phase == phase
    assert cartridge.param == param
    assert pytest.approx(cartridge.param_normalized, abs=1e-3) == 0.5
    assert cartridge.seed == seed

    # Zero-Copy byte serialization
    raw_bytes = cartridge.as_bytes()
    assert len(raw_bytes) == 8
    restored = SignalCartridge.from_bytes(raw_bytes)
    assert restored.raw == cartridge.raw


def test_signal_buffer_component():
    buf = SignalBufferComponent()
    assert buf.count == 0

    c1 = SignalCartridge.pack(1, 0, 100, 42)
    c2 = SignalCartridge.pack(2, 1, 200, 43)

    assert buf.push(c1)
    assert buf.push(c2)
    assert buf.count == 2
    assert len(buf.active_signals()) == 2

    buf.clear()
    assert buf.count == 0
    assert len(buf.active_signals()) == 0


def test_lock_free_triple_buffer():
    tb = LockFreeTripleBuffer(capacity=10)
    wb = tb.get_write_buffer()
    wb[0]['entity_id'] = 999
    wb[0]['position'] = [1.0, 2.0, 3.0]

    tb.swap_buffers()

    rb = tb.acquire_read_buffer()
    assert rb[0]['entity_id'] == 999
    assert np.allclose(rb[0]['position'], [1.0, 2.0, 3.0])


def test_domain_partitioning():
    buf = np.zeros(100, dtype=ENTITY_STATE_DTYPE)
    chunks = partition_domain_chunks(buf, num_chunks=4)
    assert len(chunks) == 4
    assert sum(len(c) for c in chunks) == 100


def test_reactive_causal_dag():
    dag = CausalDAG()
    executed_nodes = []

    def fn0(states, sigs):
        executed_nodes.append(0)

    def fn1(states, sigs):
        executed_nodes.append(1)

    def fn2(states, sigs):
        executed_nodes.append(2)

    dag.add_node(0, "Root", fn0)
    dag.add_node(1, "ChildA", fn1, dependencies=[0])
    dag.add_node(2, "ChildB", fn2, dependencies=[1])

    states = np.zeros(10, dtype=ENTITY_STATE_DTYPE)

    # First evaluation: all nodes dirty
    cnt = dag.evaluate_causal_chain(states)
    assert cnt == 3
    assert executed_nodes == [0, 1, 2]

    # Second evaluation without dirty: 0 nodes evaluated (Zero-Polling Culling)
    executed_nodes.clear()
    cnt_culled = dag.evaluate_causal_chain(states)
    assert cnt_culled == 0
    assert len(executed_nodes) == 0

    # Mark cause on Root: triggers full reactive chain
    dag.mark_cause(0)
    cnt_reactive = dag.evaluate_causal_chain(states)
    assert cnt_reactive == 3
    assert executed_nodes == [0, 1, 2]


def test_subspace_constraint_solver():
    solver = SubspaceConstraintSolver(hinge_axis=np.array([0.0, 1.0, 0.0]))
    chunk = np.zeros(5, dtype=ENTITY_STATE_DTYPE)
    chunk['velocity'] = np.array([[3.0, 4.0, 5.0]] * 5)

    solver.solve_constraints(chunk)

    # Velocity must be projected onto Y axis [0, 4, 0]
    for i in range(5):
        assert np.allclose(chunk[i]['velocity'], [0.0, 4.0, 0.0])


def test_snapshot_ring_buffer_and_rollback():
    capacity = 10
    ring = SnapshotRingBuffer(capacity)
    dag = CausalDAG()
    solver = SubspaceConstraintSolver()

    states = np.zeros(capacity, dtype=ENTITY_STATE_DTYPE)
    states['position'] = np.array([[1.0, 0.0, 0.0]] * capacity)

    def fn(st, sigs):
        st['position'] += 1.0

    dag.add_node(0, "Move", fn)

    # Save frames 0 to 5
    for f in range(6):
        dag.mark_cause(0)
        dag.evaluate_causal_chain(states)
        ring.save_snapshot(f, states, [], f * 100)

    # At frame 5, position should be 7.0 (1 initial + 6 increments)
    assert states[0]['position'][0] == 7.0

    # Rollback to frame 2 and fast-forward resimulate to frame 5
    success = ring.rollback_and_resimulate(
        target_frame=2,
        current_frame=5,
        state_buffer=states,
        dag=dag,
        solver=solver
    )

    assert success
    assert states[0]['position'][0] == 7.0  # Successfully restored to current frame state


def test_zero_copy_tensor_view():
    raw_buf = np.zeros(16, dtype=ENTITY_STATE_DTYPE)
    raw_buf['latent_param'] = np.arange(16, dtype=np.float32)

    tensor_view = ZeroCopyTensorView(raw_buf, shape=(16, 16))
    numpy_view = tensor_view.as_numpy_view()

    assert numpy_view.shape == (16, 16)
    assert numpy_view.base is not None or numpy_view.flags.c_contiguous


def test_vulkan_timeline_semaphore():
    sem = VulkanTimelineSemaphore(initial_value=0)
    assert sem.get_value() == 0
    assert not sem.can_reuse_slot(required_frame_slot=10)

    sem.signal(10)
    assert sem.get_value() == 10
    assert sem.can_reuse_slot(required_frame_slot=10)


def test_network_signal_packet_serialization():
    c1 = SignalCartridge.pack(10, 1, 1000, 9999)
    c2 = SignalCartridge.pack(20, 2, 2000, 8888)

    pkt = NetworkSignalPacket(sequence=100, entity_net_id=5, signals=[c1, c2])
    data = pkt.serialize()

    restored = NetworkSignalPacket.deserialize(data)
    assert restored.sequence == 100
    assert restored.entity_net_id == 5
    assert len(restored.signals) == 2
    assert restored.signals[0].protocol_id == 10
    assert restored.signals[1].protocol_id == 20


def test_unified_causal_pipeline_full():
    pipeline = UnifiedCausalPipeline(capacity=50)

    # Inject input signal
    cartridge = SignalCartridge.pack(protocol_id=1, phase=0, param=100, seed=42)
    pipeline.inject_signal(cartridge)

    res = pipeline.step_frame()
    assert res['frame'] == 0
    assert res['nodes_evaluated'] == 3  # All nodes executed on signal inject

    # Second step without signal -> zero-polling culling skips dormant nodes (0 nodes evaluated)
    res2 = pipeline.step_frame()
    assert res2['frame'] == 1
    assert res2['nodes_evaluated'] == 0  # Zero-polling culling active

    tensor_view, net_bytes = pipeline.dispatch_outputs()
    assert tensor_view.shape == (50, 16)
    assert len(net_bytes) == 32
