import pytest
import torch
from synaptic_architecture.autopoietic_event_gnn import (
    STEEventGate,
    RelaxedEventGate,
    EventDrivenAsyncGNN,
    ste_event_gate,
)
from synaptic_architecture.complex_quaternion_pc import (
    quaternion_mul,
    ComplexPhasePCLayer,
    QuaternionPhasePCLayer,
    ComplexQuaternionPCNetwork,
)
from synaptic_architecture.gaussian_phase_tensor_field import (
    GaussianPhaseOscillatorNode,
    GaussianPhaseTensorField,
)


def test_ste_event_gate():
    delta = torch.tensor([0.05, 0.2, 0.1, 0.3], requires_grad=True)
    threshold = 0.15
    out = ste_event_gate(delta, threshold)

    expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.equal(out, expected)

    loss = out.sum()
    loss.backward()
    assert delta.grad is not None
    assert torch.equal(delta.grad, torch.ones_like(delta))


def test_relaxed_event_gate():
    gate = RelaxedEventGate(threshold=0.15, initial_beta=5.0)
    delta = torch.tensor([0.05, 0.2])
    out = gate(delta)
    assert out.size(0) == 2
    assert 0.0 <= out[0].item() <= 1.0
    assert 0.0 <= out[1].item() <= 1.0

    gate.step_temperature(factor=2.0)
    assert gate.beta == 10.0


def test_event_driven_async_gnn():
    gnn = EventDrivenAsyncGNN(num_nodes=4, in_channels=8, out_channels=8, threshold=0.1)
    adj_list = {0: [1, 2], 1: [2, 3], 2: [3], 3: []}

    initial_payload = torch.randn(8)
    initial_events = [(0.0, 0, initial_payload)]

    states, processed_count = gnn.forward_event_loop(adj_list, initial_events)
    assert states.shape == (4, 8)
    assert processed_count >= 1


def test_quaternion_mul():
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
    q2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # i
    res = quaternion_mul(q1, q2)
    assert torch.allclose(res, q2)


def test_complex_phase_pc_layer():
    layer = ComplexPhasePCLayer(in_features=8, out_features=4, gamma=0.1)
    batch_size = 2
    device = torch.device("cpu")

    layer.reset_state(batch_size, device)
    rand_phase = torch.randn(batch_size, 8) * 3.14159
    input_wave = torch.polar(torch.ones(batch_size, 8), rand_phase)

    # Initial error
    layer.update_state(bottom_input=input_wave)
    err1 = torch.mean(torch.abs(layer.e)).item()

    # Iterate state update for local phase locking
    for _ in range(10):
        layer.update_state(bottom_input=input_wave)

    err2 = torch.mean(torch.abs(layer.e)).item()
    assert err2 <= err1 + 1e-4

    layer.update_weights(lr=0.01)


def test_quaternion_phase_pc_layer():
    layer = QuaternionPhasePCLayer(in_features=8, out_features=4, gamma=0.1)
    batch_size = 2
    device = torch.device("cpu")

    layer.reset_state(batch_size, device)
    input_q = torch.randn(batch_size, 8, 4)
    input_q = input_q / torch.norm(input_q, dim=-1, keepdim=True)

    layer.update_state(bottom_input=input_q)
    assert layer.e.shape == (batch_size, 8, 4)
    layer.update_weights(lr=0.01)


def test_complex_quaternion_pc_network():
    net = ComplexQuaternionPCNetwork(layer_dims=[16, 8, 4], gamma=0.1, is_quaternion=False)
    input_wave = torch.polar(torch.ones(2, 16), torch.randn(2, 16))

    loss = net.fit_step(input_wave, infer_steps=5, lr=0.01)
    assert isinstance(loss, float)


def test_gaussian_phase_tensor_field():
    field = GaussianPhaseTensorField(initial_nodes=4, split_threshold=0.2)
    queries = torch.randn(10, 3)

    evaluated_q = field.evaluate_field_at_points(queries)
    assert evaluated_q.shape == (10, 4)

    target_q = torch.randn(field.nodes.num_nodes, 4)
    target_q = target_q / torch.norm(target_q, dim=-1, keepdim=True)

    res = field.update_phase_and_autopoietic_split(target_q, gamma=0.2, dt=0.1)
    assert "num_splits" in res
    assert "total_nodes" in res
    assert res["total_nodes"] >= 4
