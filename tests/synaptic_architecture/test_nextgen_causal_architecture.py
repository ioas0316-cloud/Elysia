import pytest
import torch
from synaptic_architecture.phase_rectification import PhaseRectificationFunction
from synaptic_architecture.cpt_causal_machine import PhaseTransitionCausalMachine
from synaptic_architecture.sparse_hebbian_learner import SparseHebbianCausalLearner
from synaptic_architecture.hierarchical_hebbian_engine import HierarchicalHebbianCausalEngine


def test_phase_rectification_function():
    torch.manual_seed(42)
    batch_size, dim = 4, 128
    continuous_h = torch.randn(batch_size, dim)

    rectifier = PhaseRectificationFunction(embed_dim=dim, num_symbols=32, tau=0.2, num_phases=8)
    rectified_state, logs = rectifier(continuous_h, temperature=0.01)

    assert rectified_state.shape == (batch_size, dim)
    assert "active_symbol_id" in logs
    assert len(logs["active_symbol_id"]) == batch_size
    assert 0.0 <= logs["sparsity_ratio"] <= 1.0


def test_phase_transition_causal_machine():
    torch.manual_seed(777)
    N = 4
    J = torch.tensor([
        [ 0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0],
        [ 2.5,  0.0,  0.0, -1.5],
        [-2.5,  1.5, -1.5,  0.0]
    ])

    cpt_machine = PhaseTransitionCausalMachine(num_switches=N, causal_graph_J=J, T_critical=1.0)

    # Fluid phase (T = 2.0)
    noisy_h = torch.tensor([0.2, 0.1, -0.05, 0.02])
    state_fluid = cpt_machine.observe_and_transition(noisy_h, T_current=2.0)
    assert state_fluid.shape == (N,)
    assert set(state_fluid.tolist()).issubset({1.0, -1.0})

    # Crystalline phase (T = 0.1)
    danger_h = torch.tensor([1.2, 0.1, 0.0, 0.0])
    state_crystalline = cpt_machine.observe_and_transition(danger_h, T_current=0.1)

    # Check deterministic sign collapse
    assert state_crystalline[2].item() == 1.0  # Danger (+1) excites avoidance (+1)
    assert state_crystalline[3].item() == -1.0 # Danger (+1) inhibits approach (-1)

    macro_id = cpt_machine.get_macro_state_id()
    assert isinstance(macro_id, int)


def test_sparse_hebbian_causal_learner():
    torch.manual_seed(42)
    N = 4
    learner = SparseHebbianCausalLearner(num_switches=N, lr=0.1, sparsity_gamma=0.05, threshold=0.05)

    sequence_data = [
        torch.tensor([ 1.0, -1.0, -1.0,  1.0]),  # t=0: [0] active
        torch.tensor([-1.0,  1.0, -1.0, -1.0]),  # t=1: [1] active (caused by 0)
        torch.tensor([-1.0, -1.0,  1.0,  1.0]),  # t=2: [2] active (caused by 1)
        torch.tensor([ 1.0, -1.0, -1.0, -1.0]),  # t=3: [0] active
        torch.tensor([-1.0,  1.0, -1.0,  1.0]),  # t=4: [1] active
        torch.tensor([-1.0, -1.0,  1.0, -1.0]),  # t=5: [2] active
    ]

    for step_switches in sequence_data:
        learner.update_causal_matrix(step_switches)

    sparse_J = learner.get_sparse_causal_graph()
    assert sparse_J.shape == (N, N)
    # Check that self-loops are 0
    assert torch.diagonal(sparse_J).abs().sum().item() == 0.0
    # Check learned causal link 0 -> 1 (J[1, 0] > 0)
    assert sparse_J[1, 0].item() > 0.0


def test_hierarchical_hebbian_causal_engine():
    torch.manual_seed(42)
    engine = HierarchicalHebbianCausalEngine(num_micro=8, num_macro=3)

    scenario_stream = [
        torch.tensor([ 1, -1, -1, -1,  1, -1, -1, -1], dtype=torch.float32),
        torch.tensor([-1,  1, -1, -1, -1,  1, -1, -1], dtype=torch.float32),
        torch.tensor([-1, -1,  1, -1, -1, -1,  1, -1], dtype=torch.float32),
    ]

    for micro_state in scenario_stream:
        res = engine.forward_step(micro_state)
        assert "micro_switches" in res
        assert "macro_concepts" in res
        assert len(res["macro_concepts"]) == 3

    assert engine.J_micro.shape == (8, 8)
    assert engine.J_macro.shape == (3, 3)
