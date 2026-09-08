import pytest
import torch
from synaptic_architecture.lightweight_sparse_csm import (
    SparseCognitiveState,
    LightCognitiveStateMachine
)


def test_sparse_cognitive_state_memory_footprint():
    state = SparseCognitiveState(total_dim=10000, k_active=100)
    state.active_indices = torch.arange(0, 100, dtype=torch.int32)
    state.phases = torch.randint(0, 256, (100,), dtype=torch.uint8)

    mem_bytes = state.get_memory_footprint_bytes()
    assert mem_bytes <= 1024, f"Memory footprint {mem_bytes} bytes exceeds 1 KB limit"


def test_sparse_cognitive_state_clone_and_device():
    state = SparseCognitiveState(total_dim=10000, k_active=50)
    state.active_indices[:10] = 42
    cloned = state.clone()

    assert torch.equal(state.active_indices, cloned.active_indices)
    assert torch.equal(state.phases, cloned.phases)

    cloned.active_indices[0] = 999
    assert state.active_indices[0] != 999


def test_light_cognitive_state_machine_bifurcation():
    engine = LightCognitiveStateMachine(total_dim=10000, k_active=100, num_concepts=32)
    state = SparseCognitiveState(total_dim=10000, k_active=100)
    state.active_indices = torch.arange(0, 100, dtype=torch.int32)
    state.phases = torch.randint(0, 256, (100,), dtype=torch.uint8)

    # Fluid phase (mu <= 0.5)
    fluid_state = engine.step_bifurcation(state, mu_context=0.2)
    assert len(fluid_state.active_indices) == 100

    # Decision phase (mu > 0.5)
    decision_state = engine.step_bifurcation(state, mu_context=0.9)
    assert len(decision_state.active_indices) < 100
    assert len(decision_state.active_indices) >= 10


def test_light_cognitive_state_machine_attractor_recall():
    engine = LightCognitiveStateMachine(total_dim=10000, k_active=100, num_concepts=16)

    # Construct a state that matches concept 0 exactly
    state = SparseCognitiveState(total_dim=10000, k_active=100)
    state.active_indices = engine.concept_bank[0].clone()
    state.phases = torch.randint(0, 256, (100,), dtype=torch.uint8)

    winner_id, match_count = engine.recall_attractor(state)
    assert winner_id == 0
    assert match_count == 100


def test_light_cognitive_state_machine_hyperdimensional_binding():
    engine = LightCognitiveStateMachine(total_dim=10000, k_active=50, num_concepts=16)

    state1 = SparseCognitiveState(total_dim=10000, k_active=50)
    state1.active_indices = torch.arange(0, 50, dtype=torch.int32)
    state1.phases = torch.full((50,), 10, dtype=torch.uint8)

    state2 = SparseCognitiveState(total_dim=10000, k_active=50)
    state2.active_indices = torch.arange(100, 150, dtype=torch.int32)
    state2.phases = torch.full((50,), 20, dtype=torch.uint8)

    bound = engine.hyperdimensional_bind(state1, state2)
    assert len(bound.active_indices) == 50
    assert bound.active_indices[0] == (0 + 100) % 10000
    assert bound.phases[0] == 30


def test_light_cognitive_state_machine_autopoiesis():
    engine = LightCognitiveStateMachine(total_dim=10000, k_active=100, num_concepts=16)

    initial_state = SparseCognitiveState(total_dim=10000, k_active=100)
    initial_state.active_indices = torch.arange(0, 100, dtype=torch.int32)
    initial_state.phases = torch.full((100,), 100, dtype=torch.uint8)

    # Run autopoietic zero-input step
    next_state = engine.step_autopoiesis(initial_state, noise_level=0.05)

    assert len(next_state.active_indices) == 100
    assert not torch.equal(initial_state.phases, next_state.phases)
    # Self boundary maintained
    assert next_state.active_indices.dtype == torch.int32
