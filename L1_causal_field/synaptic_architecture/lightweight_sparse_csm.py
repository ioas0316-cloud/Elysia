import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class SparseCognitiveState:
    """
    10,000-dimensional cognitive state represented as a hyper-lightweight sparse structure
    with active integer indices and quantized 8-bit phases.
    Memory footprint: ~800 Bytes per state.
    """
    def __init__(self, total_dim: int = 10000, k_active: int = 100, device: Optional[torch.device] = None):
        self.total_dim = total_dim
        self.k_active = k_active
        self.device = device if device is not None else torch.device("cpu")

        # Sparse representations: active node indices (int32) and quantized phases (uint8, 0..255)
        self.active_indices = torch.zeros(k_active, dtype=torch.int32, device=self.device)
        self.phases = torch.zeros(k_active, dtype=torch.uint8, device=self.device)

    def to(self, device: torch.device) -> "SparseCognitiveState":
        self.device = device
        self.active_indices = self.active_indices.to(device)
        self.phases = self.phases.to(device)
        return self

    def get_memory_footprint_bytes(self) -> int:
        return (
            self.active_indices.element_size() * self.active_indices.nelement() +
            self.phases.element_size() * self.phases.nelement()
        )

    def clone(self) -> "SparseCognitiveState":
        new_state = SparseCognitiveState(self.total_dim, self.k_active, self.device)
        new_state.active_indices = self.active_indices.clone()
        new_state.phases = self.phases.clone()
        return new_state


class LightCognitiveStateMachine(nn.Module):
    """
    Ultra-lightweight Sparse Cognitive State Machine (LS-CSM)
    Replaces heavy tensor PDE simulations with hyperdimensional sparse operations,
    discrete bifurcation switches, and k-WTA attractor recall.
    """
    def __init__(self, total_dim: int = 10000, k_active: int = 100, num_concepts: int = 64):
        super().__init__()
        self.total_dim = total_dim
        self.k_active = k_active
        self.num_concepts = num_concepts

        # Concept attractors stored as sparse integer index tables [num_concepts, k_active]
        self.register_buffer(
            "concept_bank",
            torch.randint(0, total_dim, (num_concepts, k_active), dtype=torch.int32)
        )

        # Lateral inhibition mask between concept attractors
        self.register_buffer("inhibition_mask", torch.eye(num_concepts) * -2.0)

    def step_bifurcation(self, state: SparseCognitiveState, mu_context: float) -> SparseCognitiveState:
        """
        [Discrete Saddle-Node Bifurcation]
        mu_context < 0.5 (Fluid Exploration Phase): k_active remains large (fluid/exploratory).
        mu_context >= 0.5 (Decision Phase): k_active shrinks (attractor collapse/decision).
        """
        if mu_context > 0.5:
            # Strong convergence / decision collapse
            current_k = max(10, int(self.k_active * (1.0 - (mu_context - 0.5))))
        else:
            # Fluid exploration phase
            current_k = self.k_active

        new_state = SparseCognitiveState(self.total_dim, current_k, device=state.device)
        new_state.active_indices = state.active_indices[:current_k].clone()
        new_state.phases = state.phases[:current_k].clone()
        return new_state

    def recall_attractor(self, state: SparseCognitiveState) -> Tuple[int, int]:
        """
        [Discrete Attractor Recall & Winner-Take-All (k-WTA)]
        Computes sparse index set intersection with concept attractors in O(num_concepts).
        Returns:
            winner_concept_id (int): Index of winning concept.
            match_count (int): Number of overlapping active nodes.
        """
        state_idx = state.active_indices.unsqueeze(0)  # [1, k]
        bank_idx = self.concept_bank                   # [num_concepts, k_active]

        # Compute intersection match count
        matches = torch.isin(bank_idx, state_idx).sum(dim=-1)  # [num_concepts]

        winner_concept_id = torch.argmax(matches).item()
        match_count = matches[winner_concept_id].item()
        return winner_concept_id, match_count

    def hyperdimensional_bind(self, state1: SparseCognitiveState, state2: SparseCognitiveState) -> SparseCognitiveState:
        """
        [Hyperdimensional Binding (VSA / HDC)]
        Binds two sparse cognitive states using modular index addition (Bitwise/XOR equivalent)
        and quantized phase rotary shifts.
        """
        k = min(len(state1.active_indices), len(state2.active_indices))
        bound_state = SparseCognitiveState(self.total_dim, k, device=state1.device)

        # Modular index addition for binding
        bound_indices = (state1.active_indices[:k] + state2.active_indices[:k]) % self.total_dim

        # Phase rotary shift (uint8 wrapping)
        bound_phases = ((state1.phases[:k].to(torch.int32) + state2.phases[:k].to(torch.int32)) % 256).to(torch.uint8)

        bound_state.active_indices = bound_indices.to(torch.int32)
        bound_state.phases = bound_phases.to(torch.uint8)
        return bound_state

    def step_autopoiesis(self, state: SparseCognitiveState, noise_level: float = 0.05) -> SparseCognitiveState:
        """
        [Zero-Input Autopoietic Maintenance Loop (x = 0)]
        Maintains internal cognitive state vitality and self-boundary without external input (x=0).
        Applies internal phase oscillation and controlled index mutation to prevent state decay or blowup.
        """
        next_state = state.clone()
        k = len(next_state.active_indices)

        # Internal phase dynamic oscillation
        phase_delta = torch.randint(-5, 6, (k,), dtype=torch.int16, device=state.device)
        next_state.phases = ((next_state.phases.to(torch.int16) + phase_delta) % 256).to(torch.uint8)

        # Spontaneous low-rate internal association (autopoietic mutation)
        num_mutations = max(1, int(k * noise_level))
        mutation_positions = torch.randperm(k, device=state.device)[:num_mutations]
        new_indices = torch.randint(0, self.total_dim, (num_mutations,), dtype=torch.int32, device=state.device)
        next_state.active_indices[mutation_positions] = new_indices

        return next_state
