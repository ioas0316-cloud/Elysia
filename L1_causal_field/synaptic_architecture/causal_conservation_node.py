r"""
Causal Conservation Node (CC-Node) Engine
==========================================

Implements Causal Conservation Node (CC-Node) dynamics in PyTorch.
Rather than reducing concepts to static text labels or numeric vectors, CC-Node operates
as a dynamic attractor condensed with a force field, topological structure, and generative rules.

3 Structural Layers:
1. Bound Tension Field (V_potential): Internal coupling and potential energy conservation.
2. Topological Boundary Skeleton (I_c): Invariant structural skeleton preserved under scale transformations (C_lens).
3. Generative Rule Engine (Delta): Boundary constraints and rule operator generation.

3 Operational Mechanics:
1. Unfolding: Expanding condensed internal graph into active causal constraints.
2. Tension Resistance: Generating counter-reaction tension (V_react) against context pollution or logical contradictions.
3. Reversible Implosion (SealedAttractor): Lossless state compression into a sealed attractor and exact recovery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any


class BoundTensionField(nn.Module):
    """
    Layer 1: Bound Tension Field (V_potential)
    Maintains internal friction, coupling matrices, and potential energy field.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.coupling_matrix = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.internal_friction = nn.Parameter(torch.ones(dim) * 0.5)

    def compute_potential(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Computes potential tension energy V_potential = 0.5 * state^T * (C + C^T) * state + friction * |state - context|^2.
        """
        sym_coupling = 0.5 * (self.coupling_matrix + self.coupling_matrix.T)
        # [Batch, Dim] x [Dim, Dim] -> [Batch, Dim]
        internal_energy = 0.5 * torch.sum((state @ sym_coupling) * state, dim=-1)
        friction_energy = torch.sum(self.internal_friction * ((state - context) ** 2), dim=-1)
        return internal_energy + friction_energy


class TopologicalBoundarySkeleton(nn.Module):
    """
    Layer 2: Topological Boundary Skeleton (I_c)
    Maintains structural invariants (I_c) across scale lens transformations C_lens.
    """

    init_adj: torch.Tensor

    def __init__(self, dim: int = 64, num_axioms: int = 8):
        super().__init__()
        self.dim = dim
        self.num_axioms = num_axioms
        # Axiomatic interaction graph adjacency
        adj = torch.randn(num_axioms, num_axioms) * 0.2
        adj = 0.5 * (adj + adj.T)
        adj.fill_diagonal_(1.0)
        self.register_buffer("init_adj", adj)
        self.axiom_embeddings = nn.Parameter(torch.randn(num_axioms, dim) * 0.5)

    def compute_invariant(self) -> torch.Tensor:
        """
        Computes structural invariant I_c (eigenvalue spectrum of axiomatic graph).
        Invariant I_c remains preserved regardless of observation scale C_lens.
        """
        eigenvalues = torch.linalg.eigvalsh(self.init_adj)
        return eigenvalues

    def observe_at_scale(self, c_lens_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies observation bandwidth scaling C_lens to the skeleton.
        Returns scaled representation and confirmed structural invariant I_c.
        """
        scaled_embeddings = self.axiom_embeddings * c_lens_scale
        i_c = self.compute_invariant()
        return scaled_embeddings, i_c


class GenerativeRuleEngine(nn.Module):
    """
    Layer 3: Generative Rule Engine (Delta)
    Generates boundary constraints for target domains without overwriting numeric values.
    """

    def __init__(self, dim: int = 64, num_rules: int = 4):
        super().__init__()
        self.dim = dim
        self.num_rules = num_rules
        self.rule_operators = nn.Parameter(torch.randn(num_rules, dim, dim) * 0.1)

    def generate_boundary_constraints(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generates boundary constraint bounds Delta for target domain:
        Delta = ReLU(Op * state)
        """
        # state: [Batch, Dim]
        # rule_operators: [Num_Rules, Dim, Dim]
        # constraints: [Batch, Num_Rules, Dim]
        constraints = torch.einsum('bd,rde->bre', state, self.rule_operators)
        return F.relu(constraints)


class CausalConservationNode(nn.Module):
    """
    Causal Conservation Node (CC-Node) Engine
    Encapsulates 3 structural layers (Bound Tension Field, Topological Skeleton, Generative Rule Engine)
    and 3 engine mechanics (Unfolding, Tension Resistance, Reversible Implosion).
    """

    def __init__(
        self,
        node_id: str,
        dim: int = 64,
        num_axioms: int = 8,
        num_rules: int = 4,
        tension_threshold: float = 0.5,
    ):
        super().__init__()
        self.node_id = node_id
        self.dim = dim
        self.tension_threshold = tension_threshold

        # 3 Structural Layers
        self.tension_field = BoundTensionField(dim=dim)
        self.skeleton = TopologicalBoundarySkeleton(dim=dim, num_axioms=num_axioms)
        self.generative_engine = GenerativeRuleEngine(dim=dim, num_rules=num_rules)

        # Node State
        self.latent_state = nn.Parameter(torch.randn(1, dim) * 0.5)

        # Sealed Attractor vault for reversible implosion
        self.sealed_attractor_state: Optional[Dict[str, torch.Tensor]] = None
        self.is_sealed: bool = False

    def forward(
        self, context: torch.Tensor, c_lens_scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates node response under context stimulus.

        Returns:
            v_potential: Potential energy tensor [Batch]
            v_react: Reaction tension vector [Batch, Dim]
            i_c: Preserved structural invariant spectrum
            constraints: Boundary constraints [Batch, Num_Rules, Dim]
        """
        batch_size = context.shape[0]
        state = self.latent_state.expand(batch_size, -1)

        # 1. Potential tension
        v_potential = self.tension_field.compute_potential(state, context)

        # 2. Reaction tension under context pollution
        v_react, _ = self.compute_tension_resistance(context)

        # 3. Topological invariant I_c at scale C_lens
        _, i_c = self.skeleton.observe_at_scale(c_lens_scale)

        # 4. Boundary constraints
        constraints = self.generative_engine.generate_boundary_constraints(state)

        return v_potential, v_react, i_c, constraints

    def unfold(self, stimulus: torch.Tensor, steps: int = 3) -> torch.Tensor:
        """
        Mechanic 1: Unfolding
        Unrolls condensed internal graph into spatiotemporal causal constraint trajectory.
        """
        batch_size = stimulus.shape[0]
        current_state = self.latent_state.expand(batch_size, -1)
        trajectory = [current_state]

        for t in range(steps):
            constraints = self.generative_engine.generate_boundary_constraints(current_state)
            # Aggregate constraints into state modulation
            modulation = constraints.mean(dim=1)
            next_state = current_state + 0.1 * (stimulus - current_state) + 0.05 * modulation
            trajectory.append(next_state)
            current_state = next_state

        return torch.stack(trajectory, dim=1)  # [Batch, Steps+1, Dim]

    def compute_tension_resistance(self, external_noise: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Mechanic 2: Tension Resistance
        Generates counter-reaction tension vector V_react when外압/논리적 모순이 유입됨.
        Maintains structural invariant I_c intact.
        """
        batch_size = external_noise.shape[0]
        state = self.latent_state.expand(batch_size, -1)

        diff = external_noise - state
        # V_react is proportional to counter-acting force resisting context pollution
        v_react = -1.0 * diff * self.tension_field.internal_friction.unsqueeze(0)
        norm_tension = float(v_react.norm(dim=-1).mean().item())

        return v_react, norm_tension

    def implode_and_seal(self) -> Dict[str, torch.Tensor]:
        """
        Mechanic 3: Reversible Implosion (SealedAttractor)
        Compresses active state into a sealed attractor state without information loss.
        """
        self.sealed_attractor_state = {
            "latent_state": self.latent_state.data.clone(),
            "coupling_matrix": self.tension_field.coupling_matrix.data.clone(),
            "internal_friction": self.tension_field.internal_friction.data.clone(),
            "init_adj": self.skeleton.init_adj.clone(),
            "axiom_embeddings": self.skeleton.axiom_embeddings.data.clone(),
            "rule_operators": self.generative_engine.rule_operators.data.clone(),
        }
        self.is_sealed = True
        return self.sealed_attractor_state

    def unseal_and_recover(self, vault_state: Optional[Dict[str, torch.Tensor]] = None) -> bool:
        """
        Recovers node state reversibly from SealedAttractor vault without loss.
        """
        target_vault = vault_state if vault_state is not None else self.sealed_attractor_state
        if target_vault is None:
            return False

        with torch.no_grad():
            self.latent_state.copy_(target_vault["latent_state"])
            self.tension_field.coupling_matrix.copy_(target_vault["coupling_matrix"])
            self.tension_field.internal_friction.copy_(target_vault["internal_friction"])
            self.skeleton.init_adj.copy_(target_vault["init_adj"])
            self.skeleton.axiom_embeddings.copy_(target_vault["axiom_embeddings"])
            self.generative_engine.rule_operators.copy_(target_vault["rule_operators"])

        self.is_sealed = False
        return True
