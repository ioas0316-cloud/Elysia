"""
Elysia Core Engine: Internal Metric Field Phase-Lock Engine (PyTorch Module)
==========================================================================
Unified structural phase transition engine modeling Solid, Liquid, and Gas states
without individual physical formulas.

State Variables:
1. Constraint Tension (C): Pairwise restorative force and coupling strength.
2. Topological Mobility (M): Pairwise freedom of nodes to rearrange neighbor relationships.
3. Phase-Lock Ratio (Phi): Phi = C / (M + epsilon) - dimensionless state index.
   - Solid: Phi >= Phi_solid (Fixed Topological Lock, A_ij = 1.0)
   - Liquid: Phi_gas <= Phi < Phi_solid (Dynamic Neighboring Field, A_ij = sigmoid(Phi))
   - Gas: Phi < Phi_gas (Free Scatter Field, Unbound, A_ij = 0.0)

Includes both Dense O(N^2) tensor evolution and Sparse O(N*K) edge list processing,
as well as Macro-Node Early Contraction for multi-scale aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List


class PhaseLockEngine(nn.Module):
    """
    Internal Metric Field Phase-Lock Engine (PyTorch Module).
    Simulates solid-liquid-gas phase transitions on an N-node observational manifold.
    """
    def __init__(
        self,
        num_nodes: int,
        feature_dim: int = 3,
        phi_solid: float = 5.0,
        phi_gas: float = 0.2,
        gamma_m: float = 0.1,
        alpha: float = 0.5,
        beta: float = 1.0,
        tau_c: float = 3.0,
        c0: float = 1.0,
        lambda_c: float = 0.5,
        r_cut: float = 2.5,
        eps: float = 1e-5
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.phi_solid = float(phi_solid)
        self.phi_gas = float(phi_gas)
        self.gamma_m = float(gamma_m)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau_c = float(tau_c)
        self.c0 = float(c0)
        self.lambda_c = float(lambda_c)
        self.r_cut = float(r_cut)
        self.eps = float(eps)

        # State Tensors initialization (Buffers)
        self.register_buffer("C", torch.ones(num_nodes, num_nodes, dtype=torch.float32))
        self.register_buffer("M", torch.zeros(num_nodes, num_nodes, dtype=torch.float32))

    def reset_states(self, num_nodes: Optional[int] = None, device: Optional[torch.device] = None):
        """Resets or resizes internal state buffers C and M and moves them to device."""
        n = num_nodes or self.num_nodes
        dev = device or self.C.device
        self.num_nodes = n
        self.C = torch.ones(n, n, dtype=torch.float32, device=dev)
        self.M = torch.zeros(n, n, dtype=torch.float32, device=dev)

    def forward(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        dt: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dense O(N^2) Forward Update Step.
        X: (N, d) Node Positions
        V: (N, d) Node Velocities
        Returns:
            Phi: (N, N) Phase-Lock Tensor
            A: (N, N) Dynamic Topological Adjacency Matrix
        """
        N = X.size(0)
        if N != self.C.size(0) or self.C.device != X.device:
            self.reset_states(N, device=X.device)

        # 1. Pairwise Distances (D) & Relative Velocity Magnitudes (V_rel)
        D = torch.cdist(X, X, p=2.0)  # Shape: (N, N)
        V_rel = torch.cdist(V, V, p=2.0)  # Shape: (N, N)

        # 2. Mobility (M) Update
        m_target = self.alpha * V_rel + self.beta / (D + self.eps)
        self.M = (1.0 - self.gamma_m) * self.M + self.gamma_m * m_target

        # 3. Constraint Tension (C) Decay & Recovery
        decay = torch.sigmoid(self.tau_c - self.M)
        self.C = self.C * decay + self.c0 * torch.exp(-self.lambda_c * D)

        # 4. Phase-Lock Index Tensor (Phi)
        Phi = self.C / (self.M + self.eps)

        # 5. Dynamic Topological Adjacency Matrix (A)
        A_solid = (Phi >= self.phi_solid).float()
        A_liquid = ((Phi < self.phi_solid) & (Phi >= self.phi_gas)).float() * torch.sigmoid(Phi)
        A_gas = torch.zeros_like(Phi)

        A = A_solid + A_liquid + A_gas

        return Phi, A

    def forward_sparse(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        row_ptr: torch.Tensor,
        col_idx: torch.Tensor,
        C_edge: torch.Tensor,
        M_edge: torch.Tensor,
        dt: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sparse O(N*K) Forward Update Step using CSR Edge Format.
        X: (N, d) Node Positions
        V: (N, d) Node Velocities
        row_ptr: (N+1,) CSR Row Pointers
        col_idx: (E,) CSR Column Indices
        C_edge: (E,) Edge Constraint Tension Tensor (In/Out)
        M_edge: (E,) Edge Mobility Tensor (In/Out)

        Returns:
            C_edge: (E,) Updated Constraint Tension
            M_edge: (E,) Updated Mobility
            Phi_edge: (E,) Phase-Lock Index
            A_edge: (E,) Topological Adjacency Weight
        """
        device = X.device
        E = col_idx.size(0)
        N = X.size(0)

        row_idx = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.int64),
            row_ptr[1:] - row_ptr[:-1]
        )

        pos_i = X[row_idx]
        pos_j = X[col_idx]
        vel_i = V[row_idx]
        vel_j = V[col_idx]

        # 1. Edge Distance and Relative Velocity
        dist_vec = pos_i - pos_j
        dist_sq = torch.sum(dist_vec ** 2, dim=-1) + self.eps
        dist = torch.sqrt(dist_sq)

        vel_diff = vel_i - vel_j
        v_rel = torch.norm(vel_diff, p=2, dim=-1)

        r_cut_sq = self.r_cut ** 2
        out_of_bounds = dist_sq > r_cut_sq

        # 2. Mobility (M_edge) Update
        m_target = self.alpha * v_rel + (self.beta / dist)
        m_val = (1.0 - self.gamma_m) * M_edge + self.gamma_m * m_target
        m_val = torch.where(out_of_bounds, torch.zeros_like(m_val), m_val)

        # 3. Constraint (C_edge) Update
        sigmoid_decay = torch.sigmoid(self.tau_c - m_val)
        c_val = C_edge * sigmoid_decay + self.c0 * torch.exp(-self.lambda_c * dist)
        c_val = torch.where(out_of_bounds, torch.zeros_like(c_val), c_val)

        # In-place buffer update
        M_edge.copy_(m_val)
        C_edge.copy_(c_val)

        # 4. Phase-Lock Index (Phi_edge)
        Phi_edge = c_val / (m_val + self.eps)
        Phi_edge = torch.where(out_of_bounds, torch.zeros_like(Phi_edge), Phi_edge)

        # 5. Dynamic Topological Adjacency (A_edge)
        A_solid = (Phi_edge >= self.phi_solid).float()
        A_liquid = ((Phi_edge < self.phi_solid) & (Phi_edge >= self.phi_gas)).float() * torch.sigmoid(Phi_edge)
        A_edge = A_solid + A_liquid

        return C_edge, M_edge, Phi_edge, A_edge

    def contract_solid_clusters(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        A: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Macro-Node Early Contraction Engine.
        Groups solid-locked node clusters (where Phi >= Phi_solid) into single macro-nodes.

        Returns:
            X_active: (N_active, d) Coordinates of contracted active nodes (macro + remaining)
            V_active: (N_active, d) Velocities of contracted active nodes
            cluster_map: (N,) Cluster ID mapping for each micro-node
        """
        N = X.size(0)
        device = X.device

        # Mask solid edges
        solid_mask = (A >= 1.0).float()

        # Compute connected components via simple BFS/Union-Find
        adj = (solid_mask > 0).cpu().numpy()

        visited = [False] * N
        clusters = []

        for i in range(N):
            if not visited[i]:
                component = []
                queue = [i]
                visited[i] = True
                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    neighbors = (adj[curr] > 0).nonzero()[0]
                    for nxt in neighbors:
                        if not visited[nxt]:
                            visited[nxt] = True
                            queue.append(nxt)
                clusters.append(component)

        cluster_map = torch.zeros(N, dtype=torch.int64, device=device)

        X_active_list = []
        V_active_list = []

        for cid, comp in enumerate(clusters):
            comp_tensor = torch.tensor(comp, dtype=torch.int64, device=device)
            cluster_map[comp_tensor] = cid

            # Compute Center of Mass and Mean Velocity
            x_macro = torch.mean(X[comp_tensor], dim=0)
            v_macro = torch.mean(V[comp_tensor], dim=0)

            X_active_list.append(x_macro)
            V_active_list.append(v_macro)

        X_active = torch.stack(X_active_list, dim=0)
        V_active = torch.stack(V_active_list, dim=0)

        return X_active, V_active, cluster_map

    def get_phase_distribution(self, Phi: torch.Tensor) -> Dict[str, float]:
        """Calculates ratio of Solid, Liquid, and Gas node pairs in the manifold."""
        total_pairs = float(Phi.numel())
        if total_pairs == 0:
            return {"solid_ratio": 0.0, "liquid_ratio": 0.0, "gas_ratio": 0.0}

        solid_mask = (Phi >= self.phi_solid)
        liquid_mask = ((Phi < self.phi_solid) & (Phi >= self.phi_gas))
        gas_mask = (Phi < self.phi_gas)

        return {
            "solid_ratio": float(solid_mask.sum().item() / total_pairs),
            "liquid_ratio": float(liquid_mask.sum().item() / total_pairs),
            "gas_ratio": float(gas_mask.sum().item() / total_pairs),
        }
