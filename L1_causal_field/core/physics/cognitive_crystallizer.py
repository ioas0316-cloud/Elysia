"""
Elysia Physics Module: Cognitive Crystallizer Engine
====================================================
Implements the 4-stage phase transition loop (Gas -> Liquid -> Solid -> Shear)
with 16-byte aligned Spacetime Metric Tensor transformation and Macro-Node aggregation.

Phase States:
- Gas (Gas Phase): High mobility, random floating particle cloud, focus vector injection.
- Liquid (Liquid Phase): Vortex flow, dynamic edge coupling, entropy control, merge/split streams.
- Solid (Solid Phase): Phase-locked threshold Phi >= Phi_solid, SDF surface shell morphing,
  macro-node encapsulation, CSR topology X-Ray projection.
- Shear (Topological Shear & Recycling): Mechanical/topological fracture along weak phase edges,
  deconstruction into liquid/gas particles for re-crystallization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List


class CognitiveCrystallizerEngine(nn.Module):
    """
    Cognitive Crystallizer Engine with Spacetime Metric Tensor transformation.
    """
    def __init__(
        self,
        num_nodes: int = 100,
        feature_dim: int = 3,
        phi_solid: float = 5.0,
        phi_gas: float = 0.2,
        alpha_time: float = 0.5,
        eps: float = 1e-5
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.phi_solid = float(phi_solid)
        self.phi_gas = float(phi_gas)
        self.alpha_time = float(alpha_time)
        self.eps = float(eps)

        # Internal state buffers
        self.register_buffer("C", torch.ones(num_nodes, num_nodes, dtype=torch.float32))
        self.register_buffer("M", torch.zeros(num_nodes, num_nodes, dtype=torch.float32))
        self.register_buffer("phase_angles", torch.rand(num_nodes, dtype=torch.float32) * 2.0 * math.pi)

    def compute_metric_tensors(
        self,
        X: torch.Tensor,
        Phi: torch.Tensor,
        A: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates 3x3 Metric Tensors M_i and Time Dilation factors gamma_i for each node.

        M_i = sum_{j in N(i)} A_ij * sigmoid(Phi_ij) * (u_ij (x) u_ij^T) + eps * I_3
        gamma_i = 1 + alpha_time * (1 - avg_phi_i)^2

        Returns:
            metrics: (N, 3, 3) Metric Tensors
            gammas: (N,) Time Dilation factors
        """
        N = X.size(0)
        device = X.device

        # Pairwise displacement vectors (N, N, 3)
        diff = X.unsqueeze(1) - X.unsqueeze(0)  # X_j - X_i
        dist = torch.norm(diff, dim=-1, keepdim=True) + self.eps
        u = diff / dist  # Unit direction vectors (N, N, 3)

        # Outer product u (x) u^T shape: (N, N, 3, 3)
        u_outer = u.unsqueeze(-1) * u.unsqueeze(-2)

        # Weight factors: A_ij * sigmoid(Phi_ij - 0.5)
        weights = A * torch.sigmoid(Phi - 0.5)  # (N, N)
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (N, N, 1, 1)

        # Weighted sum over neighbors
        M_sum = torch.sum(weights_expanded * u_outer, dim=1)  # (N, 3, 3)

        # Add Identity base metric (N, 3, 3)
        I_3 = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3)
        M_tensor = M_sum + I_3

        # Compute average Phi for each node
        avg_phi = torch.mean(Phi, dim=1)  # (N,)
        avg_phi_norm = torch.clamp(avg_phi / (self.phi_solid + self.eps), 0.0, 1.0)

        # Time dilation: gamma_i = 1 + alpha_time * (1 - avg_phi_norm)^2
        gammas = 1.0 + self.alpha_time * torch.square(1.0 - avg_phi_norm)

        return M_tensor, gammas

    def forward(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        entropy: float = 1.0,
        focus_point: Optional[torch.Tensor] = None,
        focus_intensity: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward step updating node kinematics, phase locks, and metric tensors.
        """
        N = X.size(0)
        device = X.device
        if N != self.C.size(0) or self.C.device != device:
            self.C = torch.ones(N, N, dtype=torch.float32, device=device)
            self.M = torch.zeros(N, N, dtype=torch.float32, device=device)
            self.phase_angles = torch.rand(N, dtype=torch.float32, device=device) * 2.0 * math.pi

        # Apply Focus Vector Injection if present
        if focus_point is not None and focus_intensity > 0.0:
            dir_to_focus = focus_point.unsqueeze(0) - X
            dist_to_focus = torch.norm(dir_to_focus, dim=-1, keepdim=True) + self.eps
            pull_force = (dir_to_focus / dist_to_focus) * focus_intensity
            V = V + pull_force * 0.1

        # 1. Pairwise Distances and Relative Velocities
        D = torch.cdist(X, X, p=2.0)
        V_rel = torch.cdist(V, V, p=2.0)

        # 2. Mobility (M) update scaled by entropy
        m_target = (0.5 * V_rel + 1.0 / (D + self.eps)) * entropy
        self.M = 0.9 * self.M + 0.1 * m_target

        # 3. Constraint Tension (C) update
        decay = torch.sigmoid(3.0 - self.M)
        self.C = self.C * decay + torch.exp(-0.5 * D)

        # 4. Phase-Lock Tensor (Phi)
        Phi = self.C / (self.M + self.eps)

        # 5. Adjacency Matrix (A)
        A_solid = (Phi >= self.phi_solid).float()
        A_liquid = ((Phi < self.phi_solid) & (Phi >= self.phi_gas)).float() * torch.sigmoid(Phi)
        A = A_solid + A_liquid

        # 6. Metric Tensor Transformation & Time Dilation
        M_tensor, gammas = self.compute_metric_tensors(X, Phi, A)

        # Determine phase ratios
        total = float(N * N)
        solid_ratio = float((Phi >= self.phi_solid).sum().item() / total)
        liquid_ratio = float(((Phi < self.phi_solid) & (Phi >= self.phi_gas)).sum().item() / total)
        gas_ratio = float((Phi < self.phi_gas).sum().item() / total)

        return {
            "X": X,
            "V": V,
            "Phi": Phi,
            "A": A,
            "MetricTensor": M_tensor,
            "Gammas": gammas,
            "phase_ratios": {
                "solid": solid_ratio,
                "liquid": liquid_ratio,
                "gas": gas_ratio
            }
        }

    def apply_topological_shear(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        impact_point: torch.Tensor,
        impact_vector: torch.Tensor,
        radius: float = 1.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stage 4: Topological Shear Fracture.
        Applies shear impulse near impact_point, shattering weak phase edges and breaking solid locks.
        """
        dist = torch.norm(X - impact_point.unsqueeze(0), dim=-1)
        mask = (dist <= radius).float().unsqueeze(-1)

        # Scatter velocities along impact_vector and turbulent directions
        turbulent_v = torch.randn_like(V) * 5.0 + impact_vector.unsqueeze(0) * 3.0
        V_new = V * (1.0 - mask) + turbulent_v * mask

        # Reset Constraint Tension around fracture zone
        N = X.size(0)
        fracture_nodes = (dist <= radius).nonzero(as_tuple=True)[0]
        if len(fracture_nodes) > 0:
            self.C[fracture_nodes, :] *= 0.01
            self.C[:, fracture_nodes] *= 0.01
            self.M[fracture_nodes, :] *= 10.0
            self.M[:, fracture_nodes] *= 10.0

        return X, V_new, mask.squeeze(-1)

    def contract_macro_nodes(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        A: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Macro-Node Early Contraction Engine for Solid state encapsulation.
        """
        N = X.size(0)
        device = X.device
        adj = (A >= 1.0).cpu().numpy()

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
        X_macro_list, V_macro_list = [], []

        for cid, comp in enumerate(clusters):
            comp_tensor = torch.tensor(comp, dtype=torch.int64, device=device)
            cluster_map[comp_tensor] = cid

            x_macro = torch.mean(X[comp_tensor], dim=0)
            v_macro = torch.mean(V[comp_tensor], dim=0)

            X_macro_list.append(x_macro)
            V_macro_list.append(v_macro)

        X_macro = torch.stack(X_macro_list, dim=0)
        V_macro = torch.stack(V_macro_list, dim=0)

        return X_macro, V_macro, cluster_map
