r"""
Cognitive Node Engine (CognitiveNodeEngine)
===========================================

Implements the 3DGS-inspired 5-tuple Cognitive Node Architecture:
1. Position anchor \mathbf{\mu}_i \in \mathbb{R}^d
2. Effective Scope Tensor \mathbf{\Sigma}_i \in \mathbb{S}_+^d
3. Epistemic Certainty \alpha_i \in [0, 1]
4. Basis Coefficient Matrix \mathbf{K}_i \in \mathbb{R}^{M \times d}
5. Explicit Topological Causal Network \mathcal{T}_i = {(j, \mathcal{R}_{ij}, \lambda_{ij})}

Core Mechanics:
- Locality & Sparsity: Mahalanobis distance bounding query O(k)
- Basis Expansion: Orthogonal basis projection for context adaptive response
- Explicit Topology & Analytic Consistency: \mathcal{L}_{\text{consistency}} computation
- Dynamic Node Lifecycle: Automatic Split and Prune mechanisms based on position gradient moving averages,
  error residuals, and consistency tension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class CognitiveNodeEngine(nn.Module):
    """
    PyTorch implementation of the Cognitive Node Engine.
    Handles explicit cognitive nodes with Mahalanobis bounding queries, basis function responses,
    topological consistency checking, and dynamic Split & Prune algorithms.
    """

    def __init__(
        self,
        num_nodes: int = 100,
        dim: int = 64,
        num_basis: int = 4,
        top_k: int = 8,
        split_threshold: float = 0.5,
        prune_threshold: float = 0.1,
        consistency_threshold: float = 0.8,
    ):
        super().__init__()
        self.dim = dim
        self.num_basis = num_basis
        self.top_k = top_k
        self.split_threshold = split_threshold
        self.prune_threshold = prune_threshold
        self.consistency_threshold = consistency_threshold

        # 1. Node Parameters (5-tuple representation)
        self.mu = nn.Parameter(torch.randn(num_nodes, dim) * 0.5)  # Latent concept centers
        self.log_scale = nn.Parameter(torch.zeros(num_nodes, dim))  # Log precision (inv_scale = exp(-log_scale))
        self.alpha = nn.Parameter(torch.ones(num_nodes) * 2.0)  # Unconstrained epistemic certainty (sigmoid -> [0, 1])
        self.basis_coeff = nn.Parameter(torch.randn(num_nodes, num_basis, dim) * 0.1)  # Basis response coefficients K_i

        # 2. Moving averages for position gradients and accumulated errors
        self.register_buffer("grad_mu_sq_avg", torch.zeros(num_nodes))
        self.register_buffer("consistency_error_accum", torch.zeros(num_nodes))

        # 3. Explicit Topology Edge Matrix: adjacency weight matrix W_topo [N, N] and edge relation matrix R_topo [N, N]
        # W_topo[i, j] represents weight \lambda_{ij}, R_topo[i, j] represents relationship type (-1 for contradiction, +1 for support)
        self.register_buffer("topo_weight", torch.zeros(num_nodes, num_nodes))
        self.register_buffer("topo_relation", torch.zeros(num_nodes, num_nodes))

        # Initialize default sparse topological connections
        self._init_topology_edges(num_nodes)

    def _init_topology_edges(self, num_nodes: int):
        """Initializes random sparse causal/logical constraint relations."""
        with torch.no_grad():
            prob_edge = min(0.1, 10.0 / max(num_nodes, 1))
            mask = torch.rand(num_nodes, num_nodes) < prob_edge
            mask.fill_diagonal_(False)
            weights = torch.rand(num_nodes, num_nodes) * mask.float()
            relations = torch.sign(torch.randn(num_nodes, num_nodes)) * mask.float()

            # Symmetric / antisymmetric topology
            self.topo_weight.copy_(weights)
            self.topo_relation.copy_(relations)

    @property
    def current_num_nodes(self) -> int:
        return self.mu.shape[0]

    def forward(
        self,
        x: torch.Tensor,
        context_dir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Forward evaluation of cognitive nodes.

        Args:
            x: Context input vector [Batch, Dim]
            context_dir: Basis evaluation direction [Batch, Num_Basis]

        Returns:
            output_state: Synthesized output representation [Batch, Dim]
            topk_indices: Indices of selected active nodes [Batch, Top_K]
            norm_weights: Normalized activation weights [Batch, Top_K]
            consistency_loss: Scalar topological consistency loss \mathcal{L}_{\text{consistency}}
        """
        batch_size = x.shape[0]
        num_nodes = self.current_num_nodes
        effective_k = min(self.top_k, num_nodes)

        # Step 1: Mahalanobis Distance & Locality Bounding Query
        # diff: [Batch, Num_Nodes, Dim]
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)
        # inv_scale: [1, Num_Nodes, Dim] = exp(-log_scale)
        inv_scale = torch.exp(-self.log_scale).unsqueeze(0)
        mahalanobis_sq = torch.sum((diff ** 2) * inv_scale, dim=-1)  # [Batch, Num_Nodes]

        # Certainty \alpha_i \in [0, 1]
        certainty = torch.sigmoid(self.alpha)  # [Num_Nodes]

        # Activation weights w_i(x)
        weights = certainty.unsqueeze(0) * torch.exp(-0.5 * mahalanobis_sq)  # [Batch, Num_Nodes]

        # Top-K Sparse Indexing
        topk_weights, topk_indices = torch.topk(weights, k=effective_k, dim=-1)  # [Batch, Top_K]

        # Step 2: Basis Expansion
        # Gather coefficients for top-k selected nodes
        # Selected Coeffs: [Batch, Top_K, Num_Basis, Dim]
        selected_coeffs = self.basis_coeff[topk_indices]

        # Basis Evaluation (linear combination)
        # context_dir: [Batch, Num_Basis] -> [Batch, 1, Num_Basis, 1]
        basis_weights = context_dir.unsqueeze(1).unsqueeze(-1)
        # node_states: [Batch, Top_K, Dim]
        node_states = torch.sum(selected_coeffs * basis_weights, dim=2)

        # Step 3: Node State Synthesis
        norm_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)  # [Batch, Top_K]
        output_state = torch.sum(node_states * norm_weights.unsqueeze(-1), dim=1)  # [Batch, Dim]

        # Step 4: Topological Consistency & Contradiction Evaluation
        consistency_loss = self.compute_topological_consistency_loss(topk_indices, norm_weights, node_states)

        return output_state, topk_indices, norm_weights, consistency_loss

    def compute_topological_consistency_loss(
        self,
        topk_indices: torch.Tensor,
        norm_weights: torch.Tensor,
        node_states: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes explicit topological consistency loss \mathcal{L}_{\text{consistency}}.
        If two nodes i, j are simultaneously active with contradiction relation (R_ij = -1),
        their similarity creates topological contradiction tension.
        """
        batch_size, k, dim = node_states.shape
        loss = torch.tensor(0.0, device=node_states.device)

        if k <= 1:
            return loss

        # Pairwise inner products between active node states
        # [Batch, K, Dim] x [Batch, Dim, K] -> [Batch, K, K]
        sim_matrix = torch.bmm(F.normalize(node_states, dim=-1), F.normalize(node_states, dim=-1).transpose(1, 2))

        for b in range(batch_size):
            indices_b = topk_indices[b]  # [K]
            weights_b = norm_weights[b]  # [K]

            # Gather submatrix of topological weights & relations for active nodes
            sub_w = self.topo_weight[indices_b][:, indices_b]  # [K, K]
            sub_r = self.topo_relation[indices_b][:, indices_b]  # [K, K]

            # Contradiction tension penalty where relation is negative (R_ij < 0)
            contradiction_mask = (sub_r < 0).float()
            pair_weight = weights_b.unsqueeze(1) * weights_b.unsqueeze(0)  # [K, K]

            # Excess similarity under contradiction: max(0, sim_matrix - allowed_threshold)
            tension = F.relu(sim_matrix[b]) * contradiction_mask * sub_w * pair_weight
            loss = loss + tension.sum()

            # Accumulate consistency error for active nodes to trigger potential pruning
            with torch.no_grad():
                node_tension = tension.sum(dim=1)
                self.consistency_error_accum[indices_b] += node_tension

        return loss / max(batch_size, 1)

    def update_gradient_moving_averages(self, momentum: float = 0.9):
        """Updates moving average of position gradients to detect nodes needing splitting."""
        if self.mu.grad is not None:
            with torch.no_grad():
                grad_sq = torch.sum(self.mu.grad ** 2, dim=-1)
                self.grad_mu_sq_avg.copy_(momentum * self.grad_mu_sq_avg + (1 - momentum) * grad_sq)

    @torch.no_grad()
    def apply_split_and_prune(
        self,
        gamma_scale: float = 1.5,
        offset_factor: float = 0.2
    ) -> Dict[str, int]:
        r"""
        Executes dynamic Split & Prune algorithm:
        1. Split: Nodes with gradient moving average or position error > split_threshold are duplicated
           and offset in latent space, scales are reduced (\gamma), and certainty is halved.
        2. Prune: Nodes with low certainty < prune_threshold or accumulated consistency error > threshold
           are deactivated / removed.

        Returns:
            Dict containing count of split nodes and pruned nodes.
        """
        num_nodes = self.current_num_nodes
        certainty = torch.sigmoid(self.alpha)

        # Identify candidates for Split and Prune
        split_mask = self.grad_mu_sq_avg > self.split_threshold
        prune_mask = (certainty < self.prune_threshold) | (self.consistency_error_accum > self.consistency_threshold)

        # Avoid splitting nodes that are marked for pruning
        split_mask = split_mask & (~prune_mask)

        num_split = int(split_mask.sum().item())
        num_pruned = int(prune_mask.sum().item())

        if num_split == 0 and num_pruned == 0:
            return {"split": 0, "pruned": 0}

        # Indices to keep
        keep_indices = torch.where(~prune_mask)[0]
        if len(keep_indices) == 0:
            # Prevent emptying all nodes completely
            keep_indices = torch.tensor([0], device=self.mu.device)
            prune_mask[0] = False
            num_pruned -= 1

        # Gather kept parameters
        new_mu = self.mu[keep_indices].clone()
        new_log_scale = self.log_scale[keep_indices].clone()
        new_alpha = self.alpha[keep_indices].clone()
        new_basis_coeff = self.basis_coeff[keep_indices].clone()
        new_grad_mu_sq = self.grad_mu_sq_avg[keep_indices].clone()
        new_consistency_err = self.consistency_error_accum[keep_indices].clone()

        # Update topology matrices for kept nodes
        new_topo_w = self.topo_weight[keep_indices][:, keep_indices].clone()
        new_topo_r = self.topo_relation[keep_indices][:, keep_indices].clone()

        # Perform Splitting
        split_indices_in_kept = torch.where(split_mask[keep_indices])[0]
        if len(split_indices_in_kept) > 0:
            split_mu_list = [new_mu]
            split_log_scale_list = [new_log_scale]
            split_alpha_list = [new_alpha]
            split_basis_coeff_list = [new_basis_coeff]
            split_grad_list = [new_grad_mu_sq]
            split_err_list = [new_consistency_err]

            for idx in split_indices_in_kept:
                # Original node params
                orig_mu = new_mu[idx]
                orig_scale = torch.exp(0.5 * new_log_scale[idx])

                # Offset displacement vector along scope scale
                disp = torch.randn_like(orig_mu) * orig_scale * offset_factor

                # Position split into 2 nodes: mu1 = mu + disp, mu2 = mu - disp
                mu1 = orig_mu + disp
                mu2 = orig_mu - disp

                # Scope reduction: log_scale_new = log_scale + log(gamma)
                log_scale_new = new_log_scale[idx] + torch.log(torch.tensor(gamma_scale, device=self.mu.device))

                # Certainty reset: alpha_new = alpha * 0.5
                alpha_new = new_alpha[idx] * 0.5

                # Update node idx with mu1
                new_mu[idx] = mu1
                new_log_scale[idx] = log_scale_new
                new_alpha[idx] = alpha_new

                # Append second node mu2
                split_mu_list.append(mu2.unsqueeze(0))
                split_log_scale_list.append(log_scale_new.unsqueeze(0))
                split_alpha_list.append(alpha_new.unsqueeze(0))
                split_basis_coeff_list.append(new_basis_coeff[idx].unsqueeze(0) + torch.randn_like(new_basis_coeff[idx]) * 0.01)
                split_grad_list.append(torch.tensor([0.0], device=self.mu.device))
                split_err_list.append(torch.tensor([0.0], device=self.mu.device))

            new_mu = torch.cat(split_mu_list, dim=0)
            new_log_scale = torch.cat(split_log_scale_list, dim=0)
            new_alpha = torch.cat(split_alpha_list, dim=0)
            new_basis_coeff = torch.cat(split_basis_coeff_list, dim=0)
            new_grad_mu_sq = torch.cat(split_grad_list, dim=0)
            new_consistency_err = torch.cat(split_err_list, dim=0)

            # Resize topology matrices to match new node count
            final_n = new_mu.shape[0]
            expanded_topo_w = torch.zeros(final_n, final_n, device=self.mu.device)
            expanded_topo_r = torch.zeros(final_n, final_n, device=self.mu.device)
            old_n = new_topo_w.shape[0]
            expanded_topo_w[:old_n, :old_n] = new_topo_w
            expanded_topo_r[:old_n, :old_n] = new_topo_r

            new_topo_w = expanded_topo_w
            new_topo_r = expanded_topo_r

        # Reassign Parameters
        self.mu = nn.Parameter(new_mu)
        self.log_scale = nn.Parameter(new_log_scale)
        self.alpha = nn.Parameter(new_alpha)
        self.basis_coeff = nn.Parameter(new_basis_coeff)

        # Reset moving averages / buffers
        self.grad_mu_sq_avg = new_grad_mu_sq
        self.consistency_error_accum = new_consistency_err * 0.5  # Decay consistency error
        self.topo_weight = new_topo_w
        self.topo_relation = new_topo_r

        return {"split": num_split, "pruned": num_pruned}
