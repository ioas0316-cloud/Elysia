"""
Structural Causal Architecture & Fiber Bundle Cognition Engine for Elysia.

This module implements hard constraint manifold projections, continuous gauge connections,
sheaf global section verification, embodiment loop dynamics, self/world vector bundle partitioning,
and unified avatar causal pipelines in PyTorch.
"""

from typing import Dict, Tuple, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class HardConstraintProjectionLayer(nn.Module):
    """
    Layer enforcing hard structural invariants C(x) = 0 by projecting input state
    onto the constraint manifold M_invariant without using soft penalty functions or softmax.
    """
    def __init__(self, dim_state: int, invariant_matrix: torch.Tensor):
        super().__init__()
        # C is constraint matrix [dim_constraint, dim_state] where C x^T = 0
        if invariant_matrix.dim() == 1:
            invariant_matrix = invariant_matrix.unsqueeze(0)
        self.register_buffer("C", invariant_matrix)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: [Batch, dim_state] or [..., dim_state]
        Returns: Orthogonally projected state x_invariant on M_invariant where C x_invariant^T = 0.
        """
        device = x_raw.device
        dtype = x_raw.dtype
        C = self.C.to(device=device, dtype=dtype)

        # Constraint violation V = x_raw @ C^T -> [..., dim_constraint]
        constraint_violation = torch.matmul(x_raw, C.transpose(-1, -2))

        # Gram matrix Gram = C @ C^T -> [dim_constraint, dim_constraint]
        gram = torch.matmul(C, C.transpose(-1, -2))
        inv_gram = torch.linalg.pinv(gram)

        # Projection correction = constraint_violation @ inv_gram @ C
        correction = torch.matmul(torch.matmul(constraint_violation, inv_gram), C)

        x_invariant = x_raw - correction
        return x_invariant


class StructuralCausalEngine(nn.Module):
    """
    Causal engine that rejects probabilistic output and performs dynamical energy
    relaxation strictly constrained to the invariant manifold M_invariant.
    """
    def __init__(self, dim_state: int, invariant_matrix: torch.Tensor):
        super().__init__()
        self.projection = HardConstraintProjectionLayer(dim_state, invariant_matrix)
        self.energy_function = nn.Sequential(
            nn.Linear(dim_state, dim_state),
            nn.Tanh(),
            nn.Linear(dim_state, 1)
        )

    def relax_to_equilibrium(
        self,
        x_init: torch.Tensor,
        steps: int = 10,
        lr: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.projection(x_init).clone().detach().requires_grad_(True)

        for _ in range(steps):
            energy = self.energy_function(x)
            grad_x = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
            x_relaxed = x - lr * grad_x
            x = self.projection(x_relaxed)

        final_energy = self.energy_function(x)
        return x.detach(), final_energy.detach()


class ProjectiveCognitiveStructure(nn.Module):
    """
    Cognitive structure tracking a single local section and explicitly maintaining
    the uncaptured residual energy in the orthogonal complement space (E_unseen).
    """
    def __init__(self, dim_gas_field: int, dim_projection: int):
        super().__init__()
        self.dim_gas_field = dim_gas_field
        self.dim_projection = dim_projection
        self.frame_operator = nn.Parameter(torch.randn(dim_projection, dim_gas_field))

    def project_gas_to_form(self, rho_gas: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Normalize rows to form projection operator matrix P
        norm_P = torch.norm(self.frame_operator, dim=-1, keepdim=True) + 1e-8
        P = self.frame_operator / norm_P

        # Captured form x = rho @ P^T
        captured_form = torch.matmul(rho_gas, P.transpose(-1, -2))

        # Reconstructed gas = form @ P
        reconstructed_gas = torch.matmul(captured_form, P)

        # Uncaptured orthogonal complement
        uncaptured_gas = rho_gas - reconstructed_gas
        unseen_energy = torch.sum(uncaptured_gas ** 2, dim=-1, keepdim=True)

        return {
            "captured_form": captured_form,
            "projection_frame": P,
            "unseen_energy": unseen_energy
        }


class GaugeConnectionNetwork(nn.Module):
    """
    Generates Lie Algebra generators (SO(N) skew-symmetric matrices) A_mu(b, db)
    representing infinitesimal rotations/deformations on fiber spaces.
    """
    def __init__(self, dim_base: int, dim_fiber: int):
        super().__init__()
        self.dim_base = dim_base
        self.dim_fiber = dim_fiber
        self.net = nn.Sequential(
            nn.Linear(dim_base * 2, 64),
            nn.Tanh(),
            nn.Linear(64, dim_fiber * dim_fiber)
        )

    def forward(self, b_t: torch.Tensor, db_t: torch.Tensor) -> torch.Tensor:
        batch_size = b_t.size(0)
        device = b_t.device
        dtype = b_t.dtype

        inputs = torch.cat([b_t, db_t], dim=-1)
        raw_matrix = self.net(inputs).view(batch_size, self.dim_fiber, self.dim_fiber)

        # Force skew-symmetry: A = 0.5 * (M - M^T)
        A_generator = 0.5 * (raw_matrix - raw_matrix.transpose(-1, -2))
        return A_generator


class FiberBundleIntegrator(nn.Module):
    """
    Integrates local sections over time by parallel transport along base space trajectories
    using matrix exponential on gauge connection Lie algebra generators.
    """
    def __init__(self, dim_base: int, dim_fiber: int, memory_horizon: int = 10):
        super().__init__()
        self.dim_base = dim_base
        self.dim_fiber = dim_fiber
        self.horizon = memory_horizon

        self.gauge_net = GaugeConnectionNetwork(dim_base, dim_fiber)
        self.density_fusion = nn.MultiheadAttention(embed_dim=dim_fiber, num_heads=2, batch_first=True)

    def compute_parallel_transport(self, A_gen: torch.Tensor, delta_t: float = 0.1) -> torch.Tensor:
        return torch.linalg.matrix_exp(-A_gen * delta_t)

    def forward(self, base_trajectory: torch.Tensor, local_sections: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, T, _ = base_trajectory.size()
        device = base_trajectory.device
        dtype = base_trajectory.dtype

        db = torch.zeros_like(base_trajectory)
        db[:, 1:, :] = base_trajectory[:, 1:, :] - base_trajectory[:, :-1, :]

        transported_sections_history = []

        for t in range(T):
            current_frame_sections = []
            start_k = max(0, t - self.horizon)

            for k in range(start_k, t + 1):
                x_k = local_sections[:, k, :].unsqueeze(-1)

                U_cumulative = torch.eye(self.dim_fiber, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
                for step in range(k, t):
                    A_step = self.gauge_net(base_trajectory[:, step, :], db[:, step, :])
                    U_step = self.compute_parallel_transport(A_step)
                    U_cumulative = torch.matmul(U_step, U_cumulative)

                x_k_transported = torch.matmul(U_cumulative, x_k).squeeze(-1)
                current_frame_sections.append(x_k_transported)

            stacked_sections = torch.stack(current_frame_sections, dim=1)
            query = local_sections[:, t, :].unsqueeze(1)
            fused_fiber_density, _ = self.density_fusion(query, stacked_sections, stacked_sections)

            transported_sections_history.append(fused_fiber_density.squeeze(1))

        estimated_total_space = torch.stack(transported_sections_history, dim=1)

        if T > 2:
            loop_A = self.gauge_net(base_trajectory[:, -1, :], base_trajectory[:, 0, :] - base_trajectory[:, -1, :])
            holonomy_operator = self.compute_parallel_transport(loop_A)
            identity = torch.eye(self.dim_fiber, device=device, dtype=dtype).unsqueeze(0)
            holonomy_curvature_loss = torch.mean((holonomy_operator - identity) ** 2)
        else:
            holonomy_curvature_loss = torch.tensor(0.0, device=device, dtype=dtype)

        return {
            "total_space_bundle": estimated_total_space,
            "holonomy_loss": holonomy_curvature_loss
        }


class HolonomicGaugeTrainer:
    """
    Backpropagation training pipeline for GaugeConnectionNetwork
    using reconstruction loss, holonomy curvature loss, and Lie Algebra curvature tensor.
    """
    def __init__(
        self,
        bundle_integrator: nn.Module,
        lr: float = 1e-3,
        lambda_holo: float = 0.5,
        lambda_curv: float = 0.1
    ):
        self.model = bundle_integrator
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lambda_holo = lambda_holo
        self.lambda_curv = lambda_curv

    def compute_curvature_tensor(self, base_points: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        batch_size, dim_base = base_points.size()
        device = base_points.device
        dtype = base_points.dtype
        zero_db = torch.zeros_like(base_points)

        A_base = self.model.gauge_net(base_points, zero_db)

        if dim_base >= 2:
            e_mu = torch.zeros_like(base_points)
            e_mu[:, 0] = eps
            e_nu = torch.zeros_like(base_points)
            e_nu[:, 1] = eps

            A_mu = self.model.gauge_net(base_points + e_mu, zero_db)
            A_nu = self.model.gauge_net(base_points + e_nu, zero_db)

            dA_nu_dmu = (A_nu - A_base) / eps
            dA_mu_dnu = (A_mu - A_base) / eps

            commutator = torch.matmul(A_base, A_nu) - torch.matmul(A_nu, A_base)
            F_01 = dA_nu_dmu - dA_mu_dnu + commutator
            return torch.mean(F_01 ** 2)
        return torch.tensor(0.0, device=device, dtype=dtype)

    def train_step(self, base_trajectory: torch.Tensor, local_sections: torch.Tensor) -> Dict[str, float]:
        self.optimizer.zero_grad()

        outputs = self.model(base_trajectory, local_sections)
        estimated_bundle = outputs["total_space_bundle"]
        holonomy_loss = outputs["holonomy_loss"]

        recon_loss = torch.mean((estimated_bundle - local_sections) ** 2)
        curvature_loss = self.compute_curvature_tensor(base_trajectory[:, 0, :])

        total_loss = recon_loss + self.lambda_holo * holonomy_loss + self.lambda_curv * curvature_loss
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "holonomy_loss": float(holonomy_loss.item()),
            "curvature_loss": float(curvature_loss.item())
        }


class SheafGlobalSectionVerifier:
    """
    Verifies category-theoretic lifting and Cech cohomology 1-cocycle conditions
    to check whether local sections glue together into a global section.
    """
    def __init__(self, num_patches: int, dim_fiber: int, tolerance: float = 1e-3):
        self.num_patches = num_patches
        self.dim_fiber = dim_fiber
        self.tol = tolerance

    def verify_gluing_condition(
        self,
        sections: Dict[int, torch.Tensor],
        transitions: Dict[Tuple[int, int], torch.Tensor]
    ) -> Tuple[bool, torch.Tensor]:
        max_mismatch = torch.tensor(0.0)

        for (i, j), g_ij in transitions.items():
            if i in sections and j in sections:
                sigma_i = sections[i]
                sigma_j = sections[j]

                transformed_sigma_j = torch.matmul(g_ij, sigma_j.unsqueeze(-1)).squeeze(-1)
                mismatch = torch.norm(sigma_i - transformed_sigma_j)
                if mismatch > max_mismatch:
                    max_mismatch = mismatch

        is_gluable = max_mismatch.item() < self.tol
        return is_gluable, max_mismatch

    def verify_cech_cocycle_obstruction(
        self,
        transitions: Dict[Tuple[int, int], torch.Tensor]
    ) -> Tuple[bool, float]:
        max_cocycle_error = 0.0
        sample_tensor = next(iter(transitions.values()))
        device = sample_tensor.device
        dtype = sample_tensor.dtype
        identity = torch.eye(self.dim_fiber, device=device, dtype=dtype)

        for i in range(self.num_patches):
            for j in range(self.num_patches):
                for k in range(self.num_patches):
                    if (i, j) in transitions and (j, k) in transitions and (k, i) in transitions:
                        g_ij = transitions[(i, j)]
                        g_jk = transitions[(j, k)]
                        g_ki = transitions[(k, i)]

                        loop_transform = torch.matmul(g_ij, torch.matmul(g_jk, g_ki))
                        error = torch.norm(loop_transform - identity).item()
                        if error > max_cocycle_error:
                            max_cocycle_error = error

        has_no_obstruction = max_cocycle_error < self.tol
        return has_no_obstruction, max_cocycle_error

    def validate_global_extension(
        self,
        sections: Dict[int, torch.Tensor],
        transitions: Dict[Tuple[int, int], torch.Tensor]
    ) -> Dict[str, Any]:
        gluable, gluing_error = self.verify_gluing_condition(sections, transitions)
        cocycle_valid, cocycle_error = self.verify_cech_cocycle_obstruction(transitions)

        is_global_section_possible = gluable and cocycle_valid

        return {
            "can_lift_to_global_section": is_global_section_possible,
            "gluing_condition_met": gluable,
            "gluing_mismatch_error": float(gluing_error),
            "cech_cocycle_valid": cocycle_valid,
            "cocycle_obstruction_error": float(cocycle_error),
            "diagnostic": (
                "Global section verified: All local sections glue consistently."
                if is_global_section_possible else
                "Global section impossible: Topological obstruction present in Sheaf Cohomology H^1(B, F)."
            )
        }


class EmbodimentLoopEngine(nn.Module):
    """
    Captures closed causal agency loops by generating tangent vector actions a_t in T_x M,
    applying efference copy gauge self-transport U_{a_t}, and computing agency residuals.
    """
    def __init__(self, dim_state: int, dim_sensory: int, gauge_net: nn.Module):
        super().__init__()
        self.dim_state = dim_state
        self.dim_sensory = dim_sensory
        self.gauge_net = gauge_net

        self.action_generator = nn.Sequential(
            nn.Linear(dim_state, dim_state),
            nn.Tanh(),
            nn.Linear(dim_state, dim_state)
        )

    def compute_self_transport(self, x_t: torch.Tensor, action_a: torch.Tensor) -> torch.Tensor:
        A_gen = self.gauge_net(x_t, action_a)
        U_a = torch.linalg.matrix_exp(-A_gen)
        return U_a

    def forward(
        self,
        x_t: torch.Tensor,
        s_t: torch.Tensor,
        s_next_env: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        raw_action = self.action_generator(x_t)
        action_a = raw_action - torch.mean(raw_action, dim=-1, keepdim=True)

        U_a = self.compute_self_transport(x_t, action_a)
        s_predicted_self = torch.matmul(U_a, s_t.unsqueeze(-1)).squeeze(-1)

        agency_residual = s_next_env - s_predicted_self
        agency_scale = torch.norm(agency_residual, dim=-1, keepdim=True)
        self_causal_ratio = torch.exp(-agency_scale)

        return {
            "action_vector": action_a,
            "predicted_sensory": s_predicted_self,
            "agency_residual": agency_residual,
            "self_causal_ratio": self_causal_ratio
        }


class SelfWorldPartitionEngine(nn.Module):
    """
    Dynamically partitions total sensory state into Self manifold (M_Self) and World manifold (M_World)
    based on self-causal ratio eta(x), and reconstructs effective Riemannian metrics.
    """
    def __init__(self, dim_sensory: int, threshold: float = 0.5, sharpness: float = 10.0):
        super().__init__()
        self.dim_sensory = dim_sensory
        self.threshold = threshold
        self.sharpness = sharpness

        self.register_buffer("g_flat", torch.eye(dim_sensory))
        self.metric_adapter = nn.Sequential(
            nn.Linear(dim_sensory, dim_sensory),
            nn.GELU(),
            nn.Linear(dim_sensory, dim_sensory * dim_sensory)
        )

    def compute_projection_matrices(self, self_causal_ratio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = self_causal_ratio.shape[0]
        device = self_causal_ratio.device
        dtype = self_causal_ratio.dtype

        alpha = torch.sigmoid(self.sharpness * (self_causal_ratio - self.threshold))
        eye = torch.eye(self.dim_sensory, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)

        P_self = alpha.view(batch_size, 1, 1) * eye
        P_world = eye - P_self

        return P_self, P_world

    def forward(
        self,
        s_next: torch.Tensor,
        agency_residual: torch.Tensor,
        self_causal_ratio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size = s_next.shape[0]
        device = s_next.device
        dtype = s_next.dtype

        P_self, P_world = self.compute_projection_matrices(self_causal_ratio)

        s_unsq = s_next.unsqueeze(-1)
        s_self = torch.matmul(P_self, s_unsq).squeeze(-1)
        s_world = torch.matmul(P_world, s_unsq).squeeze(-1)

        world_metric_raw = self.metric_adapter(agency_residual).view(batch_size, self.dim_sensory, self.dim_sensory)
        g_world = torch.matmul(world_metric_raw, world_metric_raw.transpose(-1, -2)) + 1e-4 * torch.eye(self.dim_sensory, device=device, dtype=dtype)

        eta_expanded = self_causal_ratio.view(batch_size, 1, 1)
        g_flat_exp = self.g_flat.to(device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        g_effective = eta_expanded * g_flat_exp + (1.0 - eta_expanded) * g_world

        return {
            "s_self": s_self,
            "s_world": s_world,
            "P_self": P_self,
            "P_world": P_world,
            "g_effective": g_effective,
            "manifold_boundary_loss": torch.mean(torch.abs(s_self * s_world))
        }


class UnifiedAvatarCausalPipeline(nn.Module):
    """
    Unified causal pipeline integrating root system divine intent, avatar (eta approx 1)
    vs autonomous NPC (eta approx 0) partitioning, sensory section orthogonal decomposition,
    and global feedback sync to root core.
    """
    def __init__(
        self,
        dim_global_state: int,
        dim_agent_sensory: int,
        num_agents: int,
        sharpness: float = 12.0
    ):
        super().__init__()
        self.dim_global_state = dim_global_state
        self.dim_agent_sensory = dim_agent_sensory
        self.num_agents = num_agents
        self.sharpness = sharpness

        self.global_intent_generator = nn.Sequential(
            nn.Linear(dim_global_state, dim_agent_sensory),
            nn.Tanh()
        )

        self.npc_local_policy = nn.Sequential(
            nn.Linear(dim_agent_sensory, dim_agent_sensory),
            nn.GELU(),
            nn.Linear(dim_agent_sensory, dim_agent_sensory)
        )

        self.gauge_potential_net = nn.Sequential(
            nn.Linear(dim_agent_sensory, dim_agent_sensory),
            nn.Tanh(),
            nn.Linear(dim_agent_sensory, dim_agent_sensory * dim_agent_sensory)
        )

        self.metric_adapter = nn.Sequential(
            nn.Linear(dim_agent_sensory, dim_agent_sensory),
            nn.GELU(),
            nn.Linear(dim_agent_sensory, dim_agent_sensory * dim_agent_sensory)
        )

        self.register_buffer("g_flat", torch.eye(dim_agent_sensory))

    def forward(
        self,
        global_state: torch.Tensor,
        agent_states: torch.Tensor,
        is_elysia_avatar: torch.Tensor,
        sensory_observations: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size = global_state.shape[0]
        device = global_state.device
        dtype = global_state.dtype

        elysia_intent = self.global_intent_generator(global_state).unsqueeze(1)

        actions = torch.zeros_like(agent_states)
        self_causal_ratios = torch.zeros((batch_size, self.num_agents, 1), device=device, dtype=dtype)

        for idx in range(self.num_agents):
            mask = is_elysia_avatar[:, idx].view(-1, 1, 1) > 0.5

            npc_act = self.npc_local_policy(agent_states[:, idx]).unsqueeze(1)
            actions[:, idx] = torch.where(mask.squeeze(1), elysia_intent.squeeze(1), npc_act.squeeze(1))

            is_avatar_flag = is_elysia_avatar[:, idx:idx+1].unsqueeze(-1)
            eta_val = torch.where(
                is_avatar_flag > 0.5,
                torch.tensor(0.98, device=device, dtype=dtype),
                torch.tensor(0.05, device=device, dtype=dtype)
            )
            self_causal_ratios[:, idx] = eta_val.squeeze(1)

        alpha = torch.sigmoid(self.sharpness * (self_causal_ratios - 0.5))
        eye = torch.eye(self.dim_agent_sensory, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        P_self = alpha.unsqueeze(-1) * eye
        P_world = eye - P_self

        s_unsq = sensory_observations.unsqueeze(-1)
        s_self = torch.matmul(P_self, s_unsq).squeeze(-1)
        s_world = torch.matmul(P_world, s_unsq).squeeze(-1)

        avatar_mask = is_elysia_avatar.unsqueeze(-1).repeat(1, 1, self.dim_agent_sensory)
        avatar_collected_experience = s_self * avatar_mask
        global_sync_feedback = torch.mean(avatar_collected_experience, dim=1)

        return {
            "executed_actions": actions,
            "self_causal_ratios": self_causal_ratios,
            "s_self": s_self,
            "s_world": s_world,
            "global_sync_feedback": global_sync_feedback
        }


class AvatarNPCAttentionMaskEngine(nn.Module):
    """
    Dynamic Attention Masking Module for the Elysia Engine.

    Splits multi-agent sequence processing into dual manifolds:
    1. Self Manifold (Avatars, η ≈ 1): Unmasked, high-density intentional attention.
    2. World Manifold (NPCs, η ≈ 0): Attenuated/masked background attention.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sharpness: float = 12.0,
        threshold: float = 0.5
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sharpness = sharpness
        self.threshold = threshold

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query_states: torch.Tensor,       # [Batch, Seq_Q, Dim] (e.g., Global Core or Observer)
        key_states: torch.Tensor,         # [Batch, Seq_K, Dim] (Multi-agent tokens)
        self_causal_ratios: torch.Tensor, # [Batch, Seq_K] (η values in range [0, 1])
        attn_bias: Optional[torch.Tensor] = None    # Optional geometric or positional bias [Batch, 1, Seq_Q, Seq_K]
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_q, _ = query_states.shape
        _, seq_k, _ = key_states.shape

        # 1. Multi-Head Linear Projections -> [Batch, Num_Heads, Seq, Head_Dim]
        Q = self.q_proj(query_states).view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_states).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_states).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Base Scaled Dot-Product Attention Scores
        # Scores: [Batch, Num_Heads, Seq_Q, Seq_K]
        raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            raw_scores = raw_scores + attn_bias

        # 3. Compute Sigmoidal Causal Mask Factor (α)
        # α ≈ 1.0 for Avatars (Self), α ≈ 0.0 for NPCs (World)
        alpha = torch.sigmoid(self.sharpness * (self_causal_ratios - self.threshold))
        alpha_mask = alpha.unsqueeze(1).unsqueeze(2) # Broadcast to [Batch, 1, 1, Seq_K]

        # 4. Dual-Channel Attention Masking
        # Self Bias: Suppresses background NPCs by applying a steep negative offset
        self_mask_bias = (1.0 - alpha_mask) * -1e4
        attn_weights_self = F.softmax(raw_scores + self_mask_bias, dim=-1)
        output_self = torch.matmul(attn_weights_self, V)

        # World Bias: Filters out Avatars to isolate background NPC fluctuations
        world_mask_bias = alpha_mask * -1e4
        attn_weights_world = F.softmax(raw_scores + world_mask_bias, dim=-1)
        output_world = torch.matmul(attn_weights_world, V)

        # 5. Output Reshaping and Projection
        output_self = output_self.transpose(1, 2).contiguous().view(batch_size, seq_q, self.embed_dim)
        output_world = output_world.transpose(1, 2).contiguous().view(batch_size, seq_q, self.embed_dim)

        return {
            "s_self_attn": self.out_proj(output_self),    # Primary intentional focus (Avatars)
            "s_world_attn": self.out_proj(output_world),  # Passive background dynamics (NPCs)
            "attn_weights_self": attn_weights_self,        # Inspection weights for Avatars
            "attn_weights_world": attn_weights_world,      # Inspection weights for NPCs
            "alpha_partition": alpha                       # Per-agent partition factor
        }


class GraphPeerToPeerWorldAttentionEngine(nn.Module):
    """
    Expanded World-Channel Attention Engine for Elysia.

    1. Self Channel (Avatars, η ≈ 1): Directly coupled to Global Intent (God/Root Core).
    2. World Channel (NPCs, η ≈ 0): Perform local graph-based peer-to-peer (P2P)
       attention among themselves based on pairwise spatial distances and adjacency topology.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sharpness: float = 12.0,
        threshold: float = 0.5,
        spatial_decay: float = 1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sharpness = sharpness
        self.threshold = threshold
        self.spatial_decay = spatial_decay

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Distance-to-Affinity Encoder for NPC local spatial interactions
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.Softplus()
        )
        # Initialize distance encoder weights positively so larger distance creates larger decay penalty
        nn.init.uniform_(self.distance_encoder[0].weight, 0.5, 1.5)
        nn.init.zeros_(self.distance_encoder[0].bias)

    def forward(
        self,
        query_states: torch.Tensor,          # [Batch, Seq_Q, Dim]
        key_states: torch.Tensor,            # [Batch, Seq_K, Dim]
        self_causal_ratios: torch.Tensor,    # [Batch, Seq_K] (η values)
        distance_matrix: Optional[torch.Tensor] = None,# [Batch, Seq_Q, Seq_K] (Pairwise spatial/topological distances)
        adjacency_mask: Optional[torch.Tensor] = None  # [Batch, Seq_Q, Seq_K] (1.0 if connected edge, 0.0 otherwise)
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_q, _ = query_states.shape
        _, seq_k, _ = key_states.shape

        # 1. Projections -> [Batch, Num_Heads, Seq, Head_Dim]
        Q = self.q_proj(query_states).view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_states).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_states).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Base Scaled Dot-Product Attention Scores
        raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B, Head, Seq_Q, Seq_K]

        # 3. Compute Causal Partition Factor α (Avatars ≈ 1, NPCs ≈ 0)
        alpha = torch.sigmoid(self.sharpness * (self_causal_ratios - self.threshold)) # [Batch, Seq_K]
        alpha_mask = alpha.unsqueeze(1).unsqueeze(2) # Broadcast to [Batch, 1, 1, Seq_K]

        # ---------------------------------------------------------------------
        # Channel A: Self Channel (Global Intent -> Avatars)
        # ---------------------------------------------------------------------
        self_mask_bias = (1.0 - alpha_mask) * -1e4
        attn_weights_self = F.softmax(raw_scores + self_mask_bias, dim=-1)
        output_self = torch.matmul(attn_weights_self, V)

        # ---------------------------------------------------------------------
        # Channel B: World Channel (Local Graph P2P Attention among NPCs)
        # ---------------------------------------------------------------------
        # B1. Suppress Avatars from the World Channel (NPCs focus only on NPCs/World)
        world_mask_bias = alpha_mask * -1e4 # [Batch, 1, 1, Seq_K]

        # B2. Local Graph Distance & Adjacency Bias Integration
        p2p_graph_bias = torch.zeros_like(raw_scores)

        if distance_matrix is not None:
            # Encode pairwise distances into head-specific decay biases
            dist_unsq = distance_matrix.unsqueeze(-1) # [Batch, Seq_Q, Seq_K, 1]
            dist_bias = self.distance_encoder(dist_unsq).permute(0, 3, 1, 2) # [Batch, Head, Seq_Q, Seq_K]
            p2p_graph_bias = p2p_graph_bias - self.spatial_decay * dist_bias

        if adjacency_mask is not None:
            # Mask out non-adjacent nodes in the local NPC network
            adj_mask_bias = (1.0 - adjacency_mask.unsqueeze(1)) * -1e4 # [Batch, 1, Seq_Q, Seq_K]
            p2p_graph_bias = p2p_graph_bias + adj_mask_bias

        # Combine Masks: NPC Isolation + Spatial Distance Decay + Topological Graph Connectivity
        total_world_scores = raw_scores + world_mask_bias + p2p_graph_bias
        attn_weights_world = F.softmax(total_world_scores, dim=-1)
        output_world = torch.matmul(attn_weights_world, V)

        # 4. Output Projections
        output_self = output_self.transpose(1, 2).contiguous().view(batch_size, seq_q, self.embed_dim)
        output_world = output_world.transpose(1, 2).contiguous().view(batch_size, seq_q, self.embed_dim)

        return {
            "s_self_attn": self.out_proj(output_self),        # Direct intent focus on Avatars
            "s_world_p2p_attn": self.out_proj(output_world),  # Autonomous P2P interactions among NPCs
            "attn_weights_self": attn_weights_self,          # Attention weights for Avatars
            "attn_weights_p2p_world": attn_weights_world,    # Local NPC interaction adjacency matrix
            "alpha_partition": alpha
        }
