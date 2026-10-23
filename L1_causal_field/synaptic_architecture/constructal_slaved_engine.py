import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class ConstructalSlavedModule(nn.Module):
    """
    Constructal Slaved Module (Constructal Law + Haken's Slaving Principle)

    Combines Synergetics' Slaving Principle (Downward Causality / Macro Order Parameter
    dimensional reduction) with Bejan's Constructal Law (dynamic bifurcation /
    streamline routing to minimize computational resistance).
    """
    def __init__(self, in_features: int, out_features: int, num_order_params: int = 4, split_threshold: float = 2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = min(num_order_params, in_features)
        self.split_threshold = split_threshold

        # 1. Slaving Principle: Slow Mode orthogonal projection basis
        self.U_slow = nn.Parameter(torch.randn(in_features, self.k) * (2.0 / in_features)**0.5)

        # Fast mode manifold mapping h_s(q)
        fast_dim = max(1, in_features - self.k)
        self.slaving_manifold = nn.Sequential(
            nn.Linear(self.k, self.k * 2),
            nn.SiLU(),
            nn.Linear(self.k * 2, fast_dim)
        )

        # 2. Constructal Law: Flow channels & dynamic branching
        self.primary_path = nn.Linear(in_features, out_features)
        self.branch_path = None  # Dynamically allocated when stress exceeds threshold

        self.register_buffer("resistance_ema", torch.zeros(1))

    def _apply_slaving(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [Upward Emergence & Downward Causality]
        Extracts macro order parameters q and reconstructs high-dimensional potential state.
        """
        # [Upward Emergence] Extract macro order parameters q: [Batch, k]
        q_order = torch.matmul(x, self.U_slow)

        # [Adiabatic Elimination] Fast fluctuating modes y slaved to q
        y_slaved = self.slaving_manifold(q_order)

        # Construct orthogonal complement basis
        with torch.no_grad():
            Q, _ = torch.linalg.qr(self.U_slow)
            I = torch.eye(self.in_features, device=x.device)
            U_fast = I - torch.matmul(Q, Q.T)
            fast_basis = torch.linalg.qr(U_fast)[0][:, self.k:]
            if fast_basis.size(1) == 0:
                fast_basis = torch.zeros(self.in_features, 1, device=x.device)

        # Truncate or pad y_slaved if necessary
        if y_slaved.size(-1) != fast_basis.size(1):
            y_slaved = y_slaved[:, :fast_basis.size(1)]

        # [Downward Causality] Reconstructed constrained state
        x_reconstructed = torch.matmul(q_order, self.U_slow.T) + torch.matmul(y_slaved, fast_basis.T)
        return x_reconstructed, q_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Slaving transformation
        x_slaved, q_order = self._apply_slaving(x)

        # 2. Primary path flow
        out_primary = self.primary_path(x_slaved)

        # 3. Constructal flow resistance & stress calculation
        flow_current = torch.norm(out_primary, dim=-1).mean()
        local_variance = torch.var(out_primary, dim=0).mean() + 1e-6
        resistance = 1.0 / local_variance  # High stress when variance is suppressed / choked

        # Exponential moving average update
        self.resistance_ema = 0.9 * self.resistance_ema + 0.1 * resistance.detach()
        stress = self.resistance_ema * flow_current

        # 4. Dynamic bifurcation (Branching Rule)
        if stress.item() > self.split_threshold and self.branch_path is None:
            self.branch_path = nn.Sequential(
                nn.Linear(self.in_features, max(1, self.out_features // 2)),
                nn.GELU(),
                nn.Linear(max(1, self.out_features // 2), self.out_features)
            ).to(x.device)

        # 5. Parallel flow routing
        if self.branch_path is not None:
            out_branch = self.branch_path(x_slaved)
            return 0.65 * out_primary + 0.35 * out_branch

        return out_primary

    # Categorical Monadic Lens Interface (Optic)
    def get_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Lens Optic: Get projection."""
        return self.forward(x)

    def put_error(self, x: torch.Tensor, error: torch.Tensor, lr: float = 0.01) -> torch.Tensor:
        """Backward Lens Optic: Put error to relax slaving projection."""
        x_slaved, _ = self._apply_slaving(x)
        corrected = x_slaved - lr * torch.matmul(error, self.primary_path.weight)
        return corrected


class RealtimeSFALayer(nn.Module):
    """
    Real-Time Streaming Slow Feature Analysis (SFA)

    Extracts macro order parameters (slowest-varying features) from high-dimensional
    streaming time-series data using online covariance and temporal derivative statistics.
    """
    def __init__(self, in_features: int, num_slow_features: int, ema_decay: float = 0.99):
        super().__init__()
        self.in_features = in_features
        self.k = min(num_slow_features, in_features)
        self.ema_decay = ema_decay

        self.register_buffer("count", torch.zeros(1))
        self.register_buffer("mean_x", torch.zeros(in_features))
        self.register_buffer("cov_x", torch.eye(in_features))
        self.register_buffer("cov_diff", torch.zeros(in_features, in_features))
        self.register_buffer("W_sfa", torch.randn(in_features, self.k))

    def update_statistics(self, x_seq: torch.Tensor):
        """
        x_seq: [Batch, Sequence_Length, in_features]
        """
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)

        B, T, D = x_seq.shape
        x_flat = x_seq.reshape(-1, D)

        # 1. Mean & Covariance estimation
        current_mean = x_flat.mean(dim=0)
        x_centered = x_flat - current_mean
        current_cov = torch.matmul(x_centered.T, x_centered) / max(1, B * T - 1)

        # 2. Temporal Difference dx/dt
        if T > 1:
            x_diff = x_seq[:, 1:, :] - x_seq[:, :-1, :]
            x_diff_flat = x_diff.reshape(-1, D)
            current_cov_diff = torch.matmul(x_diff_flat.T, x_diff_flat) / max(1, B * (T - 1))
        else:
            current_cov_diff = current_cov * 0.1

        # 3. Update buffers with adaptive decay on initial step
        decay = self.ema_decay if self.count.item() > 0 else 0.0
        self.count += 1.0

        self.mean_x = decay * self.mean_x + (1.0 - decay) * current_mean
        self.cov_x = decay * self.cov_x + (1.0 - decay) * current_cov
        self.cov_diff = decay * self.cov_diff + (1.0 - decay) * current_cov_diff

    def solve_sfa_weights(self):
        """
        Solves generalized eigenvalue problem for SFA projection matrix W.
        """
        L, V = torch.linalg.eigh(self.cov_x + 1e-5 * torch.eye(self.in_features, device=self.cov_x.device))
        inv_sqrt_L = torch.diag(1.0 / torch.sqrt(torch.clamp(L, min=1e-6)))
        Q = torch.matmul(V, torch.matmul(inv_sqrt_L, V.T))

        B_tilde = torch.matmul(Q.T, torch.matmul(self.cov_diff, Q))
        eigenvalues, eigenvectors = torch.linalg.eigh(B_tilde)

        # Smallest eigenvalues correspond to slowest features
        W_whitened = eigenvectors[:, :self.k]
        self.W_sfa = torch.matmul(Q, W_whitened)

    def forward(self, x: torch.Tensor, is_training_stream: bool = True) -> torch.Tensor:
        """
        x: [Batch, Seq_Len, in_features] or [Batch, in_features]
        """
        if is_training_stream:
            with torch.no_grad():
                self.update_statistics(x)
                self.solve_sfa_weights()

        x_centered = x - self.mean_x
        q_slow_features = torch.matmul(x_centered, self.W_sfa)
        return q_slow_features


class PredictiveCodingLayer(nn.Module):
    """
    Predictive Coding Layer minimizing Local Variational Free Energy

    Generates top-down predictions and bottom-up precision-weighted prediction errors.
    Supports local SGD state relaxation without global backpropagation graph requirements.
    """
    def __init__(self, dim_current: int, dim_higher: int, lr_mu: float = 0.05):
        super().__init__()
        self.dim_current = dim_current
        self.dim_higher = dim_higher
        self.lr_mu = lr_mu

        self.g_generative = nn.Sequential(
            nn.Linear(dim_higher, dim_current),
            nn.Tanh()
        )
        self.log_precision = nn.Parameter(torch.zeros(dim_current))

    def forward(self, x_sensory: torch.Tensor, mu_higher: torch.Tensor, steps: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs local state relaxation for mu_higher to minimize Variational Free Energy F.
        """
        x_target = x_sensory.detach()
        mu_internal = mu_higher.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([mu_internal], lr=self.lr_mu)

        for _ in range(steps):
            optimizer.zero_grad()
            x_pred = self.g_generative(mu_internal)
            error = x_target - x_pred
            precision = torch.exp(self.log_precision)

            free_energy = 0.5 * torch.sum(precision * (error ** 2))
            free_energy.backward()
            optimizer.step()

        final_pred = self.g_generative(mu_internal)
        final_error = x_target - final_pred
        return mu_internal.detach(), final_error.detach()

    # Categorical Lens Methods
    def get_prediction(self, mu: torch.Tensor) -> torch.Tensor:
        return self.g_generative(mu)

    def put_error(self, x_sensory: torch.Tensor, mu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_pred = self.g_generative(mu)
        error = x_sensory - x_pred
        precision = torch.exp(self.log_precision)
        return error, precision * error


class NicheConstructionMemory(nn.Module):
    """
    Niche Construction Memory Field

    Deforms external memory manifold potential wells based on cognitive state-action
    traces, reducing retrieval friction for future similar internal states.
    """
    def __init__(self, memory_slots: int = 100, dim: int = 64, decay: float = 0.01, lr_niche: float = 0.1):
        super().__init__()
        self.memory_slots = memory_slots
        self.dim = dim
        self.decay = decay
        self.lr_niche = lr_niche

        self.register_buffer("memory_field", torch.randn(memory_slots, dim) * 0.01)

    def read_niche_context(self, query_mu: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(query_mu.unsqueeze(1), self.memory_field.unsqueeze(0), dim=-1)
        retrieval_weights = F.softmax(sim * 5.0, dim=-1)
        context = torch.matmul(retrieval_weights, self.memory_field)
        return context

    def construct_niche(self, state_mu: torch.Tensor, action_a: torch.Tensor):
        with torch.no_grad():
            self.memory_field *= (1.0 - self.decay)
            sim = F.cosine_similarity(state_mu.unsqueeze(1), self.memory_field.unsqueeze(0), dim=-1)
            target_slot = torch.argmax(sim, dim=-1)

            for b in range(state_mu.size(0)):
                slot_idx = target_slot[b]
                niche_trace = 0.5 * (state_mu[b] + action_a[b])
                self.memory_field[slot_idx] += self.lr_niche * niche_trace


class MetaLens(nn.Module):
    """
    2nd-Order Cybernetics Meta-Lens maintaining perspective matrix (B) and epistemic log-precision.
    """
    def __init__(self, dim_sensory: int, dim_total_state: int):
        super().__init__()
        self.perspective_bias = nn.Parameter(torch.randn(dim_sensory, dim_total_state) * 0.05)
        self.log_precision = nn.Parameter(torch.zeros(dim_sensory))

    def observe(self, mu_state: torch.Tensor) -> torch.Tensor:
        projected = torch.matmul(mu_state, self.perspective_bias.T)
        return torch.tanh(projected)


class GenerativeSelfModel(nn.Module):
    """
    Integrates World State (mu_world) and Self-State (mu_self) into a single state manifold.
    """
    def __init__(self, dim_world: int, dim_self: int):
        super().__init__()
        self.dim_world = dim_world
        self.dim_self = dim_self
        self.total_dim = dim_world + dim_self

        self.prior_mu_self = nn.Parameter(torch.randn(1, dim_self) * 0.1)
        self.self_world_coupling = nn.Linear(dim_self, dim_world)

    def forward_prior(self) -> torch.Tensor:
        world_prior = torch.zeros(1, self.dim_world, device=self.prior_mu_self.device)
        world_influenced_by_self = world_prior + torch.tanh(self.self_world_coupling(self.prior_mu_self))
        return torch.cat([world_influenced_by_self, self.prior_mu_self], dim=-1)


class SecondOrderCyberneticsEngine(nn.Module):
    """
    2nd-Order Cybernetics Engine executing dual relaxation:
    1st-Order: Perception state update (mu_world, mu_self)
    2nd-Order: Meta-awareness perspective adjustment (B) & epistemic uncertainty tracking.
    """
    def __init__(self, dim_sensory: int, dim_world: int, dim_self: int,
                 lr_first_order: float = 0.08, lr_second_order: float = 0.02):
        super().__init__()
        self.lr_first = lr_first_order
        self.lr_second = lr_second_order

        self.self_model = GenerativeSelfModel(dim_world, dim_self)
        self.meta_lens = MetaLens(dim_sensory, self.self_model.total_dim)

    def compute_meta_free_energy(self, x_sensory: torch.Tensor, mu_internal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_pred = self.meta_lens.observe(mu_internal)
        sensory_error = x_sensory - x_pred
        precision = torch.exp(self.meta_lens.log_precision)

        weighted_error_loss = 0.5 * torch.sum(precision * (sensory_error ** 2))
        lens_complexity = 0.5 * torch.sum(self.meta_lens.perspective_bias ** 2)
        precision_regularization = 0.5 * torch.sum(self.meta_lens.log_precision ** 2)

        free_energy = weighted_error_loss + 0.01 * (lens_complexity + precision_regularization)
        return free_energy, sensory_error

    def step_environment(self, x_sensory: torch.Tensor, relaxation_steps: int = 8) -> Dict[str, Any]:
        x_target = x_sensory.detach()
        mu_prior = self.self_model.forward_prior()
        mu_internal = mu_prior.clone().detach().requires_grad_(True)

        for _ in range(relaxation_steps):
            # 1st-Order Relaxation (Perception)
            f_meta, _ = self.compute_meta_free_energy(x_target, mu_internal)
            grad_mu = torch.autograd.grad(f_meta, mu_internal, create_graph=False)[0]
            mu_internal = (mu_internal - self.lr_first * grad_mu).detach().requires_grad_(True)

            # 2nd-Order Relaxation (Meta-Awareness)
            f_meta_updated, _ = self.compute_meta_free_energy(x_target, mu_internal)
            lens_optimizer = torch.optim.SGD(self.meta_lens.parameters(), lr=self.lr_second)
            lens_optimizer.zero_grad()
            f_meta_updated.backward()
            lens_optimizer.step()

        final_f_meta, final_error = self.compute_meta_free_energy(x_target, mu_internal)
        epistemic_uncertainty = torch.mean(torch.exp(-self.meta_lens.log_precision)).item()

        return {
            "free_energy": final_f_meta.item(),
            "epistemic_uncertainty": epistemic_uncertainty,
            "self_state_norm": torch.norm(mu_internal[:, self.self_model.dim_world:]).item(),
            "lens_bias_norm": torch.norm(self.meta_lens.perspective_bias).item(),
            "mean_error": torch.mean(torch.abs(final_error)).item(),
            "mu_world": mu_internal[:, :self.self_model.dim_world].detach(),
            "mu_self": mu_internal[:, self.self_model.dim_world:].detach()
        }


class ActiveInferenceNicheEngine(nn.Module):
    """
    End-to-End Integrated Engine

    Combines Constructal Slaved Module, SFA, Predictive Coding, Niche Memory,
    and 2nd-Order Cybernetics into a unified, self-organizing cognitive loop.
    """
    def __init__(self, dim_sensory: int = 16, dim_higher: int = 32, memory_slots: int = 50):
        super().__init__()
        self.dim_sensory = dim_sensory
        self.dim_higher = dim_higher

        self.constructal_slaved = ConstructalSlavedModule(dim_sensory, dim_sensory)
        self.sfa_layer = RealtimeSFALayer(dim_sensory, num_slow_features=4)
        self.pc_layer = PredictiveCodingLayer(dim_sensory, dim_higher)
        self.niche_memory = NicheConstructionMemory(memory_slots=memory_slots, dim=dim_higher)
        self.cybernetics_engine = SecondOrderCyberneticsEngine(dim_sensory, dim_world=8, dim_self=4)

    def forward(self, x_sensory: torch.Tensor) -> Dict[str, Any]:
        # 1. Downward causality + Constructal law flow
        x_slaved = self.constructal_slaved(x_sensory)

        # 2. Extract macro order parameters
        q_order = self.sfa_layer(x_slaved, is_training_stream=True)

        # 3. Niche memory context retrieval
        mu_prior = torch.zeros(x_sensory.size(0), self.dim_higher, device=x_sensory.device)
        context = self.niche_memory.read_niche_context(mu_prior)
        mu_init = mu_prior + context

        # 4. Local predictive coding state relaxation
        mu_converged, pc_error = self.pc_layer(x_slaved, mu_init, steps=5)

        # 5. Niche memory construction update
        self.niche_memory.construct_niche(mu_converged, mu_converged)

        # 6. 2nd-Order Cybernetics meta-relaxation
        cybernetics_telemetry = self.cybernetics_engine.step_environment(x_slaved, relaxation_steps=5)

        return {
            "x_slaved": x_slaved,
            "q_order": q_order,
            "mu_converged": mu_converged,
            "pc_error": pc_error,
            "cybernetics": cybernetics_telemetry
        }
