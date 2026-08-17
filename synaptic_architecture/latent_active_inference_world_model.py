"""
[Latent Active Inference World Model: High-Dimensional Amortized FEP Engine]

Implements Amortized Active Inference over high-dimensional latent space:
1. Encoder Network q_phi(s_t | o_t, s_{t-1}, a_{t-1}): Amortized inference mapping observations into Gaussian latent parameters (mu, log_var) in O(1) time.
2. Prior Transition Network p_theta(s_t | s_{t-1}, a_{t-1}): Predicts future latent prior state distributions.
3. Observation Decoder Network p_theta(o_t | s_t): Reconstructs high-dimensional observations from sampled latent state.
4. Reparameterization Trick: s_t = mu + std * epsilon for end-to-end backpropagation.
5. Variational Free Energy F_t: Accuracy Loss (Reconstruction) + Complexity Loss (KL divergence between recognition posterior and prior transition).
6. Expected Free Energy G_t: Epistemic Value (Latent Information Gain / Curiosity) + Pragmatic Value (Distance to target preference prior p(s_tilde)).
7. Cross-Entropy Method (CEM) Action Planner: Optimizes multi-step policy trajectories in latent space by minimizing G_t.
8. Lineage Integration: Binds latent trajectory transformations to MechanismTensor & CausalLineage for process of becoming tracking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from synaptic_architecture.mechanism_tensor import (
    CausalLineage,
    TopologicalInvariant,
    MechanismTensor
)


class LatentEncoderNetwork(nn.Module):
    """
    Amortized Recognition / Inference Model: q_phi(s_t | o_t, s_{t-1}, a_{t-1})
    Maps high-dimensional observation o_t, previous state s_{t-1}, and action a_{t-1}
    to Gaussian latent distribution parameters (mu, log_var) in O(1) time.
    """
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        input_dim = obs_dim + state_dim + action_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, state_dim)
        self.logvar_head = nn.Linear(hidden_dim, state_dim)

    def forward(self, obs: torch.Tensor, prev_state: torch.Tensor, prev_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, prev_state, prev_action], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        return mu, logvar


class PriorTransitionNetwork(nn.Module):
    """
    Prior Transition Dynamics: p_theta(s_t | s_{t-1}, a_{t-1})
    Predicts prior latent distribution parameters (prior_mu, prior_logvar) from previous latent state and action.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        input_dim = state_dim + action_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, state_dim)
        self.logvar_head = nn.Linear(hidden_dim, state_dim)

    def forward(self, prev_state: torch.Tensor, prev_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([prev_state, prev_action], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        return mu, logvar


class ObservationDecoderNetwork(nn.Module):
    """
    Likelihood / Observation Model: p_theta(o_t | s_t)
    Reconstructs high-dimensional observation from latent state s_t.
    """
    def __init__(self, state_dim: int, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.obs_head = nn.Linear(hidden_dim, obs_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(state))
        h = F.relu(self.fc2(h))
        return self.obs_head(h)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization Trick: s = mu + std * epsilon where epsilon ~ N(0, I).
    Allows backpropagation through stochastic latent sampling.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def compute_variational_free_energy(
    obs: torch.Tensor,
    recon_obs: torch.Tensor,
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
    obs_precision: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Variational Free Energy F_t:
    F_t = Accuracy Loss (Reconstruction Error) + Complexity Loss (KL Divergence q || p)
    """
    # Accuracy Loss: Negative log-likelihood (Reconstruction error)
    accuracy_loss = 0.5 * obs_precision * torch.sum((obs - recon_obs) ** 2, dim=-1).mean()

    # Complexity Loss: Analytical KL Divergence between two Gaussians N(q_mu, q_var) and N(p_mu, p_var)
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    kl_div = 0.5 * torch.sum(
        p_logvar - q_logvar + (q_var + (q_mu - p_mu) ** 2) / (p_var + 1e-8) - 1.0,
        dim=-1
    ).mean()

    free_energy = accuracy_loss + kl_div
    return free_energy, accuracy_loss, kl_div


def compute_expected_free_energy(
    predicted_state_mu: torch.Tensor,
    predicted_state_logvar: torch.Tensor,
    target_preference_mu: torch.Tensor,
    target_preference_logvar: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Expected Free Energy G_t over predicted future state distribution:
    - Epistemic Value: Information Gain / Latent Variance (Curiosity to reduce uncertainty)
    - Pragmatic Value: Distance to preferred target prior distribution p(s_tilde)
    G_t = - (Epistemic + Pragmatic)
    """
    # Epistemic Value: Entropy / Variance of predicted latent distribution
    epistemic_value = 0.5 * torch.sum(1.0 + predicted_state_logvar + math.log(2 * math.pi), dim=-1).mean()

    # Pragmatic Value: Preference satisfaction (- MSE / KL to target preference distribution)
    p_var = torch.exp(predicted_state_logvar)
    t_var = torch.exp(target_preference_logvar)
    pragmatic_value = -0.5 * torch.sum(
        (predicted_state_mu - target_preference_mu) ** 2 / (t_var + 1e-8)
    , dim=-1).mean()

    expected_free_energy = -(epistemic_value + pragmatic_value)
    return expected_free_energy, epistemic_value, pragmatic_value


class LatentActiveInferenceAgent(nn.Module):
    """
    High-Dimensional Amortized Active Inference World Model & Agent.
    Combines Encoder, Transition, Decoder, Free Energy computation, and CEM Action Planning.
    Integrates with MechanismTensor for lineage preservation.
    """
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        obs_precision: float = 1.0
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_precision = obs_precision

        self.encoder = LatentEncoderNetwork(obs_dim, state_dim, action_dim, hidden_dim)
        self.transition = PriorTransitionNetwork(state_dim, action_dim, hidden_dim)
        self.decoder = ObservationDecoderNetwork(state_dim, obs_dim, hidden_dim)

        # Target prior preference distribution p(s_tilde)
        self.register_buffer("target_pref_mu", torch.zeros(1, state_dim))
        self.register_buffer("target_pref_logvar", torch.zeros(1, state_dim))

    def set_target_preference(self, pref_mu: torch.Tensor, pref_logvar: Optional[torch.Tensor] = None):
        """Sets agent's prior goal preference distribution p(s_tilde)."""
        self.target_pref_mu = pref_mu.clone().detach()
        if pref_logvar is not None:
            self.target_pref_logvar = pref_logvar.clone().detach()
        else:
            self.target_pref_logvar = torch.zeros_like(self.target_pref_mu)

    def perceive_and_learn(
        self,
        obs: torch.Tensor,
        prev_state: torch.Tensor,
        prev_action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Amortized Perception & Learning Step:
        1. Encodes observation into posterior q(s_t | o_t, s_{t-1}, a_{t-1}).
        2. Predicts prior transition p(s_t | s_{t-1}, a_{t-1}).
        3. Samples latent state s_t via reparameterization trick.
        4. Reconstructs observation through decoder.
        5. Computes Variational Free Energy F_t.
        """
        q_mu, q_logvar = self.encoder(obs, prev_state, prev_action)
        p_mu, p_logvar = self.transition(prev_state, prev_action)

        sampled_s = reparameterize(q_mu, q_logvar)
        recon_obs = self.decoder(sampled_s)

        f_t, accuracy, complexity = compute_variational_free_energy(
            obs, recon_obs, q_mu, q_logvar, p_mu, p_logvar, self.obs_precision
        )

        return {
            "free_energy": f_t,
            "accuracy_loss": accuracy,
            "complexity_loss": complexity,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": p_mu,
            "p_logvar": p_logvar,
            "sampled_s": sampled_s,
            "recon_obs": recon_obs
        }

    def plan_action_cem(
        self,
        current_state: torch.Tensor,
        horizon: int = 5,
        num_samples: int = 32,
        top_k: int = 8,
        cem_iterations: int = 3
    ) -> Tuple[torch.Tensor, MechanismTensor]:
        """
        Cross-Entropy Method (CEM) Action Planner in Latent Space:
        Samples candidate action trajectories over planning horizon,
        evaluates Expected Free Energy G_t for each sequence,
        and selects the optimal first action a_t.
        Binds the selected trajectory to a MechanismTensor for causal lineage tracking.
        """
        device = current_state.device
        action_mean = torch.zeros(horizon, self.action_dim, device=device)
        action_std = torch.ones(horizon, self.action_dim, device=device)

        best_actions = None
        best_g = float("inf")

        for _ in range(cem_iterations):
            # Sample action sequences: [num_samples, horizon, action_dim]
            noise = torch.randn(num_samples, horizon, self.action_dim, device=device)
            action_seqs = action_mean.unsqueeze(0) + action_std.unsqueeze(0) * noise

            g_scores = torch.zeros(num_samples, device=device)

            for i in range(num_samples):
                seq_g = 0.0
                state_t = current_state.clone()

                for h in range(horizon):
                    act_t = action_seqs[i, h:h+1]
                    p_mu, p_logvar = self.transition(state_t, act_t)
                    g_step, _, _ = compute_expected_free_energy(
                        p_mu, p_logvar, self.target_pref_mu, self.target_pref_logvar
                    )
                    seq_g += g_step
                    state_t = reparameterize(p_mu, p_logvar)

                g_scores[i] = seq_g

            # Select top-k best action sequences
            _, top_indices = torch.topk(g_scores, top_k, largest=False)
            top_seqs = action_seqs[top_indices]  # [top_k, horizon, action_dim]

            action_mean = top_seqs.mean(dim=0)
            action_std = top_seqs.std(dim=0) + 1e-5

            if g_scores[top_indices[0]].item() < best_g:
                best_g = g_scores[top_indices[0]].item()
                best_actions = top_seqs[0]

        optimal_action = action_mean[0:1]

        # Bind latent transition into MechanismTensor & CausalLineage
        lineage = CausalLineage(
            node_id=f"ActiveInference_Action_t",
            transformation_history=[f"CEM_Planner_Horizon_{horizon}_G_{best_g:.4f}"]
        )
        mechanism_tensor = MechanismTensor(
            raw_tensor=optimal_action,
            lineage=lineage,
            invariant=TopologicalInvariant(name="Action_Bound", target_value=0.0)
        )

        return optimal_action, mechanism_tensor
