"""
Wave Cognition & Continuous Latent Memory Engine
=================================================
Philosophy: "Do not calculate, let it flow."

This module implements the Wave Cognition Engine and Continuous SDF Latent Memory Architecture for Elysia:
1. MultimodalFrictionLayer: Normalizes multi-sensory input entropy and prediction surprise into a 3D causal friction field R_causal(x).
2. SDFCognitionEngine: Evolves Level Set PDE wavefronts, applies thermodynamic diffusion decay (forgetting/abstraction), and solidifies frequently visited surfaces (phase-locking).
3. SDFLatentMemoryLayer: Provides fully differentiable native PyTorch $O(\\log N)$ Sphere Tracing knowledge retrieval and non-destructive Smooth Minimum (smin) memory writes (preventing catastrophic forgetting).
4. SDFTransformerDecoder & Components: A HuggingFace-style transformer decoder pipeline replacing sequence-length-dependent KV Caches and O(N^2) self-attention with O(1) continuous SDF latent memory fields.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Multimodal Friction Normalization Layer
# =============================================================================

class MultimodalFrictionLayer(nn.Module):
    """
    Transforms multi-sensory input data into a continuous 3D causal friction field R_causal(x).
    Calculates surprise (prediction error) and information entropy, then projects them onto 3D latent space via RBF.
    """
    def __init__(self, feature_dim: int = 64, grid_res: int = 32, gamma: float = 15.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_res = grid_res
        self.gamma = gamma

        # Predictor for surprise metric
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, feature_dim)
        )
        # Spatial projection to 3D latent coordinates
        self.spatial_projection = nn.Linear(feature_dim, 3)

        # 3D spatial grid coordinates [-1.0, 1.0]^3
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid_res),
            torch.linspace(-1, 1, grid_res),
            torch.linspace(-1, 1, grid_res),
            indexing='ij'
        ), dim=-1)
        self.register_buffer('grid_coords', coords)

    def forward(self, x_multimodal: torch.Tensor) -> torch.Tensor:
        """
        x_multimodal: [Batch, Feature_Dim]
        returns: R_causal [grid_res, grid_res, grid_res]
        """
        batch_size = x_multimodal.size(0)

        x_pred = self.predictor(x_multimodal)
        mismatch_error = torch.mean((x_multimodal - x_pred) ** 2, dim=-1)

        probs = F.softmax(x_multimodal, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

        friction_scalar = mismatch_error + 0.1 * entropy
        centers = torch.tanh(self.spatial_projection(x_multimodal))

        diff = self.grid_coords.unsqueeze(3) - centers.view(1, 1, 1, batch_size, 3)
        dist_sq = torch.sum(diff ** 2, dim=-1)

        rbf_weights = torch.exp(-self.gamma * dist_sq)
        R_causal = torch.sum(friction_scalar.view(1, 1, 1, batch_size) * rbf_weights, dim=-1)

        return R_causal


# =============================================================================
# 2. SDF Cognition Engine (Level Set PDE & Phase State Transitions)
# =============================================================================

class SDFCognitionEngine(nn.Module):
    """
    Manages Level Set PDE wave evolution, thermodynamic diffusion (forgetting),
    surface consolidation (phase-locking), and topological fusion (smooth minimum).
    """
    def __init__(self, grid_res: int = 32, nu_decay: float = 0.005, dt: float = 0.05):
        super().__init__()
        self.grid_res = grid_res
        self.nu_decay = nu_decay
        self.dt = dt

        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid_res),
            torch.linspace(-1, 1, grid_res),
            torch.linspace(-1, 1, grid_res),
            indexing='ij'
        ), dim=-1)
        init_sdf = torch.norm(coords, dim=-1) - 0.4

        self.sdf = nn.Parameter(init_sdf, requires_grad=False)
        self.phase_lock = nn.Parameter(torch.zeros_like(init_sdf), requires_grad=False)

    def _compute_laplacian_3d(self, tensor_3d: torch.Tensor) -> torch.Tensor:
        padded = F.pad(tensor_3d.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode='replicate').squeeze(0).squeeze(0)
        laplacian = (
            padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1] +
            padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1] +
            padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2] -
            6.0 * tensor_3d
        )
        return laplacian

    def _compute_gradient_norm(self, tensor_3d: torch.Tensor) -> torch.Tensor:
        dx = (torch.roll(tensor_3d, -1, dims=0) - torch.roll(tensor_3d, 1, dims=0)) / 2.0
        dy = (torch.roll(tensor_3d, -1, dims=1) - torch.roll(tensor_3d, 1, dims=1)) / 2.0
        dz = (torch.roll(tensor_3d, -1, dims=2) - torch.roll(tensor_3d, 1, dims=2)) / 2.0
        return torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)

    def smooth_minimum(self, sdf_a: torch.Tensor, sdf_b: torch.Tensor, k: float = 8.0) -> torch.Tensor:
        """Polynomial/logarithmic smooth minimum for topological fusion."""
        return -torch.log(torch.exp(-k * sdf_a) + torch.exp(-k * sdf_b) + 1e-8) / k

    def step(self, R_causal: torch.Tensor, F0: float = 0.3):
        """
        Advances cognition field by 1 time step:
        - Level Set PDE update: ∂φ/∂t = -F |∇φ|
        - Diffusion Decay: ∂φ/∂t = ν ∇²φ
        - Phase-Locking: solidifies stagnant surface points
        """
        active_mask = 1.0 - self.phase_lock

        velocity = (F0 - R_causal) * active_mask
        grad_norm = self._compute_gradient_norm(self.sdf)

        d_phi_wave = -velocity * grad_norm * self.dt
        laplacian = self._compute_laplacian_3d(self.sdf)
        d_phi_decay = self.nu_decay * laplacian * active_mask * self.dt

        self.sdf.copy_(self.sdf + d_phi_wave + d_phi_decay)

        near_surface = (torch.abs(self.sdf) < 0.05).float()
        stagnant = (torch.abs(velocity) < 0.05).float()

        consolidation = near_surface * stagnant * 0.05
        self.phase_lock.copy_(torch.clamp(self.phase_lock + consolidation, 0.0, 1.0))


# =============================================================================
# 3. Fully Differentiable Continuous Latent Memory Layer
# =============================================================================

class SDFLatentMemoryLayer(nn.Module):
    """
    Continuous Latent Memory Layer replacing sequence-length dependent KV Cache and Self-Attention.
    Uses fully differentiable native PyTorch Sphere Tracing ray marching over 3D SDF and Value fields.
    """
    def __init__(self, hidden_size: int = 64, grid_res: int = 32, k_smin: float = 10.0):
        super().__init__()
        self.latent_dim = hidden_size
        self.res = grid_res
        self.k_smin = k_smin

        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid_res),
            torch.linspace(-1, 1, grid_res),
            torch.linspace(-1, 1, grid_res),
            indexing='ij'
        ), dim=-1)

        init_sdf = torch.norm(coords, dim=-1) - 0.4
        self.sdf_grid = nn.Parameter(init_sdf.view(1, 1, grid_res, grid_res, grid_res), requires_grad=True)
        self.value_grid = nn.Parameter(torch.randn(1, hidden_size, grid_res, grid_res, grid_res) * 0.01, requires_grad=True)

        self.q_pos_proj = nn.Linear(hidden_size, 3)
        self.q_dir_proj = nn.Linear(hidden_size, 3)

    def read_memory(self, query: torch.Tensor, max_steps: int = 8, eps: float = 1e-3) -> torch.Tensor:
        """
        Differentiable Sphere Tracing read via grid_sample.
        Computes exact autograd gradients for query, sdf_grid, and value_grid.
        """
        num_tokens = query.shape[0]
        q_pos = torch.tanh(self.q_pos_proj(query))
        q_dir = F.normalize(self.q_dir_proj(query), dim=-1)

        curr_pos = q_pos
        sdf_batch = self.sdf_grid.repeat(num_tokens, 1, 1, 1, 1)

        for _ in range(max_steps):
            grid_coords = curr_pos.view(num_tokens, 1, 1, 1, 3)
            dist = F.grid_sample(
                sdf_batch,
                grid_coords,
                align_corners=True
            ).view(num_tokens, 1)

            curr_pos = curr_pos + q_dir * torch.clamp(dist, min=eps)
            curr_pos = torch.clamp(curr_pos, -1.0, 1.0)

        final_coords = curr_pos.view(num_tokens, 1, 1, 1, 3)
        value_batch = self.value_grid.repeat(num_tokens, 1, 1, 1, 1)
        readout = F.grid_sample(
            value_batch,
            final_coords,
            align_corners=True
        ).squeeze(-1).squeeze(-1).squeeze(-1)

        return readout

    def write_memory(self, new_sdf: torch.Tensor, new_value: torch.Tensor):
        """
        Non-Destructive Write via Smooth Minimum (smin).
        Preserves existing knowledge field while blending new potential wells.
        """
        k = self.k_smin
        old_sdf = self.sdf_grid

        w_old = torch.exp(-k * old_sdf)
        w_new = torch.exp(-k * new_sdf)

        updated_sdf = -torch.log(w_old + w_new + 1e-8) / k
        w_total = w_old + w_new + 1e-8

        updated_value = (w_old / w_total) * self.value_grid + (w_new / w_total) * new_value

        with torch.no_grad():
            self.sdf_grid.copy_(updated_sdf)
            self.value_grid.copy_(updated_value)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        memory_context = self.read_memory(query)
        return query + memory_context


# =============================================================================
# 4. HuggingFace Style SDF Transformer Decoder Pipeline
# =============================================================================

@dataclass
class SDFTransformerConfig:
    hidden_size: int = 64
    intermediate_size: int = 128
    num_decoder_layers: int = 2
    grid_res: int = 32
    k_smin: float = 10.0
    friction_threshold: float = 0.25
    layer_norm_eps: float = 1e-5


class SDFMLP(nn.Module):
    def __init__(self, config: SDFTransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SDFDecoderLayer(nn.Module):
    """
    Single Decoder Block:
    Replaces Self-Attention and KV Cache with SDFLatentMemoryLayer.
    """
    def __init__(self, config: SDFTransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.friction_threshold = config.friction_threshold

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.sdf_memory = SDFLatentMemoryLayer(
            hidden_size=config.hidden_size,
            grid_res=config.grid_res,
            k_smin=config.k_smin
        )

        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SDFMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_friction: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_states
        normed_states = self.input_layernorm(hidden_states)

        batch_size, seq_len, hidden_dim = normed_states.size()
        flat_states = normed_states.view(-1, hidden_dim)

        memory_readout = self.sdf_memory.read_memory(flat_states)
        memory_readout = memory_readout.view(batch_size, seq_len, hidden_dim)

        hidden_states = residual + memory_readout

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))

        if causal_friction is not None and causal_friction.mean() > self.friction_threshold:
            delta_sdf = torch.ones_like(self.sdf_memory.sdf_grid) * 0.3
            delta_value = torch.randn_like(self.sdf_memory.value_grid) * 0.05
            self.sdf_memory.write_memory(delta_sdf, delta_value)

        return hidden_states


class SDFTransformerDecoder(nn.Module):
    """
    Transformer Decoder Pipeline operating with O(1) continuous SDF Latent Memory.
    """
    def __init__(self, config: SDFTransformerConfig, vocab_size: int = 1000):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SDFDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        target_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            friction = None
            if target_ids is not None:
                with torch.no_grad():
                    temp_logits = self.lm_head(self.norm(hidden_states))
                    friction = F.cross_entropy(temp_logits.view(-1, temp_logits.size(-1)), target_ids.view(-1))

            hidden_states = layer(hidden_states, causal_friction=friction)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int = 5) -> torch.LongTensor:
        """
        Generates tokens sequentially using fixed O(1) memory field without KV Cache.
        """
        curr_input = input_ids
        for _ in range(max_new_tokens):
            logits, _ = self.forward(curr_input)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            curr_input = torch.cat([curr_input, next_token], dim=1)
        return curr_input
