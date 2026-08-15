"""
4-Layer Sandwich Hybrid Architecture with Closed Feedback Loop & Active Inference.

This module implements the transitional hybrid architecture bridging statistical neural models
(System 1) and non-numerical topological causal reasoning (System 2):

Layers:
    1. Neural Perception Layer (Continuous vector / latent feature extraction + Bottom-Up threshold accumulation)
    2. Topological Transducer (Hierarchical 64-bit LSH + Topological Invariants & Motifs)
    3. Topological Causal Core (M-GRIS non-numerical O(1) bit matching & 2-morphism meta-rewriting)
    4. Constrained Synthesis Layer (Top-down causal mask injection into decoder logits / attention)

Dynamic Loops:
    - Bottom-Up Phase Transition: Numerical energy accumulation reaching critical threshold triggers topological phase shift.
    - Top-Down Boundary Constraint: Causal core topological mask restricts decoder search pathways to eliminate hallucinations.
    - Active Inference Loop: Minimizes Variational Free Energy F and Expected Free Energy G(pi) through perception & action intervention.

References:
    - THE_ABSOLUTE_COMMANDMENT.md: "Do not calculate, let it flow."
    - AGENTS.md: Continuous Causal Intelligence Principles.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from core.physics.mgris_engine import (
    MolecularGraph,
    MGRISInferenceEngine,
    MGRISCausalBridge,
    Node,
    StickyEnd,
    Polarity,
)


# =====================================================================
# 1. Straight-Through Estimator (STE) Binarization for LSH
# =====================================================================
class STEBinarize(torch.autograd.Function):
    """
    Forward: Hard Sign quantization (+1 / -1) for 64-bit LSH projection.
    Backward: Smooth gradient via tanh derivative approximation sech^2(beta * x).
    """
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.beta = beta
        return torch.sign(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        input_tensor, = ctx.saved_tensors
        beta = ctx.beta
        # sech^2(beta * x) approximation for backpropagation
        sech_sq = 1.0 - torch.tanh(beta * input_tensor) ** 2
        grad_input = grad_output * beta * sech_sq
        return grad_input, None


class TopologyLSHProjection(nn.Module):
    """
    64-bit Topology-Preserving LSH Projection Layer with Hierarchical Allocation:
        - Bits 0 – 15: Macro Topological Invariants (H_0, H_1, Betti numbers)
        - Bits 16 – 47: Meso Causal Field Geometry & Curvature Constraints
        - Bits 48 – 63: Micro Sticky Ends (Polarity & O(1) Complementary Binding)
    """
    def __init__(self, in_features: int, out_bits: int = 64, beta: float = 5.0):
        super().__init__()
        self.in_features = in_features
        self.out_bits = out_bits
        self.beta = beta
        # Projection weight matrix W in R^{64 x in_features}
        self.weight = nn.Parameter(torch.randn(out_bits, in_features) * (2.0 / in_features) ** 0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [Batch, in_features]
        Returns:
            binary_code: [Batch, 64] with values in {-1, 1}
            logits: [Batch, 64] continuous pre-activation logits for Soft-OT alignment
        """
        logits = F.linear(x, self.weight)  # [Batch, 64]
        binary_code = STEBinarize.apply(logits, self.beta)
        return binary_code, logits

    @staticmethod
    def extract_bit_fields(binary_code: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Splits 64-bit binary codes into multi-scale hierarchical bit fields:
            - macro: [Batch, 16] (Bits 0-15)
            - meso:  [Batch, 32] (Bits 16-47)
            - micro: [Batch, 16] (Bits 48-63)
        """
        macro = binary_code[:, 0:16]
        meso = binary_code[:, 16:48]
        micro = binary_code[:, 48:64]
        return {"macro": macro, "meso": meso, "micro": micro}


# =====================================================================
# 2. Sinkhorn-based Unbiased Soft-Optimal Transport Loss
# =====================================================================
class SinkhornSoftOTLoss(nn.Module):
    """
    C^infty Smooth Sinkhorn Topological Divergence Loss.
    S_eps(D1, D2) = OT_eps(D1, D2) - 0.5 * OT_eps(D1, D1) - 0.5 * OT_eps(D2, D2)
    """
    def __init__(self, eps: float = 0.05, max_iter: int = 20, p: float = 2.0):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.p = p

    def _cost_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, y, p=self.p) ** self.p

    def _sinkhorn_loop(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C = self._cost_matrix(x, y)  # [Batch, N, M]
        B, N, M = C.shape

        a = torch.full((B, N), 1.0 / N, device=x.device, dtype=x.dtype)
        b = torch.full((B, M), 1.0 / M, device=y.device, dtype=y.dtype)

        K = torch.exp(-C / self.eps)  # [Batch, N, M]
        u = torch.ones_like(a)

        for _ in range(self.max_iter):
            # v: [B, M] = b / (K^T @ u)
            v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)
            # u: [B, N] = a / (K @ v)
            u = a / (torch.bmm(K, v.unsqueeze(2)).squeeze(2) + 1e-8)

        P = u.unsqueeze(2) * K * v.unsqueeze(1)  # [Batch, N, M]
        loss = torch.sum(P * C, dim=(-2, -1))
        return loss

    def forward(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        ot_12 = self._sinkhorn_loop(d1, d2)
        ot_11 = self._sinkhorn_loop(d1, d1)
        ot_22 = self._sinkhorn_loop(d2, d2)

        sinkhorn_div = ot_12 - 0.5 * ot_11 - 0.5 * ot_22
        return torch.mean(torch.clamp(sinkhorn_div, min=0.0))


# =====================================================================
# 3. Layer 1: Neural Perception Layer (Continuous Feature & Accumulation)
# =====================================================================
class NeuralPerceptionLayer(nn.Module):
    """
    Layer 1: Continuous Sensory Perception.
    Encodes unstructured inputs into high-dimensional latent vectors and tracks
    numerical energy density / signal accumulation to detect critical thresholds.
    """
    def __init__(self, in_dim: int, latent_dim: int = 256, threshold: float = 5.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.threshold = threshold

    def forward(self, raw_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            latents: [Batch, latent_dim] continuous feature vector
            accumulated_energy: [Batch] signal norm / energy density accumulation
            is_critical: [Batch] boolean mask indicating whether energy exceeds critical threshold
        """
        latents = self.encoder(raw_input)
        accumulated_energy = torch.norm(latents, p=2, dim=-1)
        is_critical = accumulated_energy >= self.threshold
        return latents, accumulated_energy, is_critical


# =====================================================================
# 4. Layer 2: Topological Transducer Interface
# =====================================================================
class TopologicalTransducer(nn.Module):
    """
    Layer 2: Discretization Transducer.
    Maps continuous vectors to 64-bit topological bitmasks and extracts invariants/motifs.
    """
    def __init__(self, latent_dim: int = 256, out_bits: int = 64, beta: float = 5.0):
        super().__init__()
        self.lsh_proj = TopologyLSHProjection(latent_dim, out_bits, beta)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        binary_code, logits = self.lsh_proj(latents)
        bit_fields = self.lsh_proj.extract_bit_fields(binary_code)
        return binary_code, logits, bit_fields

    def convert_to_sticky_end(self, binary_code_row: torch.Tensor) -> StickyEnd:
        """
        Converts a 64-bit binary code row into a 64-bit StickyEnd for M-GRIS O(1) matching.
        """
        # Convert {-1, 1} tensor to 64-bit integer bit pattern
        bits = (binary_code_row > 0).long()
        pattern = 0
        for b in bits:
            pattern = (pattern << 1) | int(b.item())
        polarity = Polarity.DONOR if (pattern & 1) == 0 else Polarity.ACCEPTOR
        return StickyEnd(polarity, pattern)


# =====================================================================
# 5. Layer 3: Non-Numerical Topological Causal Core (M-GRIS)
# =====================================================================
class TopologicalCausalCore:
    """
    Layer 3: M-GRIS Non-numerical Topological Causal Core Wrapper.
    Executes graph rewriting, O(1) bit matching, 2-morphism meta-rewriting, and produces
    topological constraint masks (Phi_constraint) for top-down decoding.
    """
    def __init__(self, atp_budget: int = 100):
        self.engine = MGRISInferenceEngine(atp_budget=atp_budget)

    def process_topological_motifs(
        self,
        query_sticky_end: StickyEnd,
        knowledge_pool_nodes: List[Node],
        contradiction_masks: Optional[List[int]] = None
    ) -> Tuple[MolecularGraph, List[str], torch.Tensor]:
        """
        Runs M-GRIS graph rewriting cycle and constructs a 64-bit constraint mask.
        """
        query_node = Node(
            node_id=0,
            label="QueryPercept",
            sticky_ends=[query_sticky_end],
            valence_limit=4
        )
        graph, narrative = self.engine.execute_inference_cycle(
            query_strand=query_node,
            knowledge_pool=knowledge_pool_nodes,
            max_depth=5,
            contradiction_masks=contradiction_masks
        )

        # Generate topological relaxation constraint mask Phi_constraint
        # Encodes remaining active nodes and bonds into a 64-bit mask
        mask_val = 0
        for nid, n in graph.nodes.items():
            mask_val ^= (n.constraint_mask | (nid * 0x1F2E3D4C5B6A7981))

        mask_bits = [(mask_val >> (63 - i)) & 1 for i in range(64)]
        constraint_tensor = torch.tensor(mask_bits, dtype=torch.float32)
        return graph, narrative, constraint_tensor


# =====================================================================
# 6. Layer 4: Constrained Synthesis Layer (Decoder Mask Injection)
# =====================================================================
class ConstrainedSynthesisLayer(nn.Module):
    """
    Layer 4: Top-Down Constrained Synthesis Layer.
    Applies top-down topological constraints (Phi_constraint) to decoder logits / attention masks,
    restricting search space and blocking hallucination.
    """
    def __init__(self, latent_dim: int = 256, vocab_dim: int = 1000):
        super().__init__()
        self.decoder_head = nn.Linear(latent_dim, vocab_dim)
        self.mask_projection = nn.Linear(64, vocab_dim)

    def forward(
        self,
        decoder_latents: torch.Tensor,
        constraint_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        decoder_latents: [Batch, latent_dim]
        constraint_mask: [Batch, 64] or [64] top-down topological boundary condition
        Returns:
            constrained_logits: [Batch, vocab_dim]
            allowed_space_mask: [Batch, vocab_dim] boolean mask of allowed search space
        """
        raw_logits = self.decoder_head(decoder_latents)  # [Batch, vocab_dim]

        if constraint_mask.dim() == 1:
            constraint_mask = constraint_mask.unsqueeze(0).expand(decoder_latents.size(0), -1)

        # Map 64-bit topological constraint to vocabulary search mask
        bias_logits = self.mask_projection(constraint_mask)

        # Apply top-down boundary condition constraint
        # Negative penalty closes invalid search paths
        penalty = (1.0 - torch.sigmoid(bias_logits)) * -1e9
        constrained_logits = raw_logits + penalty
        allowed_space_mask = bias_logits > 0.0

        return constrained_logits, allowed_space_mask


# =====================================================================
# 7. End-to-End Sandwich Hybrid Architecture & Active Inference Loop
# =====================================================================
class SandwichHybridArchitecture(nn.Module):
    """
    Unified 4-Layer Sandwich Hybrid Architecture with Closed Feedback Loop & Active Inference.
    """
    def __init__(
        self,
        in_dim: int = 128,
        latent_dim: int = 256,
        vocab_dim: int = 1000,
        threshold: float = 5.0,
        beta: float = 5.0,
        eps: float = 0.05
    ):
        super().__init__()
        self.perception_layer = NeuralPerceptionLayer(in_dim, latent_dim, threshold)
        self.transducer_layer = TopologicalTransducer(latent_dim, out_bits=64, beta=beta)
        self.causal_core = TopologicalCausalCore(atp_budget=100)
        self.synthesis_layer = ConstrainedSynthesisLayer(latent_dim, vocab_dim)
        self.sinkhorn_loss = SinkhornSoftOTLoss(eps=eps, max_iter=25)

    def forward(
        self,
        raw_input: torch.Tensor,
        target_barcodes: Optional[torch.Tensor] = None,
        knowledge_pool: Optional[List[Node]] = None
    ) -> Dict[str, Any]:
        """
        Full Bottom-Up & Top-Down Forward Cycle:
        1. Layer 1: Neural perception & energy accumulation check.
        2. Layer 2: Discretization transducer (64-bit LSH + bit fields).
        3. Layer 3: M-GRIS topological causal graph rewriting (Bottom-Up Phase Transition).
        4. Layer 4: Top-down constrained synthesis decoding.
        """
        # 1. Neural Perception Layer
        latents, energy, is_critical = self.perception_layer(raw_input)

        # 2. Topological Transducer Layer
        binary_code, logits, bit_fields = self.transducer_layer(latents)

        # 3. Non-Numerical Causal Core Pass
        batch_size = raw_input.size(0)
        constraint_masks = []
        graphs = []
        narratives = []

        if knowledge_pool is None:
            # Default empty knowledge pool or basic concept node
            knowledge_pool = [
                MGRISCausalBridge.create_concept_node(1, "EffectA", "CauseA"),
                MGRISCausalBridge.create_concept_node(2, "EffectB", "CauseB"),
            ]

        for i in range(batch_size):
            sticky_end = self.transducer_layer.convert_to_sticky_end(binary_code[i])
            g, narr, c_mask = self.causal_core.process_topological_motifs(sticky_end, knowledge_pool)
            graphs.append(g)
            narratives.append(narr)
            constraint_masks.append(c_mask)

        constraint_masks_tensor = torch.stack(constraint_masks).to(raw_input.device)  # [Batch, 64]

        # 4. Constrained Synthesis Layer
        constrained_logits, allowed_mask = self.synthesis_layer(latents, constraint_masks_tensor)

        # Sinkhorn Soft-OT Loss if target topological barcodes provided
        ot_loss = None
        if target_barcodes is not None:
            pred_diagrams = logits.view(batch_size, 16, 4)
            ot_loss = self.sinkhorn_loss(pred_diagrams, target_barcodes)

        return {
            "latents": latents,
            "accumulated_energy": energy,
            "is_critical": is_critical,
            "binary_code": binary_code,
            "logits": logits,
            "bit_fields": bit_fields,
            "constraint_masks": constraint_masks_tensor,
            "constrained_logits": constrained_logits,
            "allowed_space_mask": allowed_mask,
            "narratives": narratives,
            "sinkhorn_loss": ot_loss
        }

    # =========================================================================
    # ACTIVE INFERENCE LOOP (Karl Friston Active Inference)
    # =========================================================================

    def compute_free_energy(
        self,
        observation: torch.Tensor,
        internal_state: torch.Tensor,
        target_diagram: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates Variational Free Energy F = Prediction Error + Topological Sinkhorn Loss.
        Perception updates internal model state s_t to minimize F.
        """
        pred_error = F.mse_loss(observation, internal_state)
        if target_diagram is not None:
            _, logits, _ = self.transducer_layer(internal_state)
            pred_diag = logits.view(logits.size(0), 16, 4)
            ot_loss = self.sinkhorn_loss(pred_diag, target_diagram)
            return pred_error + ot_loss
        return pred_error

    def active_inference_step(
        self,
        observation: torch.Tensor,
        internal_state: torch.Tensor,
        candidate_actions: torch.Tensor,
        env_transition_fn: Any,
        target_diagram: Optional[torch.Tensor] = None,
        lr: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes one Active Inference step:
        1. Perception: Updates internal_state via gradient descent on Free Energy F.
        2. Action Selection: Evaluates Candidate Actions to minimize Expected Free Energy G(pi).
        3. Intervention: Applies selected action a_t to environment (Judea Pearl do(x)).

        Returns:
            updated_internal_state: s_{t+1}
            best_action: a_t
            new_observation: o_{t+1}
        """
        # 1. Perception Step (Internal Model Optimization)
        state_param = internal_state.clone().detach().requires_grad_(True)
        free_energy = self.compute_free_energy(observation, state_param, target_diagram)
        free_energy.backward()

        with torch.no_grad():
            updated_internal_state = state_param - lr * state_param.grad

        # 2. Policy Selection (Expected Free Energy G(pi) Minimization)
        best_action = None
        min_expected_fe = float("inf")

        with torch.no_grad():
            for i in range(candidate_actions.size(0)):
                act = candidate_actions[i : i + 1]
                # Simulate predicted future observation from action
                predicted_obs = env_transition_fn(updated_internal_state, act)
                expected_fe = self.compute_free_energy(predicted_obs, updated_internal_state, target_diagram).item()
                if expected_fe < min_expected_fe:
                    min_expected_fe = expected_fe
                    best_action = act

        # 3. Active Intervention Step (Action execution on environment)
        new_observation = env_transition_fn(updated_internal_state, best_action)

        return updated_internal_state, best_action, new_observation
