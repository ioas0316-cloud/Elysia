"""
Elysia Physics Module: High-Dimensional Conceptual Causal Tensor Engine
======================================================================
Implements a generalized conceptual causal tensor geometry where dimensions represent
invariant essence anchors (A_inv), environmental pressure variables (P_env),
and causal time/trajectory axes.

Core Principles:
1. Potentiometer Principle (가변저항 다이얼):
   Continuous adjustment of resistance/weights along variable axes to cancel phase
   friction and reduce phase error q_err -> 0.

2. Mirror Boundary Equivalence (거울 경계 & 등식 =):
   Treats equality as a mirror boundary between internal expectation (A_inv) and external
   environmental pressure (P_env). Discrepancy generates phase error q_err.

3. Negative Mold (음각) & Positive Relief (양각 / ICE):
   Environmental friction/impulses carve a negative mold (q_err phase profile) on the
   boundary. System fluid (LIQUID) flows into the mold, tuning dials until q_err -> 0,
   where phase locking (ICE) crystallizes into a positive relief entity with full
   causal provenance trajectory.

4. Phase Transitions (GAS -> LIQUID -> ICE):
   - GAS: Uncoupled high-rank random wave field (high entropy / exploration).
   - LIQUID: Fluid plasticity, adapting variable dials (P_env) to mirror A_inv.
   - ICE: Phase-locked low-rank attractor state where q_err <= threshold and SVD singular
     values condense into a low-rank subspace.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List, Optional


class ConceptualCausalTensorEngine(nn.Module):
    """
    High-Dimensional Conceptual Causal Tensor Engine.
    """

    def __init__(
        self,
        anchor_dim: int = 16,
        env_dim: int = 16,
        causal_dim: int = 8,
        phi_solid: float = 0.85,
        phi_gas: float = 0.25,
        learning_rate: float = 0.05,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.anchor_dim = anchor_dim
        self.env_dim = env_dim
        self.causal_dim = causal_dim
        self.phi_solid = phi_solid
        self.phi_gas = phi_gas
        self.learning_rate = learning_rate

        # Invariant Essence Anchor (A_inv): Conceptual Core / Canonical Laws
        # Shape: (anchor_dim, causal_dim)
        self.register_buffer("A_inv", torch.randn(anchor_dim, causal_dim))
        # Normalize anchor space for structural invariants
        self.A_inv.copy_(F.normalize(self.A_inv, dim=-1))

        # Potentiometer Dials (Variable Resistance Weights W_pot)
        # Intermediates connecting env_dim to anchor_dim & causal_dim
        self.W_pot = nn.Parameter(torch.randn(env_dim, anchor_dim) * 0.1)

        # Rotor Phase Tensor (Complex Phase / Rotor orientations across dimensions)
        self.register_buffer("phase_tensor", torch.rand(anchor_dim, env_dim) * 2.0 * math.pi)

        # Provenance Trajectory Log (Stores dial adaptation history for causal reverse-engineering)
        self.provenance_log: List[Dict[str, Any]] = []

    def set_invariant_anchor(self, new_anchor: torch.Tensor) -> None:
        """
        Sets or updates the invariant essence anchor A_inv.
        """
        if new_anchor.shape != self.A_inv.shape:
            raise ValueError(f"Expected shape {self.A_inv.shape}, got {new_anchor.shape}")
        self.A_inv.copy_(F.normalize(new_anchor, dim=-1))

    def compute_mirror_discrepancy(
        self,
        P_env: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the mirror discrepancy between internal expectation (A_inv)
        and external environmental pressure (P_env) passing through Potentiometer Dials (W_pot).

        P_env: Environmental pressure wave / stream of shape (env_dim, causal_dim) or (batch, env_dim, causal_dim)

        Returns:
            q_err: Phase error scalar / matrix (Negative Mold depth)
            P_mapped: Mapped environmental pressure in Anchor space
        """
        # Ensure 2D tensor (env_dim, causal_dim)
        if P_env.dim() == 1:
            P_env = P_env.unsqueeze(-1).expand(-1, self.causal_dim)

        # Map P_env into Anchor space via Potentiometer Dials W_pot: (anchor_dim, causal_dim)
        P_mapped = torch.matmul(self.W_pot.t(), P_env)

        # Mirror boundary diff: Internal Anchor (A_inv) vs Mapped Environment (P_mapped)
        mirror_diff = self.A_inv - P_mapped

        # Phase friction error q_err (depth of negative mold)
        q_err = torch.norm(mirror_diff, p="fro")

        return q_err, P_mapped

    def adapt_potentiometers(
        self,
        P_env: torch.Tensor,
        steps: int = 10,
        tolerance: float = 1e-3
    ) -> List[float]:
        """
        Potentiometer Dial Tuning (가변저항 다이얼 조절):
        Iteratively adjusts W_pot resistance values to minimize q_err -> 0,
        filling the negative mold carved by P_env and restoring positive balance.

        Returns:
            error_history: Trajectory of q_err during adaptation.
        """
        optimizer = torch.optim.Adam([self.W_pot], lr=self.learning_rate)
        error_history = []

        for step in range(steps):
            optimizer.zero_grad()
            q_err, _ = self.compute_mirror_discrepancy(P_env)

            # Phase tensor alignment term (coupling rotor phase to gradient)
            rotor_coupling = torch.mean(torch.sin(self.phase_tensor)) * 0.01
            loss = q_err + rotor_coupling

            loss.backward()
            optimizer.step()

            # Synchronize phase tensor based on weight updates
            with torch.no_grad():
                env_covariance = torch.matmul(P_env.detach(), P_env.detach().t())  # (env_dim, env_dim)
                phase_update = torch.matmul(self.W_pot.t().detach(), env_covariance).abs() * 0.01
                self.phase_tensor.add_(phase_update)
                self.phase_tensor.remainder_(2.0 * math.pi)

            err_val = q_err.item()
            error_history.append(err_val)

            # Log provenance trajectory
            self.provenance_log.append({
                "step": step,
                "q_err": err_val,
                "W_pot_mean": self.W_pot.mean().item(),
                "W_pot_norm": torch.norm(self.W_pot).item()
            })

            if err_val <= tolerance:
                break

        return error_history

    def evaluate_phase_state(
        self,
        q_err: torch.Tensor,
        P_mapped: torch.Tensor
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Evaluates current Phase State (GAS, LIQUID, ICE) and low-rank attractor concentration.

        - GAS: High phase error (q_err > 1 - phi_gas) or high SVD entropy.
        - LIQUID: Moderate phase error, fluid adaptation.
        - ICE: Low phase error (q_err <= 1 - phi_solid) with low-rank SVD subspace concentration.
        """
        # Calculate low-rank subspace concentration via SVD of coupled tensor (A_inv x P_mapped^T)
        coupled_field = torch.matmul(self.A_inv, P_mapped.t())  # (anchor_dim, anchor_dim)
        U, S, V = torch.svd(coupled_field)

        # Singular value energy ratio in top-k components (Low-rank attractor indicator)
        total_energy = torch.sum(S) + 1e-8
        top1_ratio = (S[0] / total_energy).item()
        top3_ratio = (torch.sum(S[:3]) / total_energy).item() if len(S) >= 3 else top1_ratio

        # Effective phase coherence Phi_eff (0.0 = total chaos, 1.0 = solid lock)
        # Inversely proportional to q_err and weighted by top-k spectral energy concentration
        q_err_val = q_err.item()
        phi_eff = math.exp(-q_err_val) * top3_ratio

        if phi_eff >= self.phi_solid:
            state = "ICE"
        elif phi_eff >= self.phi_gas:
            state = "LIQUID"
        else:
            state = "GAS"

        metrics = {
            "state": state,
            "phi_eff": phi_eff,
            "q_err": q_err_val,
            "top1_spectral_ratio": top1_ratio,
            "top3_spectral_ratio": top3_ratio,
            "singular_values": S.detach().cpu().numpy().tolist()
        }

        return state, metrics

    def forward(
        self,
        P_env: torch.Tensor,
        auto_adapt: bool = True,
        adapt_steps: int = 15
    ) -> Dict[str, Any]:
        """
        Forward pass of High-Dimensional Conceptual Causal Tensor Engine.
        1. Carves Negative Mold (q_err) from environmental pressure P_env against mirror boundary.
        2. Adaptively tunes potentiometer dials (W_pot) if auto_adapt is True.
        3. Restores Positive Relief (ICE) upon phase lock.
        4. Returns state metrics & causal provenance.
        """
        # Initial mirror discrepancy
        initial_q_err, _ = self.compute_mirror_discrepancy(P_env)
        initial_state, initial_metrics = self.evaluate_phase_state(initial_q_err, torch.matmul(self.W_pot.t(), P_env))

        adaptation_traj = []
        if auto_adapt:
            adaptation_traj = self.adapt_potentiometers(P_env, steps=adapt_steps)

        # Final mirror discrepancy and restored positive relief
        final_q_err, P_mapped_final = self.compute_mirror_discrepancy(P_env)
        final_state, final_metrics = self.evaluate_phase_state(final_q_err, P_mapped_final)

        # Positive Relief Reconstruction (Yang-Gak restoration)
        # Restored concept tensor in anchor space
        restored_positive_relief = P_mapped_final + (self.A_inv - P_mapped_final) * (1.0 - torch.clamp(final_q_err, 0.0, 1.0))

        return {
            "initial_state": initial_state,
            "final_state": final_state,
            "initial_q_err": initial_q_err.item(),
            "final_q_err": final_q_err.item(),
            "adaptation_trajectory": adaptation_traj,
            "metrics": final_metrics,
            "positive_relief": restored_positive_relief,
            "negative_mold_depth": final_q_err.item(),
            "provenance_history_len": len(self.provenance_log)
        }

    def get_causal_provenance(self) -> List[Dict[str, Any]]:
        """
        Returns full history of dial tuning trajectories for reverse-engineering causal pathways.
        """
        return self.provenance_log
