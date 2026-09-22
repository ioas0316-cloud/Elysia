"""
Elysia Physics Subsystem: Recurrent Processing Theory (RPT) Dynamic Phase-Lock Control Loop
==========================================================================================
Implements V1 <-> V4 recurrent phase synchronization model with complex/quaternion tensor fields,
Kuramoto phase coherence metrics, and dynamic loop escape conditions based on topological phase-locking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPhaseLockLoop(nn.Module):
    """
    elysia_engine: Phase-Lock Based Local Recurrent Control Loop

    Implements local recurrent coupling between lower-dimensional sensory tensors (Z_low)
    and higher-dimensional conceptual tensors (Z_high) using complex phase representations.
    Convergence is monitored via the Kuramoto phase order parameter S_lock.
    """
    def __init__(
        self,
        dim_low: int = 64,
        dim_high: int = 128,
        max_recurrent_steps: int = 12,
        coherence_threshold: float = 0.85,
        coupling_alpha: float = 0.25
    ):
        super().__init__()
        self.dim_low = dim_low
        self.dim_high = dim_high
        self.max_steps = max_recurrent_steps
        self.threshold = coherence_threshold
        self.alpha = coupling_alpha  # Recurrent phase coupling gain

        # 1. Forward Projection Layer (V1 -> V4)
        self.W_ff = nn.Linear(dim_low * 2, dim_high * 2)  # [Real, Imag] concatenated

        # 2. Top-down Feedback Layer (V4 -> V1)
        self.W_fb = nn.Linear(dim_high * 2, dim_low * 2)

    def _to_complex_tensor(self, flat_tensor: torch.Tensor) -> torch.Tensor:
        """Convert flat tensor [Batch, Dim * 2] to complex tensor [Batch, Dim]."""
        real, imag = torch.chunk(flat_tensor, 2, dim=-1)
        return torch.complex(real, imag)

    def _to_flat_tensor(self, complex_tensor: torch.Tensor) -> torch.Tensor:
        """Convert complex tensor [Batch, Dim] to flat real tensor [Batch, Dim * 2]."""
        return torch.cat([complex_tensor.real, complex_tensor.imag], dim=-1)

    def compute_phase_coherence(self, Z_a: torch.Tensor, Z_b: torch.Tensor) -> torch.Tensor:
        """
        Compute Kuramoto Phase Order Parameter / Coherence S_lock between two complex tensors.
        Returns tensor of shape [Batch].
        """
        phase_a = torch.angle(Z_a)
        phase_b = torch.angle(Z_b)
        phase_diff = phase_a - phase_b

        # Phasor average magnitude
        phasor_diff = torch.exp(1j * phase_diff.to(torch.complex64))
        coherence = torch.abs(torch.mean(phasor_diff, dim=-1))  # [Batch]
        return coherence

    def forward(self, x_input: torch.Tensor):
        """
        Args:
            x_input: [Batch, dim_low * 2] flat real tensor (representing complex sensory input)
        Returns:
            dict: Converged high-level latent tensor state, steps taken, phase locking status, coherence history.
        """
        # --- Stage 1: Feedforward Sweep (V1 -> V4) ---
        Z_low = self._to_complex_tensor(x_input)
        high_raw = self.W_ff(x_input)
        Z_high = self._to_complex_tensor(high_raw)

        step = 0
        is_locked = False
        coherence_trace = []

        # --- Stage 2: Local Recurrent Loop ---
        while step < self.max_steps and not is_locked:
            # 2a. Top-down Feedback Projection
            high_flat = self._to_flat_tensor(Z_high)
            fb_raw = self.W_fb(high_flat)
            Z_fb = self._to_complex_tensor(fb_raw)

            # 2b. Phase Coherence Check
            coherence = self.compute_phase_coherence(Z_low, Z_fb)
            avg_coherence = coherence.mean().item()
            coherence_trace.append(avg_coherence)

            # 2c. Check convergence condition
            if avg_coherence >= self.threshold:
                is_locked = True
                break

            # 2d. Recurrent Phase Coupling Update
            phase_diff = torch.angle(Z_fb) - torch.angle(Z_low)
            dZ_low = torch.polar(torch.abs(Z_fb), phase_diff)

            Z_low = Z_low + self.alpha * dZ_low

            # Re-project to High-Level
            low_flat = self._to_flat_tensor(Z_low)
            Z_high = self._to_complex_tensor(self.W_ff(low_flat))

            step += 1

        # --- Stage 3: Return Phenomenal Locked Tensor Output ---
        return {
            "Z_locked": Z_high,
            "Z_low_converged": Z_low,
            "final_coherence": coherence_trace[-1] if coherence_trace else 1.0,
            "recurrent_steps": step,
            "is_locked": is_locked,
            "coherence_history": coherence_trace
        }
