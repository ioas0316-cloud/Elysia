"""
Causal Boundary Tensor & Causal DNA 4-Base Connectors Module.

This module implements CausalBoundaryTensor, redefining standard arithmetic operators
(+, -, *, /) not as static numerical calculations, but as the 4 Base Connectors of
Causal DNA (Binding, Cleavage, Replication, Differentiation), alongside dynamic boundary
logic gates (AND, OR, NOT, XOR) operating on phase resonance and boundary permeability.
"""

from typing import Union
import torch
import numpy as np


class CausalBoundaryTensor:
    """
    Tensor representation of causal boundary dynamics.

    Attributes:
        state (torch.Tensor): Substantive physical/information state tensor.
        phase (torch.Tensor): Boundary permeability, directionality, and phase tensor.
        value_ground (float): Internal intrinsic value ground (0_value).
    """

    def __init__(
        self,
        state: Union[torch.Tensor, np.ndarray, float, list],
        boundary_phase: Union[torch.Tensor, np.ndarray, float, list],
        value_ground: float = 1.0
    ):
        if not isinstance(state, torch.Tensor):
            self.state = torch.tensor(state, dtype=torch.float32)
        else:
            self.state = state.to(torch.float32)

        if not isinstance(boundary_phase, torch.Tensor):
            self.phase = torch.tensor(boundary_phase, dtype=torch.float32)
        else:
            self.phase = boundary_phase.to(torch.float32)

        self.value_ground = float(value_ground)

    def _ensure_causal_tensor(self, other: Union['CausalBoundaryTensor', torch.Tensor, float, int]) -> 'CausalBoundaryTensor':
        if isinstance(other, CausalBoundaryTensor):
            return other
        if isinstance(other, (torch.Tensor, np.ndarray)):
            return CausalBoundaryTensor(other, torch.zeros_like(torch.tensor(other, dtype=torch.float32)), self.value_ground)
        return CausalBoundaryTensor(
            torch.tensor(other, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
            self.value_ground
        )

    def add(self, other: Union['CausalBoundaryTensor', torch.Tensor, float]) -> 'CausalBoundaryTensor':
        """
        Addition (+): Causal Chain Binding & Synthesis (Boundary Merging).
        Combines independent causal entities self and other into a higher-order causal structure
        modulated by phase resonance.
        """
        other_cb = self._ensure_causal_tensor(other)
        phase_diff = torch.abs(self.phase - other_cb.phase)
        resonance = torch.cos(phase_diff)  # Complete integration when phases align
        new_state = self.state + other_cb.state * resonance
        new_phase = (self.phase + other_cb.phase) / 2.0
        return CausalBoundaryTensor(new_state, new_phase, self.value_ground)

    def sub(self, other: Union['CausalBoundaryTensor', torch.Tensor, float]) -> 'CausalBoundaryTensor':
        """
        Subtraction (-): Causal Cleavage & Gradient Formation.
        Cleaves specific bonds in structured causality, creating potential gradients
        and potential diff level flows.
        """
        other_cb = self._ensure_causal_tensor(other)
        gradient = self.state - other_cb.state
        potential_diff = self.phase - other_cb.phase
        return CausalBoundaryTensor(gradient, potential_diff, self.value_ground)

    def mul(self, other: Union['CausalBoundaryTensor', torch.Tensor, float]) -> 'CausalBoundaryTensor':
        """
        Multiplication (*): Causal Replication & Dimensional Transcription.
        Replicates invariant causal structure across dimensions, amplifying
        or scaling state through resonance.
        """
        other_cb = self._ensure_causal_tensor(other)
        if self.state.dim() == 1 and other_cb.state.dim() == 1 and self.state.shape != other_cb.state.shape:
            amplified_state = torch.outer(self.state, other_cb.state)
        else:
            amplified_state = self.state * other_cb.state
        new_phase = self.phase * other_cb.phase
        return CausalBoundaryTensor(amplified_state, new_phase, self.value_ground)

    def div(self, other: Union['CausalBoundaryTensor', torch.Tensor, float]) -> 'CausalBoundaryTensor':
        """
        Division (/): Causal Differentiation & Seed Formation.
        Divides macro causality into autonomous sub-structures, distributing
        independent seeds for new causal generation.
        """
        other_cb = self._ensure_causal_tensor(other)
        safe_other_state = torch.clamp(torch.abs(other_cb.state), min=1e-6) * torch.sign(other_cb.state + 1e-12)
        partitioned = self.state / safe_other_state
        return CausalBoundaryTensor(partitioned, self.phase, self.value_ground)

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        other_cb = self._ensure_causal_tensor(other)
        return other_cb.sub(self)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        other_cb = self._ensure_causal_tensor(other)
        return other_cb.div(self)

    # Dynamic Logic Gates

    def and_gate(self, other: Union['CausalBoundaryTensor', torch.Tensor, float]) -> 'CausalBoundaryTensor':
        """
        AND Gate: Strict Resonance & Inter-fidelity.
        Transmits energy only when both input signals align in direction and phase.
        """
        other_cb = self._ensure_causal_tensor(other)
        phase_diff = torch.abs(self.phase - other_cb.phase)
        resonance = torch.clamp(torch.cos(phase_diff), min=0.0)
        output_state = torch.minimum(self.state, other_cb.state) * resonance
        output_phase = (self.phase + other_cb.phase) / 2.0
        return CausalBoundaryTensor(output_state, output_phase, self.value_ground)

    def or_gate(self, other: Union['CausalBoundaryTensor', torch.Tensor, float]) -> 'CausalBoundaryTensor':
        """
        OR Gate: Boundary Permeability & Acceptance.
        Opens the flow whenever either entity expresses presence, accepting possibilities.
        """
        other_cb = self._ensure_causal_tensor(other)
        phase_diff = torch.abs(self.phase - other_cb.phase)
        permeability = 1.0 - 0.5 * torch.tanh(phase_diff)
        output_state = torch.maximum(self.state, other_cb.state) * permeability
        output_phase = torch.where(self.state >= other_cb.state, self.phase, other_cb.phase)
        return CausalBoundaryTensor(output_state, output_phase, self.value_ground)

    def not_gate(self) -> 'CausalBoundaryTensor':
        """
        NOT Gate: Inversion & Polarity Shift.
        Flips boundary state to create a new flow trigger.
        """
        output_state = 1.0 - self.state
        output_phase = -self.phase
        return CausalBoundaryTensor(output_state, output_phase, self.value_ground)

    def xor_gate(self, other: Union['CausalBoundaryTensor', torch.Tensor, float]) -> 'CausalBoundaryTensor':
        """
        XOR Gate: Differential Sensing & Motion Triggering.
        Activated only when sensing difference / phase gap between entities.
        """
        other_cb = self._ensure_causal_tensor(other)
        state_diff = torch.abs(self.state - other_cb.state)
        phase_gap = torch.abs(self.phase - other_cb.phase)
        motion_trigger = torch.tanh(state_diff + phase_gap)
        return CausalBoundaryTensor(motion_trigger, phase_gap, self.value_ground)

    def __repr__(self) -> str:
        return f"CausalBoundaryTensor(state={self.state.numpy()}, phase={self.phase.numpy()}, value_ground={self.value_ground})"
