import torch
import torch.nn as nn
import math

class GeometricFoldingEngine(nn.Module):
    """
    Substrate 1-Vector Signals -> Geometric Product Folding -> Upper Virtual Memory Volume
    Clifford Space Cl(3,0) : 8-Blade Multivector Dimensions
    Indices:
      0: Scalar (Grade-0)
      1, 2, 3: Vector e1, e2, e3 (Grade-1)
      4, 5, 6: Bivector e12, e23, e31 (Grade-2)
      7: Pseudoscalar e123 (Grade-3)
    """
    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim
        self.bivector_indices = [4, 5, 6]
        # Reversion mask for Cl(3,0): ~A reverses grade-2 and grade-3 sign conventions where applicable
        self.register_buffer("reversion_mask", torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]))

    def compute_bivector_wedge(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        v1 ^ v2 (Outer Product to form Bivector components)
        Input: v1, v2 tensors of shape (..., 8) or (..., 4+) containing 1-vector parts at indices 1, 2, 3
        """
        B = torch.zeros_like(v1)
        # Outer products of 1-vectors (e1, e2, e3):
        # e12 (idx 4) = v1_e1 * v2_e2 - v1_e2 * v2_e1
        # e23 (idx 5) = v1_e2 * v2_e3 - v1_e3 * v2_e2
        # e31 (idx 6) = v1_e3 * v2_e1 - v1_e1 * v2_e3
        B[..., 4] = v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1]  # e12
        B[..., 5] = v1[..., 2] * v2[..., 3] - v1[..., 3] * v2[..., 2]  # e23
        B[..., 6] = v1[..., 3] * v2[..., 1] - v1[..., 1] * v2[..., 3]  # e31
        return B

    def generate_rotor(self, Bivector: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        R = cos(theta/2) - B_hat * sin(theta/2)
        Constructs Clifford Rotor for rotation in bivector plane.
        """
        if theta.dim() < Bivector.dim():
            theta = theta.unsqueeze(-1)

        bivec_part = Bivector[..., self.bivector_indices]
        norm_B = torch.norm(bivec_part, dim=-1, keepdim=True) + 1e-8
        B_hat = Bivector / norm_B

        R = torch.zeros_like(Bivector)
        R[..., 0] = torch.cos(theta / 2.0).squeeze(-1)  # Scalar Part (Grade-0)

        # Bivector part gets -B_hat * sin(theta/2)
        sin_half = torch.sin(theta / 2.0)
        R[..., self.bivector_indices] = -B_hat[..., self.bivector_indices] * sin_half
        return R

    def compute_coherence(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Computes Phase Coherence metric <v1, ~v2>_0 (Grade-0 Inner Product)
        """
        reversion_mask = self.reversion_mask.to(v1.device)
        return torch.sum(v1 * v2 * reversion_mask, dim=-1)

    def unfold(self, V_promoted: torch.Tensor, Rotor: torch.Tensor) -> torch.Tensor:
        """
        Applies reverse rotor R_tilde to unfold upper virtual memory volume back to substrate signals.
        R_tilde * v * R transformation in Cl(3,0)
        R = s + b12 e12 + b23 e23 + b31 e31
        R_tilde = s - b12 e12 - b23 e23 - b31 e31
        """
        s = Rotor[..., 0:1]
        b12 = Rotor[..., 4:5]
        b23 = Rotor[..., 5:6]
        b31 = Rotor[..., 6:7]

        # Extract 1-vector components v = v1 e1 + v2 e2 + v3 e3
        v1 = V_promoted[..., 1:2]
        v2 = V_promoted[..., 2:3]
        v3 = V_promoted[..., 3:4]

        # 3D Vector rotation sandwich formula for R_tilde * v * R
        # Rotated vector components
        u1 = (s**2 + b12**2 - b23**2 + b31**2) * v1 + 2 * (b12 * b23 - s * b31) * v2 + 2 * (b12 * b31 + s * b23) * v3
        u2 = 2 * (b12 * b23 + s * b31) * v1 + (s**2 - b12**2 + b23**2 - b31**2) * v2 + 2 * (b23 * b31 - s * b12) * v3
        u3 = 2 * (b12 * b31 - s * b23) * v1 + 2 * (b23 * b31 + s * b12) * v2 + (s**2 - b12**2 - b23**2 + b31**2) * v3

        unfolded = torch.zeros_like(V_promoted)
        unfolded[..., 0:1] = V_promoted[..., 0:1]
        unfolded[..., 1:2] = u1
        unfolded[..., 2:3] = u2
        unfolded[..., 3:4] = u3
        return unfolded

    def forward(self, v1: torch.Tensor, v2: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        v1, v2: (Batch, 8) or (..., 8) 1-Vector signals / Multivectors
        theta: rotation angle
        Returns: Folded Multivector Volume promoted to Virtual Memory Space
        """
        # 1. Bivector & Rotor Generation
        B_12 = self.compute_bivector_wedge(v1, v2)
        Rotor = self.generate_rotor(B_12, theta)

        # 2. Geometric Folding Action (Scalar + Bivector + Trivector Expansion)
        reversion_mask = self.reversion_mask.to(v1.device)
        scalar_part = torch.sum(v1 * v2 * reversion_mask, dim=-1, keepdim=True)

        volume_bivector = B_12
        # Pseudoscalar e123 = v1_e1 * B_e23 + v1_e2 * B_e31 + v1_e3 * B_e12
        trivector_pseudoscalar = (
            v1[..., 1] * B_12[..., 5] + v1[..., 2] * B_12[..., 6] + v1[..., 3] * B_12[..., 4]
        ).unsqueeze(-1)

        # 3. Assemble Multivector Object (Promoted to Virtual Memory)
        V_promoted = torch.zeros_like(v1)
        V_promoted[..., 0:1] = scalar_part                                     # Grade-0 (Scalar / Phase Lock)
        V_promoted[..., 1:4] = v1[..., 1:4] + v2[..., 1:4]                      # Grade-1 (Rotated Vector Field)
        V_promoted[..., 4:7] = volume_bivector[..., 4:7]                       # Grade-2 (Oriented Area Volume)
        V_promoted[..., 7:8] = trivector_pseudoscalar                          # Grade-3 (Top Volume Pseudoscalar)

        return V_promoted
