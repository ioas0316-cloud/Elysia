import torch
import torch.nn as nn

class CliffordPhaseLockKernelFallback(nn.Module):
    """
    Pure PyTorch / CPU Fallback implementation of Clifford Multivector Phase-Lock Relaxation (CMPLR).
    Uses vectorized PyTorch tensor operations equivalent to the CUDA C++ kernel logic.
    """
    def __init__(self, Psi, row_ptr, col_ind, K_tensors, eps=1e-8):
        super().__init__()
        # State Multivector Tensor: [N, 8]
        # index: 0=Scalar, 1..3=Vector(e1,e2,e3), 4..6=Bivector(e12,e23,e31), 7=Pseudoscalar
        self.Psi = nn.Parameter(Psi) if not isinstance(Psi, nn.Parameter) else Psi
        self.register_buffer("row_ptr", row_ptr)
        self.register_buffer("col_ind", col_ind)
        self.K_tensors = nn.Parameter(K_tensors) if not isinstance(K_tensors, nn.Parameter) else K_tensors
        self.eps = eps

    def _reverse(self, Psi):
        """Clifford reverse operator ~Psi"""
        mask = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], device=Psi.device, dtype=Psi.dtype)
        return Psi * mask

    def _geometric_product_grade2(self, Psi_b, Reverse_Psi_a):
        """
        Extract Grade-2 Bivector (e12, e23, e31) components from geometric product Psi_b * ~Psi_a.
        """
        b_s, b_v, b_b, b_p = Psi_b[:, 0], Psi_b[:, 1:4], Psi_b[:, 4:7], Psi_b[:, 7]
        a_s, a_v, a_b, a_p = Reverse_Psi_a[:, 0], Reverse_Psi_a[:, 1:4], Reverse_Psi_a[:, 4:7], Reverse_Psi_a[:, 7]

        e12 = (b_s * a_b[:, 0] + b_b[:, 0] * a_s) + (b_v[:, 0] * a_v[:, 1] - b_v[:, 1] * a_v[:, 0]) - (b_b[:, 1] * a_b[:, 2] - b_b[:, 2] * a_b[:, 1])
        e23 = (b_s * a_b[:, 1] + b_b[:, 1] * a_s) + (b_v[:, 1] * a_v[:, 2] - b_v[:, 2] * a_v[:, 1]) - (b_b[:, 2] * a_b[:, 0] - b_b[:, 0] * a_b[:, 2])
        e31 = (b_s * a_b[:, 2] + b_b[:, 2] * a_s) + (b_v[:, 2] * a_v[:, 0] - b_v[:, 0] * a_v[:, 2]) - (b_b[:, 0] * a_b[:, 1] - b_b[:, 1] * a_b[:, 0])

        return torch.stack([e12, e23, e31], dim=-1)

    def _project_spin(self, Psi):
        """Gauge Fixer: S^3 Spin manifold surface projection operator using Clifford reversion <Psi * ~Psi>_0 = 1"""
        rev_Psi = self._reverse(Psi)
        scalar_norm = torch.abs(torch.sum(Psi * rev_Psi, dim=-1, keepdim=True))
        return Psi / torch.sqrt(scalar_norm + self.eps)

    def step(self, dt=0.01):
        """Single relaxation step execution using Pure PyTorch/CPU logic"""
        rev_Psi = self._reverse(self.Psi)
        N = self.Psi.shape[0]

        total_bivector_torque = torch.zeros(N, 3, device=self.Psi.device, dtype=self.Psi.dtype)

        for a in range(N):
            start = self.row_ptr[a].item()
            end = self.row_ptr[a + 1].item()
            if start == end:
                continue

            neighbors = self.col_ind[start:end]
            K_sub = self.K_tensors[start:end].view(-1, 3, 3)

            delta_ab = self._geometric_product_grade2(self.Psi[neighbors], rev_Psi[a].unsqueeze(0)) # [num_neighbors, 3]
            torque = torch.matmul(K_sub, delta_ab.unsqueeze(-1)).squeeze(-1) # [num_neighbors, 3]
            total_bivector_torque[a] = torch.sum(torque, dim=0)

        theta = torch.norm(total_bivector_torque, dim=-1, keepdim=True) + self.eps
        b_hat = total_bivector_torque / theta

        half_dt_theta = 0.5 * dt * theta
        cos_val = torch.cos(half_dt_theta)
        sin_val = torch.sin(half_dt_theta) / theta

        b12 = total_bivector_torque[:, 0:1] * sin_val
        b23 = total_bivector_torque[:, 1:2] * sin_val
        b31 = total_bivector_torque[:, 2:3] * sin_val

        # Rotor multiplication: Psi_next = (cos + B_hat * sin) * Psi_a
        a_v0 = self.Psi[:, 0:4]
        a_v1 = self.Psi[:, 4:8]

        next_v0 = torch.zeros_like(a_v0)
        next_v1 = torch.zeros_like(a_v1)

        next_v0[:, 0] = cos_val[:, 0] * a_v0[:, 0] - (b12[:, 0] * a_v1[:, 0] + b23[:, 0] * a_v1[:, 1] + b31[:, 0] * a_v1[:, 2])
        next_v0[:, 1] = cos_val[:, 0] * a_v0[:, 1] + (b12[:, 0] * a_v0[:, 2] - b31[:, 0] * a_v0[:, 3])
        next_v0[:, 2] = cos_val[:, 0] * a_v0[:, 2] + (b23[:, 0] * a_v0[:, 3] - b12[:, 0] * a_v0[:, 1])
        next_v0[:, 3] = cos_val[:, 0] * a_v0[:, 3] + (b31[:, 0] * a_v0[:, 1] - b23[:, 0] * a_v0[:, 2])

        next_v1[:, 0] = cos_val[:, 0] * a_v1[:, 0] + (b12[:, 0] * a_v0[:, 0])
        next_v1[:, 1] = cos_val[:, 0] * a_v1[:, 1] + (b23[:, 0] * a_v0[:, 0])
        next_v1[:, 2] = cos_val[:, 0] * a_v1[:, 2] + (b31[:, 0] * a_v0[:, 0])
        next_v1[:, 3] = cos_val[:, 0] * a_v1[:, 3]

        next_Psi = torch.cat([next_v0, next_v1], dim=-1)

        # Apply Spin(3) Gauge Locking Projection
        with torch.no_grad():
            self.Psi.copy_(self._project_spin(next_Psi))

        return self.Psi
