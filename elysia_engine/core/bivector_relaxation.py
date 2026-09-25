import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class ElysiaBivectorRelaxation(nn.Module):
    """
    Clifford Algebra Cℓ(3,0) Bivector Tension Self-Relaxation Engine.

    Replaces 1D scalar loss with multidimensional bivector tension Ω_ij
    between nodes in a graph. Applies self-relaxation flows via exponential map
    updates on Lie algebra rotors without global backpropagation.
    """
    def __init__(self, num_nodes: int, num_edges: int, dt: float = 0.01, eta: float = 0.1, omega_break: float = 1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.dt = dt
        self.eta = eta
        self.omega_break = omega_break

    def geometric_product_3d(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Clifford Algebra Cℓ(3,0) 8D Multivector Product (..., 8) x (..., 8) -> (..., 8).
        Multivector basis order: [s, v1, v2, v3, b12, b23, b31, p]
        """
        res = torch.zeros_like(A)
        # Scalar part (Grade 0)
        res[..., 0] = (A[..., 0] * B[..., 0] + A[..., 1] * B[..., 1] + A[..., 2] * B[..., 2] + A[..., 3] * B[..., 3]
                       - A[..., 4] * B[..., 4] - A[..., 5] * B[..., 5] - A[..., 6] * B[..., 6] - A[..., 7] * B[..., 7])

        # Vector part (Grade 1: e1, e2, e3)
        res[..., 1] = (A[..., 0] * B[..., 1] + A[..., 1] * B[..., 0] - A[..., 2] * B[..., 4] + A[..., 4] * B[..., 2]
                       + A[..., 3] * B[..., 6] - A[..., 6] * B[..., 3] - A[..., 5] * B[..., 7] - A[..., 7] * B[..., 5])
        res[..., 2] = (A[..., 0] * B[..., 2] + A[..., 2] * B[..., 0] + A[..., 1] * B[..., 4] - A[..., 4] * B[..., 1]
                       - A[..., 3] * B[..., 5] + A[..., 5] * B[..., 3] - A[..., 6] * B[..., 7] - A[..., 7] * B[..., 6])
        res[..., 3] = (A[..., 0] * B[..., 3] + A[..., 3] * B[..., 0] - A[..., 1] * B[..., 6] + A[..., 6] * B[..., 1]
                       + A[..., 2] * B[..., 5] - A[..., 5] * B[..., 2] - A[..., 4] * B[..., 7] - A[..., 7] * B[..., 4])

        # Bivector part (Grade 2: e12, e23, e31)
        res[..., 4] = (A[..., 0] * B[..., 4] + A[..., 4] * B[..., 0] + A[..., 1] * B[..., 2] - A[..., 2] * B[..., 1]
                       + A[..., 3] * B[..., 7] + A[..., 7] * B[..., 3] - A[..., 5] * B[..., 6] + A[..., 6] * B[..., 5])
        res[..., 5] = (A[..., 0] * B[..., 5] + A[..., 5] * B[..., 0] + A[..., 2] * B[..., 3] - A[..., 3] * B[..., 2]
                       + A[..., 1] * B[..., 7] + A[..., 7] * B[..., 1] - A[..., 6] * B[..., 4] + A[..., 4] * B[..., 6])
        res[..., 6] = (A[..., 0] * B[..., 6] + A[..., 6] * B[..., 0] + A[..., 3] * B[..., 1] - A[..., 1] * B[..., 3]
                       + A[..., 2] * B[..., 7] + A[..., 7] * B[..., 2] - A[..., 4] * B[..., 5] + A[..., 5] * B[..., 4])

        # Pseudoscalar part (Grade 3: e123)
        res[..., 7] = (A[..., 0] * B[..., 7] + A[..., 7] * B[..., 0] + A[..., 1] * B[..., 5] + A[..., 5] * B[..., 1]
                       + A[..., 2] * B[..., 6] + A[..., 6] * B[..., 2] + A[..., 3] * B[..., 4] + A[..., 4] * B[..., 3])

        return res

    def forward(
        self,
        Psi_nodes: torch.Tensor,
        R_edges: torch.Tensor,
        edge_index: torch.Tensor,
        g_edges: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Psi_nodes: (N, 8) Multivector states
        R_edges: (E, 4) Spin(3) Rotors [w, x, y, z]
        edge_index: (2, E) [src, tgt]
        g_edges: (E, 1) Edge metric weights

        Returns:
            Psi_nodes_updated: (N, 8)
            Omega_ij: (E, 3) Edge bivector tension
            g_edges_updated: (E, 1) Updated edge metric (pruned if tension > omega_break)
        """
        src, tgt = edge_index[0], edge_index[1]
        Psi_i = Psi_nodes[src]
        Psi_j = Psi_nodes[tgt]

        # Reversal of Psi_j
        Psi_j_rev = Psi_j.clone()
        # Vectors (1..3) and Bivectors (4..6) negate under Clifford reversal
        Psi_j_rev[..., [1, 2, 3, 4, 5, 6]] *= -1.0

        Mismatch = self.geometric_product_3d(Psi_i, Psi_j_rev)

        # Extract Grade-2 (Bivector Tension Omega_ij: e12, e23, e31)
        Omega_ij = Mismatch[..., [4, 5, 6]] * g_edges  # (E, 3)

        # Dynamic Topology Metric Adjustment (Pruning / Metric shift if tension exceeds threshold)
        tension_norms = torch.norm(Omega_ij, dim=-1, keepdim=True)
        g_edges_updated = torch.where(
            tension_norms > self.omega_break,
            torch.zeros_like(g_edges),
            g_edges
        )

        # Re-weight tension after pruning
        Omega_ij = Omega_ij * (g_edges_updated > 0).float()

        # Scatter Accumulation into target nodes
        Omega_accum = torch.zeros((Psi_nodes.shape[0], 3), device=Psi_nodes.device, dtype=Psi_nodes.dtype)
        Omega_accum.index_add_(0, tgt, Omega_ij)

        # Self-Relaxation Update (Exponential Map on Bivector)
        tension_norm = torch.norm(Omega_accum, dim=-1, keepdim=True) + 1e-8
        axis = Omega_accum / tension_norm
        angle = -self.eta * self.dt * tension_norm

        Psi_nodes_updated = Psi_nodes.clone()
        Psi_nodes_updated[..., [4, 5, 6]] += torch.sin(angle) * axis

        return Psi_nodes_updated, Omega_ij, g_edges_updated


class GaugeCommutativeDiagramSolver(nn.Module):
    """
    Category-Theoretic Commutative Diagram Auto-Solver via Gauge Holonomy Relaxation.
    Solves for unresolved or noisy morphisms R_g by relaxing non-commutative holonomy bivector tension
    Omega_comm -> 0 using Spin(3) exponential map updates without backpropagation.
    """
    def __init__(self, eta: float = 0.05, dt: float = 0.01):
        super().__init__()
        self.eta = eta
        self.dt = dt

    def rotor_multiply(self, R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        """Quaternion / Spinor Product: R1 * R2 (..., 4) [w, x, y, z]"""
        w1, x1, y1, z1 = R1.unbind(-1)
        w2, x2, y2, z2 = R2.unbind(-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    def rotor_reversal(self, R: torch.Tensor) -> torch.Tensor:
        """Rotor Reversal: ~R = [w, -x, -y, -z]"""
        rev_mask = torch.tensor([1.0, -1.0, -1.0, -1.0], device=R.device, dtype=R.dtype)
        return R * rev_mask

    def extract_bivector_tension(self, R_path1: torch.Tensor, R_path2: torch.Tensor) -> torch.Tensor:
        """
        Holonomy bivector tension between composite paths:
        Omega_comm = Grade-2 Projection of (R_path1 * ~R_path2)
        """
        R_mismatch = self.rotor_multiply(R_path1, self.rotor_reversal(R_path2))
        return R_mismatch[..., 1:4]  # (..., 3)

    def exp_map_bivector(self, Omega: torch.Tensor) -> torch.Tensor:
        """Lie Algebra exponential map -> Spin(3) rotor: dR = exp(-0.5 * dt * eta * Omega)"""
        norm = torch.norm(Omega, dim=-1, keepdim=True) + 1e-8
        axis = Omega / norm
        half_angle = -0.5 * self.dt * self.eta * norm

        w = torch.cos(half_angle)
        xyz = torch.sin(half_angle) * axis
        return torch.cat([w, xyz], dim=-1)

    def forward(
        self,
        R_f: torch.Tensor,
        R_g: torch.Tensor,
        R_h: torch.Tensor,
        R_k: torch.Tensor,
        max_iters: int = 100,
        tol: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Relaxes unknown morphism R_g to satisfy path commutativity R_g * R_f == R_k * R_h.

        Returns:
            R_g_relaxed: (..., 4)
            tension_energy: scalar tensor
        """
        R_g_relaxed = R_g.clone()
        tension_energy = torch.tensor(float('inf'), device=R_g.device)

        for step in range(max_iters):
            # 1. Path synthesis
            R_path1 = self.rotor_multiply(R_g_relaxed, R_f)
            R_path2 = self.rotor_multiply(R_k, R_h)

            # 2. Extract holonomy bivector tension
            Omega_comm = self.extract_bivector_tension(R_path1, R_path2)

            tension_energy = torch.mean(torch.sum(Omega_comm ** 2, dim=-1))
            if tension_energy.item() < tol:
                break

            # 3. Spin relaxation update on unknown morphism R_g
            dR = self.exp_map_bivector(Omega_comm)
            R_g_relaxed = self.rotor_multiply(dR, R_g_relaxed)
            R_g_relaxed = R_g_relaxed / torch.norm(R_g_relaxed, dim=-1, keepdim=True)

        return R_g_relaxed, tension_energy


class DimensionalFoldingCl30(nn.Module):
    """
    Cℓ(3,0) 8D Multivector Dimensional Folding Tensor Operator.
    Converts literal 1-Vector conflict energy into Grade-2 Bivector and Grade-3 Pseudoscalar
    components via Hodge Dual and Exterior Product elevation upon semantic context collision.
    """
    def __init__(self, omega_break: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.omega_break = omega_break
        self.eps = eps

    def forward(self, psi: torch.Tensor, omega_sem: torch.Tensor) -> torch.Tensor:
        """
        psi: (..., 8) Multivector state [s, v1, v2, v3, b12, b23, b31, p]
        omega_sem: (..., 3) Semantic bivector tension [B12, B23, B31]

        Returns:
            psi_folded: (..., 8) Folded multivector state
        """
        scalar = psi[..., 0:1]
        v = psi[..., 1:4]
        bivector = psi[..., 4:7]
        pseudoscalar = psi[..., 7:8]

        b_norm = torch.norm(omega_sem, dim=-1, keepdim=True)
        sigma_fold = torch.tanh(b_norm / (self.omega_break + self.eps))

        # Hodge Dual: Grade-1 Vector -> Grade-2 Bivector
        # I * (v1*e1 + v2*e2 + v3*e3) = v3*e12 + v1*e23 + v2*e31
        v1, v2, v3 = v.unbind(-1)
        bivector_folded = torch.stack([v3, v1, v2], dim=-1)

        # Exterior Product (v ^ B): Grade-1 ^ Grade-2 -> Grade-3 Pseudoscalar
        # (v1*B23 + v2*B31 + v3*B12) * e123
        b12, b23, b31 = omega_sem.unbind(-1)
        gamma_fold = (v1 * b23 + v2 * b31 + v3 * b12).unsqueeze(-1)

        # Grade elevation & decay
        v_out = (1.0 - sigma_fold) * v
        b_out = bivector + sigma_fold * bivector_folded + omega_sem
        p_out = pseudoscalar + sigma_fold * gamma_fold

        psi_folded = torch.cat([scalar, v_out, b_out, p_out], dim=-1)
        return psi_folded


class GaugeSymmetryBreakingInquiryEngine(nn.Module):
    """
    Global Gauge Symmetry Breaking Inquiry Generation & Goldstone Mode Restoration Engine.
    Computes Field Strength Tensor curvature F_ijk over triangular holonomy loops (i-j-k)
    to generate active inquiry flux and restoration forces for phase equilibrium.
    """
    def __init__(self, curvature_threshold: float = 0.1):
        super().__init__()
        self.curvature_threshold = curvature_threshold

    def spin_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion / Spinor Product: q1 * q2 (..., 4)"""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ], dim=-1)

    def forward(
        self,
        R_ij: torch.Tensor,
        R_jk: torch.Tensor,
        R_ki: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        R_ij, R_jk, R_ki: (..., 4) Spin(3) rotors along triangular connection loop.

        Returns dict containing:
            inquiry_flux: (..., 3) Field strength bivector tensor F
            inquiry_energy: (..., 1) Curvature magnitude
            inquiry_axis: (..., 3) Unit axis of symmetry breaking
            restoration_force: (..., 3) Gauge restoration torque force
        """
        R_jk_ij = self.spin_multiply(R_jk, R_ij)
        H_ijk = self.spin_multiply(R_ki, R_jk_ij)  # (..., 4)

        bivector_F = H_ijk[..., 1:4]  # Field strength curvature F
        inquiry_energy = torch.norm(bivector_F, dim=-1, keepdim=True)

        is_inquiry_active = (inquiry_energy > self.curvature_threshold).float()
        inquiry_axis = bivector_F / (inquiry_energy + 1e-8)
        restoration_force = -bivector_F * is_inquiry_active

        return {
            "inquiry_flux": bivector_F * is_inquiry_active,
            "inquiry_energy": inquiry_energy * is_inquiry_active,
            "inquiry_axis": inquiry_axis,
            "restoration_force": restoration_force
        }
