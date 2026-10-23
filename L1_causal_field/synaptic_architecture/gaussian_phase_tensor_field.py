import torch
import torch.nn as nn
from synaptic_architecture.complex_quaternion_pc import quaternion_mul


class GaussianPhaseOscillatorNode(nn.Module):
    """
    3D Gaussian Phase Oscillator Node.

    Attributes:
    - pos (x, y, z): 3D spatial anchor position.
    - phase_q (w, x, y, z): 4D quaternion phase angle representing 3D spatial orientation & phase.
    - scale (sx, sy, sz): Anisotropic 3D spatial extent.
    - opacity (alpha): Translucency / density weight.
    - amplitude (A = scale_mean * opacity): Wavefront impact strength in continuous tensor field.
    """
    def __init__(self, num_nodes: int, device: torch.device = None):
        super().__init__()
        self.num_nodes = num_nodes

        # 1. 3D Position Anchor
        self.pos = nn.Parameter(torch.randn(num_nodes, 3, device=device))

        # 2. 4D Quaternion Phase Angle
        q_raw = torch.randn(num_nodes, 4, device=device)
        self.phase_q = nn.Parameter(q_raw / torch.norm(q_raw, dim=-1, keepdim=True))

        # 3. Anisotropic Scale and Opacity
        self.scale = nn.Parameter(torch.abs(torch.randn(num_nodes, 3, device=device)) * 0.1 + 0.05)
        self.opacity = nn.Parameter(torch.sigmoid(torch.randn(num_nodes, 1, device=device)))

    def get_normalized_quaternions(self) -> torch.Tensor:
        """Returns unit quaternions [num_nodes, 4]."""
        return self.phase_q / (torch.norm(self.phase_q, dim=-1, keepdim=True) + 1e-8)

    def get_amplitudes(self) -> torch.Tensor:
        """Computes wave amplitude A = mean(scale) * opacity."""
        mean_scale = torch.mean(self.scale, dim=-1, keepdim=True)
        return mean_scale * torch.sigmoid(self.opacity)


class GaussianPhaseTensorField(nn.Module):
    """
    Continuous 3D Gaussian Quaternion Phase Tensor Field Substrate.

    Evaluates continuous spatial field potential T(x) = sum_i A_i * exp(-0.5 * ||x - pos_i||^2 / sigma_i^2) * q_i.
    Performs autopoietic self-splitting when local mismatch energy exceeds viability threshold tau.
    """
    def __init__(self, initial_nodes: int = 16, split_threshold: float = 0.5):
        super().__init__()
        self.split_threshold = split_threshold
        self.nodes = GaussianPhaseOscillatorNode(initial_nodes)

    def evaluate_field_at_points(self, query_points: torch.Tensor) -> torch.Tensor:
        """
        Evaluates 4D quaternion phase tensor field T(x) at query points.
        query_points: [num_queries, 3]
        returns: [num_queries, 4] quaternion phase field
        """
        # query_points: [M, 3], pos: [N, 3]
        M = query_points.size(0)
        N = self.nodes.num_nodes

        pos = self.nodes.pos  # [N, 3]
        q = self.nodes.get_normalized_quaternions()  # [N, 4]
        amps = self.nodes.get_amplitudes()  # [N, 1]
        sigma = torch.mean(self.nodes.scale, dim=-1, keepdim=True) + 1e-6  # [N, 1]

        # Distances between query points and node anchors
        diff = query_points.unsqueeze(1) - pos.unsqueeze(0)  # [M, N, 3]
        dist_sq = torch.sum(diff ** 2, dim=-1)  # [M, N]

        # Gaussian spatial falloff kernel
        weight = amps.squeeze(-1).unsqueeze(0) * torch.exp(-0.5 * dist_sq / (sigma.squeeze(-1).unsqueeze(0) ** 2))  # [M, N]

        # Field superposition of quaternion phase waves
        field_q = torch.matmul(weight, q)  # [M, 4]
        field_q_norm = field_q / (torch.norm(field_q, dim=-1, keepdim=True) + 1e-8)
        return field_q_norm

    def compute_torque_errors(self, target_q_field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes local quaternion torque mismatch e_i = target_q x q_curr^{-1}.
        target_q_field: [num_nodes, 4] target phase angle for each node anchor position.
        returns: (torque_vectors [num_nodes, 3], mismatch_energy [num_nodes])
        """
        curr_q = self.nodes.get_normalized_quaternions()  # [N, 4]
        q_inv = curr_q.clone()
        q_inv[..., 1:] = -q_inv[..., 1:]  # Quaternion conjugate / inverse

        # err_q = target_q x q_inv
        err_q = quaternion_mul(target_q_field, q_inv)

        # Torque vector is the 3D imaginary component of quaternion residual
        sign_w = torch.where(err_q[..., 0:1] >= 0.0, 1.0, -1.0)
        torque = 2.0 * sign_w * err_q[..., 1:4]  # [N, 3]
        mismatch_energy = torch.norm(torque, dim=-1)  # [N]

        return torque, mismatch_energy

    def update_phase_and_autopoietic_split(
        self,
        target_q_field: torch.Tensor,
        gamma: float = 0.1,
        dt: float = 0.1,
    ) -> dict:
        """
        Applies local torque update to node quaternion phases and executes autopoietic cell division
        for nodes whose local phase mismatch energy exceeds split_threshold tau.
        """
        torque, mismatch_energy = self.compute_torque_errors(target_q_field)

        # 1. Update quaternion phase using local torque (No autograd / backprop required)
        curr_q = self.nodes.phase_q.data
        dq = quaternion_mul(
            torch.cat([torch.zeros_like(torque[..., 0:1]), torque], dim=-1),
            curr_q,
        )
        new_q = curr_q + gamma * dt * 0.5 * dq
        self.nodes.phase_q.data = new_q / (torch.norm(new_q, dim=-1, keepdim=True) + 1e-8)

        # 2. Check for Autopoietic Cell Division (Self-Splitting)
        split_mask = mismatch_energy > self.split_threshold
        num_splits = split_mask.sum().item()

        if num_splits > 0:
            with torch.no_grad():
                split_indices = torch.nonzero(split_mask, as_tuple=True)[0]

                # Parent attributes to divide
                p_pos = self.nodes.pos[split_indices]
                p_q = self.nodes.phase_q[split_indices]
                p_scale = self.nodes.scale[split_indices]
                p_opacity = self.nodes.opacity[split_indices]

                # Daughter nodes offset along principal axis
                offset = p_scale * 0.5
                child1_pos = p_pos + offset
                child2_pos = p_pos - offset

                child_scale = p_scale * 0.7  # Scale reduction upon division
                child_opacity = p_opacity * 0.8

                # Append daughter nodes to node parameters
                new_pos = torch.cat([self.nodes.pos.data, child1_pos, child2_pos], dim=0)
                new_q_param = torch.cat([self.nodes.phase_q.data, p_q, p_q], dim=0)
                new_scale = torch.cat([self.nodes.scale.data, child_scale, child_scale], dim=0)
                new_opacity = torch.cat([self.nodes.opacity.data, child_opacity, child_opacity], dim=0)

                # Reconstruct GaussianPhaseOscillatorNode with expanded node count
                new_num_nodes = new_pos.size(0)
                new_nodes = GaussianPhaseOscillatorNode(new_num_nodes, device=self.nodes.pos.device)
                new_nodes.pos = nn.Parameter(new_pos)
                new_nodes.phase_q = nn.Parameter(new_q_param)
                new_nodes.scale = nn.Parameter(new_scale)
                new_nodes.opacity = nn.Parameter(new_opacity)

                self.nodes = new_nodes

        return {
            "num_splits": num_splits,
            "total_nodes": self.nodes.num_nodes,
            "mean_mismatch_energy": mismatch_energy.mean().item(),
        }
