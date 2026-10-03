import torch
import torch.nn as nn
import torch.nn.functional as F

class RotorPLLStandingWaveReadout(nn.Module):
    """
    Rotor-based rotational geodesic exploration and Phase-Locking Loop (PLL) standing wave readout layer.
    """
    def __init__(self, dim=64, num_bivectors=16, k_gain=2.5):
        super().__init__()
        self.dim = dim
        self.num_bivectors = num_bivectors
        self.k_gain = k_gain # PLL Loop Gain

        # Anti-symmetric Bivector weights for Clifford rotation
        self.bivector_weights = nn.Parameter(torch.randn(num_bivectors, dim, dim) * 0.02)

        # Standing wave eigen-wavevector k
        self.wave_k = nn.Parameter(torch.randn(dim) * 0.5)

        # Readout projection head
        self.out_proj = nn.Linear(dim, dim)

    def get_skew_bivector(self, idx):
        """ Anti-symmetric matrix B^T = -B """
        B = self.bivector_weights[idx]
        return B - B.transpose(-1, -2)

    def forward(self, query_pos, query_dir, sdf_fn, max_steps=16, eps=1e-3):
        """
        query_pos: [Batch, dim]
        query_dir: [Batch, dim]
        sdf_fn: Continuous SDF distance function d(x)
        """
        batch_size = query_pos.size(0)
        curr_pos = query_pos.clone()
        curr_dir = query_dir / (torch.norm(query_dir, dim=-1, keepdim=True) + 1e-8)

        theta = torch.zeros(batch_size, 1, device=query_pos.device)
        accum_standing_wave = torch.zeros_like(query_pos)

        for step in range(max_steps):
            # 1. SDF distance and Autograd gradient (Phase)
            curr_pos.requires_grad_(True)
            dist = sdf_fn(curr_pos) # [Batch, 1]
            grad = torch.autograd.grad(dist.sum(), curr_pos, create_graph=True)[0]
            curr_pos = curr_pos.detach()

            # 2. PLL Phase Error
            grad_phase = torch.atan2(grad[:, 1:2], grad[:, 0:1] + 1e-8)
            phase_error = grad_phase - theta

            # 3. Rotor phase angle update: dθ/dt = ω0 + K * sin(Δφ)
            d_theta = 0.1 + self.k_gain * torch.sin(phase_error)
            theta = theta + d_theta

            # 4. Rotor update (Bivector Rotation: R = cos(θ/2) - B*sin(θ/2))
            B_matrix = self.get_skew_bivector(step % self.num_bivectors)
            rot_matrix = torch.cos(theta / 2.0).unsqueeze(-1) * torch.eye(self.dim, device=query_pos.device) \
                         + torch.sin(theta / 2.0).unsqueeze(-1) * B_matrix

            # Rotate Direction Vector
            curr_dir = torch.bmm(rot_matrix, curr_dir.unsqueeze(-1)).squeeze(-1)
            curr_dir = curr_dir / (torch.norm(curr_dir, dim=-1, keepdim=True) + 1e-8)

            # 5. Standing Wave Envelope Accumulation: A_standing = 2*A0 * exp(-d/σ) * cos(k·x + θ)
            decay = torch.exp(-torch.abs(dist) / 0.1)
            spatial_phase = torch.sum(curr_pos * self.wave_k, dim=-1, keepdim=True)
            standing_amplitude = 2.0 * decay * torch.cos(spatial_phase + theta)

            accum_standing_wave = accum_standing_wave + standing_amplitude * curr_dir

            # 6. Ray Marching Step
            curr_pos = curr_pos + curr_dir * torch.clamp(dist, min=eps)

        # 7. Extract final latent embedding from standing wave interference pattern
        readout_vector = self.out_proj(accum_standing_wave)
        return readout_vector, curr_pos
