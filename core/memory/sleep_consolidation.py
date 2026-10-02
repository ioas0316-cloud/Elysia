import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFMemoryField2D(nn.Module):
    """
    Continuous SDF Latent Memory Grid with Idle-Time Memory Consolidation via Laplacian Diffusion PDE.
    """
    def __init__(self, grid_res=128):
        super().__init__()
        self.res = grid_res

        # [1, 1, H, W] Continuous Latent SDF Field
        self.sdf_field = nn.Parameter(torch.ones(1, 1, grid_res, grid_res) * 5.0)

        # Saliency Field: Tracks important/frequently accessed memory regions
        self.register_buffer("saliency_field", torch.zeros(1, 1, grid_res, grid_res))

        # 2D discrete Laplacian Kernel (5-point stencil)
        laplacian_kernel = torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("laplacian_kernel", laplacian_kernel)

        # Sobel Kernel for Eikonal Constraint (|∇d| computation)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def write_wake_memory(self, center_x, center_y, radius, saliency_weight=1.0, noise_std=0.3):
        """
        Record new knowledge during wakefulness using smooth minimum (smin) with micro-noise artifacts.
        """
        y = torch.linspace(-1, 1, self.res, device=self.sdf_field.device)
        x = torch.linspace(-1, 1, self.res, device=self.sdf_field.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        new_sdf = torch.sqrt((grid_x - center_x)**2 + (grid_y - center_y)**2) - radius
        new_sdf = new_sdf.unsqueeze(0).unsqueeze(0)

        wake_noise = torch.randn_like(new_sdf) * noise_std
        noisy_new_sdf = new_sdf + wake_noise

        k = 5.0
        diff = self.sdf_field - noisy_new_sdf
        smin_val = torch.minimum(self.sdf_field, noisy_new_sdf) - (1.0 / k) * torch.log(1.0 + torch.exp(-k * torch.abs(diff)))

        with torch.no_grad():
            self.sdf_field.copy_(smin_val)
            core_mask = (torch.abs(new_sdf) < 0.1).float()
            self.saliency_field.add_(core_mask * saliency_weight)

    def sleep_consolidation_step(self, nu=0.05, gamma=0.02, dt=0.1):
        """
        Execute 1 step of asynchronous Laplacian Diffusion PDE during idle-time sleep state.
        ∂d/∂t = ν_eff · ∇²d + γ · S · (1 - |∇d|)
        """
        laplacian = F.conv2d(self.sdf_field, self.laplacian_kernel, padding=1)

        grad_x = F.conv2d(self.sdf_field, self.sobel_x, padding=1)
        grad_y = F.conv2d(self.sdf_field, self.sobel_y, padding=1)
        grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        eikonal_loss = 1.0 - grad_norm

        effective_nu = nu / (1.0 + self.saliency_field)
        d_sdf_dt = effective_nu * laplacian + gamma * self.saliency_field * eikonal_loss

        with torch.no_grad():
            self.sdf_field.add_(d_sdf_dt * dt)
