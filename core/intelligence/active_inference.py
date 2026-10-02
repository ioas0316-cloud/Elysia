import torch
import torch.nn as nn
import math

class ActiveInferenceSDFModule(nn.Module):
    """
    Active Inference module using Top-down prediction wave & Bottom-up sensory wave destructive interference (Predictive Error).
    """
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.topdown_predictor = nn.Sequential(
            nn.Linear(dim, 128),
            nn.SiLU(),
            nn.Linear(128, dim + 2) # [k_pred (dim), omega_pred (1), amp_pred (1)]
        )

    def forward(self, query_x, sensory_k, sensory_omega, sensory_amp, prev_latent_state, t_curr, sigma=0.15):
        """
        query_x: [Batch, N_points, dim]
        sensory_k: [Batch, dim]
        prev_latent_state: [Batch, dim]
        """
        pred_params = self.topdown_predictor(prev_latent_state)
        k_pred = pred_params[:, :self.dim]
        omega_pred = pred_params[:, self.dim:self.dim+1].squeeze(-1)
        amp_pred = torch.sigmoid(pred_params[:, self.dim+1:])

        k_dot_x_sense = torch.matmul(query_x, sensory_k.unsqueeze(-1))
        phase_sense = k_dot_x_sense - (sensory_omega.view(-1, 1, 1) * t_curr) % (2 * math.pi)
        wave_sense = sensory_amp.unsqueeze(1) * torch.cos(phase_sense)

        k_dot_x_pred = torch.matmul(query_x, k_pred.unsqueeze(-1))
        phase_pred = k_dot_x_pred - (omega_pred.view(-1, 1, 1) * t_curr) % (2 * math.pi)
        wave_pred = amp_pred.unsqueeze(1) * torch.cos(phase_pred)

        wave_error = wave_sense - wave_pred

        d_base = torch.norm(query_x, dim=-1, keepdim=True) - 1.0
        mask = torch.exp(-torch.abs(d_base) / sigma)

        d_updated = d_base + mask * wave_error
        free_energy = torch.mean(wave_error ** 2)

        return d_updated, wave_error, free_energy
