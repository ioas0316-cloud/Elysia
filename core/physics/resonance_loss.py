import torch
import torch.nn as nn
import torch.nn.functional as F

class ResonanceLoss(nn.Module):
    """
    Resonance Loss enforcing Phase Alignment (cos_sim(k_ext, ∇d)), InfoNCE Contrastive Standing Wave Energy, and Eikonal Regularization.
    """
    def __init__(self, tau=0.1, l_phase=1.0, l_contrast=0.5, l_eikonal=0.1):
        super().__init__()
        self.tau = tau
        self.l_phase = l_phase
        self.l_contrast = l_contrast
        self.l_eikonal = l_eikonal

    def forward(self, query_pos, sdf_fn, wavevector_ext, target_pair_mask):
        """
        query_pos: [Batch, dim]
        sdf_fn: Continuous SDF function
        wavevector_ext: [Batch, dim]
        target_pair_mask: [Batch, Batch]
        """
        query_pos.requires_grad_(True)

        d_val = sdf_fn(query_pos)
        grad_d = torch.autograd.grad(
            outputs=d_val.sum(),
            inputs=query_pos,
            create_graph=True,
            retain_graph=True
        )[0]

        grad_norm = torch.norm(grad_d, dim=-1, keepdim=True) + 1e-8
        grad_unit = grad_d / grad_norm

        k_norm = torch.norm(wavevector_ext, dim=-1, keepdim=True) + 1e-8
        k_unit = wavevector_ext / k_norm
        cos_sim = torch.sum(k_unit * grad_unit, dim=-1)
        loss_phase = torch.mean(1.0 - cos_sim)

        decay = torch.exp(-torch.abs(d_val) / 0.1)
        spatial_phase = torch.sum(query_pos * wavevector_ext, dim=-1, keepdim=True)
        standing_amp = (2.0 * decay * torch.cos(spatial_phase)).squeeze(-1)

        energy_sim_matrix = standing_amp.unsqueeze(0) * standing_amp.unsqueeze(1) / self.tau
        log_prob = F.log_softmax(energy_sim_matrix, dim=-1)
        loss_contrastive = -torch.mean(torch.sum(target_pair_mask * log_prob, dim=-1))

        loss_eikonal = torch.mean((grad_norm - 1.0)**2)

        total_loss = (self.l_phase * loss_phase +
                      self.l_contrast * loss_contrastive +
                      self.l_eikonal * loss_eikonal)

        return total_loss, {
            "loss_phase": loss_phase.item(),
            "loss_contrast": loss_contrastive.item(),
            "loss_eikonal": loss_eikonal.item()
        }
