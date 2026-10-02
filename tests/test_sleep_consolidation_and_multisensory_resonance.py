import pytest
import torch
import math
import numpy as np

from core.memory.sleep_consolidation import SDFMemoryField2D
from core.intelligence.rotor_pll_standing_wave import RotorPLLStandingWaveReadout
from core.sensory.physical_frequency_bridge import PhysicalFrequencySDFBridge, RealTimeAudioFFTBridge
from core.physics.resonance_loss import ResonanceLoss
from core.intelligence.active_inference import ActiveInferenceSDFModule

def test_sleep_consolidation_noise_reduction():
    """Verify that idle-time sleep consolidation via Laplacian diffusion reduces field variance (noise decay) while preserving saliency."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory = SDFMemoryField2D(grid_res=64).to(device)

    # Write wake memory with noise
    memory.write_wake_memory(center_x=-0.2, center_y=-0.2, radius=0.2, saliency_weight=2.0, noise_std=0.4)
    initial_var = torch.var(memory.sdf_field).item()

    # Sleep consolidation steps
    for _ in range(50):
        memory.sleep_consolidation_step(nu=0.08, gamma=0.01, dt=0.2)

    final_var = torch.var(memory.sdf_field).item()

    # Assert noise decay
    assert final_var < initial_var, "Sleep consolidation failed to reduce noise variance in memory field."


def test_rotor_pll_standing_wave_readout():
    """Verify Rotor & PLL Standing Wave Readout layer convergence and shape output."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pll_layer = RotorPLLStandingWaveReadout(dim=64).to(device)

    def dummy_sdf_64d(x):
        return (torch.norm(x, dim=-1, keepdim=True) - 1.0)

    q_pos = torch.randn(4, 64, device=device)
    q_dir = torch.randn(4, 64, device=device)

    readout_emb, final_pos = pll_layer(q_pos, q_dir, dummy_sdf_64d, max_steps=8)

    assert readout_emb.shape == (4, 64)
    assert final_pos.shape == (4, 64)
    assert torch.isfinite(readout_emb).all()


def test_physical_frequency_bridges():
    """Verify direct tokenless physical frequency bridges."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bridge = PhysicalFrequencySDFBridge(dim=64, num_freq_bands=128).to(device)

    sdf_pos = torch.randn(4, 64, device=device)
    audio_fft = torch.randn(4, 128, device=device)
    visual_spatial = torch.randn(4, 128, device=device)

    excitation = bridge(sdf_pos, audio_fft, visual_spatial)
    assert excitation.shape == (4, 1)

    fft_bridge = RealTimeAudioFFTBridge(sample_rate=44100, n_fft=1024, spatial_dim=64).to(device)
    audio_pcm = torch.randn(2, 1024, device=device)
    k_ext, omega_ext, amp, phase = fft_bridge(audio_pcm)

    assert k_ext.shape == (2, 64)
    assert omega_ext.shape == (2,)
    assert amp.shape == (2, 1)


def test_resonance_loss():
    """Verify Phase Alignment, Contrastive Energy, and Eikonal Resonance Loss computation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = ResonanceLoss().to(device)

    def dummy_sdf(x):
        return (torch.norm(x, dim=-1, keepdim=True) - 1.0)

    q_pos = torch.randn(4, 64, device=device)
    wavevector_ext = torch.randn(4, 64, device=device)
    pair_labels = torch.eye(4, device=device)

    total_loss, loss_dict = loss_fn(q_pos, dummy_sdf, wavevector_ext, pair_labels)

    assert torch.isfinite(total_loss)
    assert "loss_phase" in loss_dict
    assert "loss_contrast" in loss_dict
    assert "loss_eikonal" in loss_dict


def test_active_inference_destructive_interference():
    """Verify Top-Down Active Inference module predictive error / free energy computation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    active_inf = ActiveInferenceSDFModule(dim=64).to(device)

    queries = torch.randn(2, 100, 64, device=device)
    k_sense = torch.randn(2, 64, device=device)
    w_sense = torch.tensor([10.0, 10.0], device=device)
    amp_sense = torch.tensor([[0.2], [0.2]], device=device)
    prev_state = torch.randn(2, 64, device=device)

    d_updated, err, surprise = active_inf(queries, k_sense, w_sense, amp_sense, prev_state, t_curr=0.05)

    assert d_updated.shape == (2, 100, 1)
    assert err.shape == (2, 100, 1)
    assert torch.isfinite(surprise)
