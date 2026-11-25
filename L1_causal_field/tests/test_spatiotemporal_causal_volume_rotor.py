import torch
import torch.nn.functional as F

from core.topology.spatiotemporal_causal_volume_rotor import SpatiotemporalCausalVolumeRotor


def test_3d_4d_phenomenal_volume_and_causal_of_causal_meta_rotor_on_gpu() -> None:
    rotor = SpatiotemporalCausalVolumeRotor(dim=128)
    dev = rotor.device

    # Build T=8 connected 2D frames `[8, 16, 16, 128]` (e.g., 8 consecutive A4 pages or 8 video frames)
    g = torch.Generator(device="cpu").manual_seed(101)
    base_a = F.normalize(torch.randn(16, 16, 128, generator=g), dim=-1).to(dev)
    base_b = F.normalize(torch.randn(16, 16, 128, generator=g), dim=-1).to(dev)
    frames = [
        F.normalize((1.0 - t / 7.0) * base_a + (t / 7.0) * base_b, dim=-1)
        for t in range(8)
    ]

    # 1) Open Spatiotemporal Volume (book / video time-arrow)
    vol_open = rotor.build_volume_from_2d_frames(frames, closed_globe_topology=False)
    assert vol_open.volume.shape == (8, 16, 16, 128)
    assert vol_open.theta1_laws.shape == (4, 128)
    assert vol_open.theta2_meta.shape == (3, 128)

    # Rotate ONE Meta-Causal axis in Theta_2 -> MUST bend First-Order Laws Theta_1 AND co-evolve ALL 8 frames!
    cf = rotor.simulate_counterfactual_meta_causal_shift(vol_open, meta_axis_idx=1, steps=8)
    assert cf["theta1_mean_law_bend"] > 0.05
    assert cf["all_frames_co_evolved"] is True
    assert cf["final_temporal_continuity_cos"] > 0.80

    # 2) Closed Globe Topology (spherical/toroidal closure where frame T-1 wraps seamlessly to frame 0!)
    vol_globe = rotor.build_volume_from_2d_frames(frames, closed_globe_topology=True)
    rep_g = {}
    for _ in range(10):
        rep_g = rotor.step_coupled_volume_and_causal_rotors(vol_globe, dt=0.35)
    assert rep_g["closed_globe_topology"] is True
    # Closed globe continuity pulls orthogonal boundary frames (0.00 -> 0.82+) into continuous closure!
    assert rep_g["globe_closure_cos"] > 0.80
    assert rep_g["globe_closure_cos"] > cf["final_globe_closure_cos"]
    assert rep_g["temporal_continuity_cos"] > 0.90
