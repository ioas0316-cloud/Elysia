"""
test_unified_scale_rotor_agi_engine.py

Verifies ELYSIA's Unified Scale-Rotor AGI Engine that absorbs and re-creates:
1. Diffusion (`White-to-Blue Multi-Scale Rotor Condensation` in 3 hierarchical turns:
   `L0 White Noise -> L12 Macro Atmosphere -> L6 Meso 3D Relief -> L2 Micro 4K Grain`).
2. Multimodal AI (`Voice Wave + Heart Intent -> Visual Scale-Rotor Phase Locking`).
3. DLSS (`4x Spatial Super-Resolution 64x64 -> 256x256` via internalized 4K scale ladder
   + `4x Temporal Frame Generation 4 keyframes -> 16 continuous frames`).
"""

from __future__ import annotations

from pathlib import Path

from core.sensory.unified_scale_rotor_agi_engine import UnifiedScaleRotorAGIEngine


def test_unified_scale_rotor_diffusion_multimodal_and_dlss() -> None:
    engine = UnifiedScaleRotorAGIEngine()
    report = engine.run_unified_agi_absorption_showcase()

    # 1. Multimodal Voice + Heart coupling
    mm = report["multimodal_coupling"]
    assert "ASCENDING_FLOW" in mm["perceived_sound_qualities"]
    assert mm["rotor_sun_warmth_boost"] > 0.35

    # 2. 3-Turn Scale-Rotor Diffusion from White Noise
    d_steps = report["diffusion_steps"]
    assert len(d_steps) == 4
    assert d_steps[0].entropy_dispersion_yellow == 1.00
    assert d_steps[-1].entropy_dispersion_yellow < 0.10
    assert d_steps[-1].structural_order_blue > 0.90

    # 3. Scale-Rotor DLSS 4x Super-Resolution & 4x Temporal Frame Generation
    dlss = report["dlss_report"]
    assert dlss.low_res_input_shape == (64, 64, 3)
    assert dlss.super_res_output_shape == (256, 256, 3)
    assert dlss.highfreq_detail_gain_ratio > 1.12
    assert dlss.ground_truth_structural_ssim_proxy > 0.90
    assert dlss.temporal_jerk_reduction_ratio > 2.0

    # 4. Visual dashboard & generated video exist
    dash_path = Path(report["output_dashboard_path"])
    vid_path = Path(report["output_dlss_video_path"])
    assert dash_path.exists() and dash_path.stat().st_size > 100000
    assert vid_path.exists() and vid_path.stat().st_size > 30000
