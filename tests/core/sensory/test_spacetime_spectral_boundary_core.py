r"""
test_spacetime_spectral_boundary_core.py

Zero-mock, zero-LLM verification of ELYSIA's category-free Continuous
Wavelength-Boundary Spacetime Array Core across 7 unlabelled raw arrays:
  - Genuine external `.mp4` video (960 x 540 x 3, 24 frames)
  - Genuine human `.wav` speech (24kHz, 83,520 samples)
  - Genuine native 4K photo (`2160 x 3840 x 3` Yosemite, zero downsampling)
  - Unseen PDE A: 2D Heat Diffusion evolution
  - Unseen PDE B: 2D Coherent Traveling Wave
  - Unseen PDE C: Static Crystalline Boundary Grid
  - Unseen PDE D: Uncoupled White Noise

Proves ELYSIA can autonomously observe, control (`modulate_wavelength_boundary_and_reobserve`),
and re-discriminate structural kinship across completely unfamiliar domains solely from
continuous wavelength order and spacetime causality.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter

from core.sensory.spacetime_spectral_boundary_core import (
    SpacetimeSpectralBoundaryCore,
)

REAL_MP4_PATH = Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")
REAL_WAV_PATH = Path(
    r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data_archive\S1_Body\L1_Foundation\external\CosyVoice\asset\zero_shot_prompt.wav"
)
REAL_4K_YOSEMITE_PATH = Path(
    r"c:\Elysia\demos\real_world_assets\genuine_optical_4k_yosemite.jpg"
)
DASHBOARD_OUT = Path(
    r"c:\Elysia\docs\assets\elysia_spacetime_spectral_boundary_dashboard.png"
)


def _load_real_mp4_array(sample_count: int = 24) -> np.ndarray:
    assert REAL_MP4_PATH.exists()
    cap = cv2.VideoCapture(str(REAL_MP4_PATH))
    assert cap.isOpened()
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(np.linspace(0, max(0, tot - 1), sample_count).astype(int).tolist())
    frames = []
    i = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if i in idxs:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return np.stack(frames, axis=0)


def _generate_unseen_pde_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    h, w, t_steps = 64, 64, 24
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")

    # PDE A: Heat diffusion relaxing fine structure toward macro long wavelengths
    u = ((np.floor(xx * 16) + np.floor(yy * 16)) % 2).astype(np.float32) + 0.45 * rng.standard_normal(
        (h, w)
    ).astype(np.float32)
    diff_frames = []
    for _ in range(t_steps):
        diff_frames.append(u.copy())
        u = gaussian_filter(u, sigma=1.1, mode="wrap").astype(np.float32)
    pde_diffusion = np.stack(diff_frames, axis=0)

    # PDE B: Coherent spatiotemporal traveling wave
    wave_frames = [
        (
            np.sin(2.0 * np.pi * (3.0 * xx + 2.0 * yy) - 0.55 * t)
            + 0.5 * np.cos(2.0 * np.pi * (1.0 * xx - 2.0 * yy) - 0.30 * t)
        ).astype(np.float32)
        for t in range(t_steps)
    ]
    pde_wave = np.stack(wave_frames, axis=0)

    # PDE C: Static sharp crystalline boundary structure
    crystal_slice = (((np.floor(xx * 8) + np.floor(yy * 8)) % 2) * 0.8 + 0.2 * np.sin(6.0 * np.pi * xx)).astype(
        np.float32
    )
    pde_crystal = np.repeat(crystal_slice[None, :, :], t_steps, axis=0)

    # PDE D: Uncoupled spatiotemporal white noise
    pde_noise = rng.standard_normal((t_steps, h, w)).astype(np.float32)

    return {
        "PDE_Diffusion": pde_diffusion,
        "PDE_Wave": pde_wave,
        "PDE_Crystal": pde_crystal,
        "PDE_WhiteNoise": pde_noise,
    }


def test_category_free_wavelength_boundary_rediscrimination_and_control() -> None:
    core = SpacetimeSpectralBoundaryCore()

    # 1. Load 3 genuine external media + 4 unseen physical PDEs (ZERO downsampling on 4K Yosemite)
    mp4_arr = _load_real_mp4_array(24)
    _, wav_raw = wavfile.read(str(REAL_WAV_PATH))
    wav_arr = wav_raw.astype(np.float32)
    if wav_arr.ndim > 1:
        wav_arr = wav_arr.mean(axis=1)
    yosemite_4k = np.asarray(Image.open(REAL_4K_YOSEMITE_PATH).convert("RGB"))
    assert yosemite_4k.shape == (2160, 3840, 3)

    arrays = {
        "Real_MP4_Flower": mp4_arr,
        "Real_WAV_Speech": wav_arr,
        "Real_4K_Yosemite": yosemite_4k,
        **_generate_unseen_pde_arrays(),
    }

    sigs = {eid: core.observe_raw_spacetime_array(arr, eid) for eid, arr in arrays.items()}
    for eid, s in sigs.items():
        print(
            f"\n[{eid:16s}] shape={str(s.raw_shape):18s} ({s.observe_latency_ms:6.1f}ms) | "
            f"bal={s.invariants_dict['long_vs_short_wavelength_balance']:+.3f} "
            f"order={s.invariants_dict['spectral_order_negentropy']:.3f} "
            f"flux={s.invariants_dict['temporal_flux_ratio']:.3f} "
            f"cont={s.invariants_dict['spacetime_phase_continuity']:.3f} "
            f"drift={s.invariants_dict['wavelength_drift_direction']:+.3f} "
            f"spars={s.invariants_dict['spatial_boundary_sparsity']:.3f}"
        )

    # 2. Verify category-free physical separation on the 4 unseen PDE regimes
    assert sigs["PDE_Diffusion"].invariants_dict["wavelength_drift_direction"] > 0.05
    assert sigs["PDE_Diffusion"].invariants_dict["spacetime_phase_continuity"] > 0.85

    assert sigs["PDE_Wave"].invariants_dict["spectral_order_negentropy"] > 0.45
    assert sigs["PDE_Wave"].invariants_dict["temporal_flux_ratio"] > 0.30

    assert sigs["PDE_Crystal"].invariants_dict["temporal_flux_ratio"] < 0.02
    assert sigs["PDE_Crystal"].invariants_dict["spacetime_phase_continuity"] > 0.99

    assert sigs["PDE_WhiteNoise"].invariants_dict["spacetime_phase_continuity"] < 0.10
    assert sigs["PDE_WhiteNoise"].invariants_dict["spectral_order_negentropy"] < 0.30

    # 3. Structural re-discrimination & cross-domain isomorphism matrix
    ordered_ids = list(arrays.keys())
    report = core.discriminate_principle_structures(ordered_ids)
    idx = {eid: i for i, eid in enumerate(report.entity_ids)}
    iso = report.isomorphism_matrix

    print("\n[Discovered Regimes & Nearest Structural Kin]")
    for eid in ordered_ids:
        kin_id, kin_sim = report.nearest_structural_kin[eid]
        print(
            f"  {eid:16s} -> kin={kin_id:16s} ({kin_sim:.3f}) | {report.discovered_regime_descriptions[eid]}"
        )

    # Full 4K static Yosemite photo: invariant time structure & closer to real/crystalline order than white noise
    assert sigs["Real_4K_Yosemite"].invariants_dict["temporal_flux_ratio"] == 0.0
    assert sigs["Real_4K_Yosemite"].invariants_dict["spacetime_phase_continuity"] == 1.0
    sim_4k_flower = float(iso[idx["Real_4K_Yosemite"], idx["Real_MP4_Flower"]])
    sim_4k_crystal = float(iso[idx["Real_4K_Yosemite"], idx["PDE_Crystal"]])
    sim_4k_noise = float(iso[idx["Real_4K_Yosemite"], idx["PDE_WhiteNoise"]])
    assert sim_4k_flower > 0.75
    assert sim_4k_crystal > sim_4k_noise

    # Real blooming flower MP4 & real speech WAV both resonate stronger with ordered regimes than noise
    assert sigs["Real_WAV_Speech"].invariants_dict["spacetime_phase_continuity"] > 0.35

    sim_mp4_noise = float(iso[idx["Real_MP4_Flower"], idx["PDE_WhiteNoise"]])
    sim_mp4_ordered_max = max(
        float(iso[idx["Real_MP4_Flower"], idx[other]])
        for other in ("Real_4K_Yosemite", "PDE_Diffusion", "PDE_Wave", "PDE_Crystal")
    )
    assert sim_mp4_ordered_max > sim_mp4_noise

    sim_wav_noise = float(iso[idx["Real_WAV_Speech"], idx["PDE_WhiteNoise"]])
    sim_wav_ordered_max = max(
        float(iso[idx["Real_WAV_Speech"], idx[other]])
        for other in ("Real_MP4_Flower", "PDE_Diffusion", "PDE_Wave", "PDE_Crystal")
    )
    assert sim_wav_ordered_max > sim_wav_noise

    # CausalField connectivity graph has both voxels and resonant beams
    assert len(core.causal_field.voxels) == len(arrays)
    assert len(core.causal_field.beams) >= 2

    # 4. Closed-loop round-trip wavelength boundary modulation on a real MP4 frame:
    # Real frame starts strongly macro-dominated (+0.93), so shift short-wavelength first,
    # then restore back toward long-wavelength macro balance on the modulated frame.
    sample_rgb = mp4_arr[-1]
    mod_short = core.modulate_wavelength_boundary_and_reobserve(
        sample_rgb, "Real_MP4_FrameEnd", mode="shift_short_wavelength"
    )
    assert mod_short.modulated_frame is not None
    mod_long = core.modulate_wavelength_boundary_and_reobserve(
        mod_short.modulated_frame, "Real_MP4_Restored", mode="shift_long_wavelength"
    )
    print(
        f"\n[Modulation Short] before={mod_short.before_balance:+.3f} -> after={mod_short.after_balance:+.3f} (delta={mod_short.delta_balance:+.3f})"
    )
    print(
        f"[Modulation Long ] before={mod_long.before_balance:+.3f} -> after={mod_long.after_balance:+.3f} (delta={mod_long.delta_balance:+.3f})"
    )
    assert mod_short.control_verified is True
    assert mod_long.control_verified is True

    # 5. Render dashboard
    out_png = core.render_spacetime_boundary_dashboard(
        report, [mod_short, mod_long], DASHBOARD_OUT
    )
    assert out_png.exists() and out_png.stat().st_size > 25000
