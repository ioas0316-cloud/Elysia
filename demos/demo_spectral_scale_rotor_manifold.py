"""
Spectral Scale-Rotor Manifold Empirical Benchmark & Real-World Demo
===================================================================

Demonstrates escaping 1-to-100 sequential point-array computation using
hierarchical Spectral Scale-Rotor Manifolds, continuous Light/Dark & Scale
balance axes ("명암의 저울"), adaptive knot splitting, CausalField sync,
and real-world audio/image grounding.
"""

from __future__ import annotations

import math
import sys
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.physics.causal_field import CausalField
from core.physics.spectral_scale_rotor_manifold import (
    ScaleRotorBand,
    SpectralScaleRotorManifold,
    encode_surface_to_scale_rotors,
    encode_waveform_to_scale_rotors,
)


def section_1_escape_1_to_100_benchmark() -> None:
    print("=" * 92)
    print(" [SECTION 1] 1->100 ESCAPE BENCHMARK: POINT-ARRAY STEPPING vs. SCALE-ROTOR O(K)")
    print("=" * 92)
    print(f"{'N Points':>10} | {'Encode(ms)':>10} | {'Point 1..100(ms)':>16} | {'Rotor O(K)(ms)':>14} | {'Speedup':>9} | {'Mem Ratio':>10} | {'Integral Err':>12}")
    print("-" * 92)

    w1 = 2.0 * math.pi * 3.0
    w2 = 2.0 * math.pi * 7.0
    c_speed = 1.0
    dt_step = 0.01
    total_dt = 1.0  # 100 steps of 0.01 vs direct dt=1.0 (equally dt=100.0 for 100 steps of 1.0)

    for n_pts in (100, 10_000, 1_000_000):
        x = np.linspace(0.0, 1.0, n_pts, endpoint=False)
        # Analytic wave at t=0: 1.4*cos(w1*x) + 0.8*sin(w2*x) + localized pulse
        raw_wave = (
            1.4 * np.cos(w1 * x)
            + 0.8 * np.sin(w2 * x)
            + 1.8 * np.exp(-0.5 * ((x - 0.58) / 0.03) ** 2)
        )

        # Measure one-time boundary ingestion separately (honest accounting)
        t_enc0 = time.perf_counter()
        manifold = encode_waveform_to_scale_rotors(
            raw_wave,
            domain=(0.0, 1.0),
            max_macro_modes=8,
            max_levels=3,
            max_knots_per_level=6,
            residual_tol=0.02,
            dispersion_speed=c_speed,
        )
        enc_ms = (time.perf_counter() - t_enc0) * 1000.0
        partner = manifold.evolve_spacetime(dt=0.37)

        # Dense Point-Array baseline: 100 sequential time-stepping updates + quadrature + L2 overlap
        arr = raw_wave.copy()
        partner_arr = partner.slice_manifestation(x)
        t_pt0 = time.perf_counter()
        # Simulate 100 sequential local stencil / phase ticks over N points
        for _ in range(100):
            arr = arr * 0.999 + 0.001 * np.roll(arr, 1)
        pt_integral = float(np.mean(arr))
        _ = float(np.mean(arr * partner_arr))
        pt_ms = (time.perf_counter() - t_pt0) * 1000.0

        # Scale-Rotor O(K) exact direct jump (no 1..100 loop, no point grid!)
        # Average over 20 calls for stable sub-millisecond timing
        _ = manifold.evolve_spacetime(dt=100.0 * total_dt).integrate_over_domain()
        t_rt0 = time.perf_counter()
        for _ in range(20):
            ev = manifold.evolve_spacetime(dt=100.0 * total_dt)
            rotor_integral = ev.integrate_over_domain()
            _ = ev.wave_resonance(partner)
        rt_ms = ((time.perf_counter() - t_rt0) / 20.0) * 1000.0

        # Check exact integral error against dense numerical quadrature on evolved continuous manifold
        exact_ev_slice = manifold.evolve_spacetime(dt=100.0 * total_dt).slice_manifestation(
            np.linspace(0.0, 1.0, min(n_pts, 50_000), endpoint=False)
        )
        quad_integral = float(np.mean(exact_ev_slice))
        int_err = abs(rotor_integral - quad_integral)

        speedup = pt_ms / max(rt_ms, 1e-6)
        mem_ratio = raw_wave.nbytes / max(manifold.nbytes, 1)
        print(
            f"{n_pts:10,d} | {enc_ms:10.3f} | {pt_ms:16.3f} | {rt_ms:14.4f} | "
            f"{speedup:8.1f}x | {mem_ratio:9.1f}x | {int_err:12.2e}"
        )


def section_2_adaptive_scale_roles() -> None:
    print("\n" + "=" * 92)
    print(" [SECTION 2] SCALE-RELATIVE ROLES: SMOOTH CARRIER vs. ADAPTIVE EVENT KNOTS")
    print("=" * 92)

    x = np.linspace(0.0, 1.0, 1024, endpoint=False)
    smooth_1d = 1.2 * np.cos(2.0 * math.pi * 4.0 * x) + 0.7 * np.sin(2.0 * math.pi * 9.0 * x)
    shock_1d = smooth_1d + 2.8 * np.exp(-0.5 * ((x - 0.63) / 0.02) ** 2)

    m_smooth = encode_waveform_to_scale_rotors(smooth_1d, residual_tol=0.02)
    m_shock = encode_waveform_to_scale_rotors(
        shock_1d, max_macro_modes=8, max_levels=3, max_knots_per_level=6, residual_tol=0.03
    )

    print("1D Smooth Harmonic Signal:")
    for b in m_smooth.bands:
        print(f"  - Level {b.level} ({b.role:<16}): {b.num_modes} modes, global={b.is_global}")

    print("1D Signal + Localized Shock at x=0.63:")
    for b in m_shock.bands:
        extra = "" if b.is_global else f", centers={np.round(b.centers[:, 0], 3).tolist()}"
        print(f"  - Level {b.level} ({b.role:<16}): {b.num_modes} modes, global={b.is_global}{extra}")

    # 2D Surface + Diagonal Ridge + Sharp Knot
    ny, nx = 64, 64
    ys = np.linspace(0.0, 1.0, ny, endpoint=False)
    xs = np.linspace(0.0, 1.0, nx, endpoint=False)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    smooth_2d = np.cos(2.0 * math.pi * 2.0 * yy) + 0.6 * np.sin(2.0 * math.pi * 3.0 * xx)
    ridge_and_knot_2d = (
        smooth_2d
        + 1.6 * np.exp(-0.5 * ((yy - xx) / 0.04) ** 2)
        + 2.5 * np.exp(-(((yy - 0.25) ** 2 + (xx - 0.78) ** 2) / (2.0 * 0.03 ** 2)))
    )
    m2_smooth = encode_surface_to_scale_rotors(smooth_2d, residual_tol=0.02)
    m2_complex = encode_surface_to_scale_rotors(
        ridge_and_knot_2d, max_macro_modes=12, max_levels=3, max_knots_per_level=8, residual_tol=0.05
    )

    print("2D Smooth Surface:")
    for b in m2_smooth.bands:
        print(f"  - Level {b.level} ({b.role:<16}): {b.num_modes} modes, global={b.is_global}")
    print("2D Surface + Diagonal Ridge + Point Defect:")
    for b in m2_complex.bands:
        print(f"  - Level {b.level} ({b.role:<16}): {b.num_modes} modes, global={b.is_global}")


def section_3_balance_scale_and_coupling() -> None:
    print("\n" + "=" * 92)
    print(" [SECTION 3] CONTINUOUS BALANCE SCALE ('명암의 저울') & BIDIRECTIONAL COUPLING")
    print("=" * 92)

    b0 = ScaleRotorBand(
        level=0,
        dim=1,
        wavevectors=np.array([[2.0 * math.pi * 2.0], [2.0 * math.pi * 5.0]]),
        temporal_omegas=np.array([2.0, 5.0]),
        amplitudes=np.array([1.0 + 0.0j, -0.8 + 0.0j]),
        centers=np.zeros((2, 1)),
        sigmas=np.full(2, np.inf),
        is_global=True,
        parent_indices=np.array([-1, -1], dtype=np.int32),
        parent_couplings=np.array([1.0, 1.0]),
    )
    b1 = ScaleRotorBand(
        level=1,
        dim=1,
        wavevectors=np.array([[2.0 * math.pi * 11.0], [2.0 * math.pi * 17.0]]),
        temporal_omegas=np.array([11.0, 17.0]),
        amplitudes=np.array([0.8 + 0.0j, -0.8 + 0.0j]),
        centers=np.array([[0.35], [0.72]]),
        sigmas=np.array([0.05, 0.05]),
        is_global=False,
        parent_indices=np.array([0, 1], dtype=np.int32),
        parent_couplings=np.array([0.8, 0.8]),
    )
    manifold = SpectralScaleRotorManifold(dim=1, domain_bounds=([0.0], [1.0]), bands=[b0, b1])

    print("Scale Tilt Sweep (contrast_tilt=0.0):")
    print(f"{'scale_tilt':>10} | {'Macro E':>10} | {'Micro E':>10} | {'Micro/Macro':>11} | {'Chromatic [R, B, Y]'}")
    print("-" * 92)
    for st in (-1.0, -0.5, 0.0, 0.5, 1.0):
        prof = manifold.tune_balance(scale_tilt=st).spectral_axis_profile()
        triad = [round(c, 3) for c in prof["chromatic_triad"]]
        print(
            f"{st:+10.1f} | {prof['macro_energy']:10.4f} | {prof['micro_energy']:10.4f} | "
            f"{prof['micro_to_macro_ratio']:11.4f} | {triad}"
        )

    print("\nContrast Tilt Sweep (Light vs. Dark Balance, scale_tilt=0.0):")
    print(f"{'contrast':>10} | {'Bright E':>10} | {'Dark E':>10} | {'Bright/Dark':>11} | {'L-D Balance':>11}")
    print("-" * 92)
    for ct in (-1.0, -0.5, 0.0, 0.5, 1.0):
        prof = manifold.tune_balance(contrast_tilt=ct).spectral_axis_profile()
        print(
            f"{ct:+10.1f} | {prof['bright_energy']:10.4f} | {prof['dark_energy']:10.4f} | "
            f"{prof['bright_to_dark_ratio']:11.4f} | {prof['light_dark_balance']:+11.4f}"
        )

    # Bidirectional Emit <-> Observe coupling
    observer = SpectralScaleRotorManifold(
        dim=1,
        domain_bounds=([0.0], [1.0]),
        bands=[
            ScaleRotorBand(
                level=0,
                dim=1,
                wavevectors=b0.wavevectors,
                temporal_omegas=b0.temporal_omegas,
                amplitudes=b0.amplitudes * np.exp(1j * 0.5),
                centers=b0.centers,
                sigmas=b0.sigmas,
                is_global=True,
                parent_indices=b0.parent_indices,
                parent_couplings=b0.parent_couplings,
            ),
            ScaleRotorBand(
                level=1,
                dim=1,
                wavevectors=b1.wavevectors,
                temporal_omegas=b1.temporal_omegas,
                amplitudes=b1.amplitudes * np.exp(1j * 0.5),
                centers=b1.centers,
                sigmas=b1.sigmas,
                is_global=False,
                parent_indices=b1.parent_indices,
                parent_couplings=b1.parent_couplings,
            ),
        ],
    )
    res_init = manifold.wave_resonance(observer)["normalized_resonance"]
    e_c, o_c, c_met = manifold.bidirectional_couple(observer, coupling_gain=0.35, inhibition_gain=0.25)
    print(
        f"\nBidirectional Emit<->Observe Coupling: resonance {res_init:+.4f} -> {c_met['resonance_after']:+.4f} "
        f"(damping={c_met['observer_damping_sum']:.4f})"
    )


def section_4_real_world_grounding_and_causal_field() -> None:
    print("\n" + "=" * 92)
    print(" [SECTION 4] REAL-WORLD ASSETS (WAV AUDIO + HD IMAGE) & CAUSAL FIELD SYNC")
    print("=" * 92)

    wav_path = Path(r"c:\Elysia\data\unified_creation_dawn.wav")
    img_path = Path(r"c:\Elysia\demos\real_world_assets\real_human_face_hd.jpg")

    # Audio encoding
    with wave.open(str(wav_path), "rb") as wf:
        raw_bytes = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64)
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels())[:, 0]
        sr = float(wf.getframerate())
    audio_norm = audio / max(float(np.max(np.abs(audio))), 1.0)
    duration = len(audio_norm) / sr

    t0 = time.perf_counter()
    m_audio = encode_waveform_to_scale_rotors(
        audio_norm, domain=(0.0, duration), max_macro_modes=32, max_levels=3, max_knots_per_level=12, residual_tol=0.10
    )
    enc_audio_ms = (time.perf_counter() - t0) * 1000.0
    recon_audio = m_audio.slice_manifestation(np.linspace(0.0, duration, len(audio_norm), endpoint=False)[:, None])
    corr_audio = float(np.corrcoef(audio_norm, recon_audio)[0, 1])

    t_op0 = time.perf_counter()
    _ = m_audio.evolve_spacetime(dt=100.0).tune_balance(scale_tilt=0.4).integrate_over_domain()
    op_audio_ms = (time.perf_counter() - t_op0) * 1000.0

    print(
        f"Audio ({wav_path.name}): N={len(audio_norm):,d} samples ({audio_norm.nbytes:,d} B) -> "
        f"K={m_audio.total_modes} modes ({m_audio.nbytes:,d} B, {audio_norm.nbytes / m_audio.nbytes:.1f}x comp)"
    )
    print(
        f"  Encode={enc_audio_ms:.2f} ms | In-Manifold O(K) Jump+Tune+Integrate={op_audio_ms:.4f} ms | "
        f"Corr={corr_audio:.4f}"
    )

    # Image encoding (128x128 grayscale crop/resize for interactive speed)
    img = Image.open(img_path).convert("L").resize((128, 128), Image.Resampling.BILINEAR)
    img_arr = np.asarray(img, dtype=np.float64) / 255.0
    t0 = time.perf_counter()
    m_img = encode_surface_to_scale_rotors(
        img_arr, max_macro_modes=48, max_levels=3, max_knots_per_level=16, residual_tol=0.08
    )
    enc_img_ms = (time.perf_counter() - t0) * 1000.0

    ys = np.linspace(0.0, 1.0, 128, endpoint=False)
    xs = np.linspace(0.0, 1.0, 128, endpoint=False)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pts2 = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
    recon_img = m_img.slice_manifestation(pts2)
    corr_img = float(np.corrcoef(img_arr.reshape(-1), recon_img)[0, 1])

    t_op0 = time.perf_counter()
    img_tuned = m_img.tune_balance(scale_tilt=0.5, contrast_tilt=0.4)
    _ = img_tuned.integrate_over_domain()
    op_img_ms = (time.perf_counter() - t_op0) * 1000.0

    print(
        f"Image ({img_path.name} @128x128): N={img_arr.size:,d} px ({img_arr.nbytes:,d} B) -> "
        f"K={m_img.total_modes} modes ({m_img.nbytes:,d} B, {img_arr.nbytes / m_img.nbytes:.1f}x comp)"
    )
    print(
        f"  Encode={enc_img_ms:.2f} ms | In-Manifold O(K) Tune+Integrate={op_img_ms:.4f} ms | "
        f"Corr={corr_img:.4f}"
    )

    # CausalField sync
    cf = CausalField(dimensions=3)
    sync_res = img_tuned.sync_to_causal_field(cf, prefix="face_rotor")
    cf.step(dt=0.05)
    print(
        f"CausalField Sync: voxels={len(sync_res['voxel_ids'])}, beams={sync_res['beams_linked']}, "
        f"chromatic_triad={[round(c, 3) for c in sync_res['chromatic_triad']]}"
    )
    print("=" * 92)


if __name__ == "__main__":
    section_1_escape_1_to_100_benchmark()
    section_2_adaptive_scale_roles()
    section_3_balance_scale_and_coupling()
    section_4_real_world_grounding_and_causal_field()
