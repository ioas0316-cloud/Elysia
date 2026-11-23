"""
Causal Mechanism Palette & Rectification Empirical Demo
=======================================================

Demonstrates treating scientific laws, existing Elysia code modules, and
distilled experiential memories as directly composable "Palette Pigments" (물감)
over SpectralScaleRotorManifold, avoiding from-scratch formula re-derivation
and 1-to-100+ point-grid stepping.
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
from core.physics.causal_mechanism_palette import CausalMechanismPalette
from core.physics.spectral_scale_rotor_manifold import (
    ScaleRotorBand,
    SpectralBalanceScale,
    SpectralScaleRotorManifold,
    encode_surface_to_scale_rotors,
    encode_waveform_to_scale_rotors,
)


def _make_1d(
    global_w=(0.0, 2.0 * math.pi * 3.0),
    global_amp=(0.4 + 0.0j, 1.0 + 0.3j),
    local_w=(2.0 * math.pi * 4.0,),
    local_amp=(1.2 + 0.2j,),
    local_centers=(0.5,),
    local_sigmas=(0.045,),
) -> SpectralScaleRotorManifold:
    b0 = ScaleRotorBand(
        level=0,
        dim=1,
        wavevectors=np.array(global_w, dtype=np.float64)[:, None],
        temporal_omegas=np.abs(np.array(global_w, dtype=np.float64)),
        amplitudes=np.array(global_amp, dtype=np.complex128),
        centers=np.zeros((len(global_w), 1)),
        sigmas=np.full(len(global_w), np.inf),
        is_global=True,
        parent_indices=-np.ones(len(global_w), dtype=np.int32),
        parent_couplings=np.ones(len(global_w)),
    )
    bands = [b0]
    if local_w:
        b1 = ScaleRotorBand(
            level=1,
            dim=1,
            wavevectors=np.array(local_w, dtype=np.float64)[:, None],
            temporal_omegas=np.abs(np.array(local_w, dtype=np.float64)),
            amplitudes=np.array(local_amp, dtype=np.complex128),
            centers=np.array(local_centers, dtype=np.float64)[:, None],
            sigmas=np.array(local_sigmas, dtype=np.float64),
            is_global=False,
            parent_indices=np.zeros(len(local_w), dtype=np.int32),
            parent_couplings=np.ones(len(local_w)),
        )
        bands.append(b1)
    return SpectralScaleRotorManifold(dim=1, domain_bounds=([0.0], [1.0]), bands=bands)


def section_1_heat_pde_palette_escape() -> None:
    print("=" * 94)
    print(" [SECTION 1] HEAT PDE: 1->N FINITE-DIFFERENCE STEPPING vs. PALETTE PIGMENT O(K)")
    print("=" * 94)
    print(f"{'N Grid':>8} | {'FTCS Steps':>10} | {'FTCS (ms)':>11} | {'Palette O(K)(ms)':>16} | {'Speedup':>9} | {'Rel L2 Err':>11} | {'Mass Err':>10}")
    print("-" * 94)

    palette = CausalMechanismPalette.with_default_principles()
    manifold = _make_1d()
    diffusivity = 0.0005
    tau = 1.0

    # Warmup palette execution
    _ = palette.compose_and_apply(
        manifold, [("physics.thermal_diffusion", 1.0, {"diffusivity": diffusivity, "tau": tau})]
    )

    for n in (256, 512, 2048):
        x = np.linspace(0.0, 1.0, n, endpoint=False)
        dx = 1.0 / n
        dt_cfl = 0.25 * (dx ** 2) / diffusivity
        steps = int(math.ceil(tau / dt_cfl))
        dt = tau / steps
        alpha = diffusivity * dt / (dx ** 2)

        u = manifold.slice_manifestation(x)
        mass_init = float(np.mean(u))
        t0 = time.perf_counter()
        for _ in range(steps):
            u = u + alpha * (np.roll(u, -1) - 2.0 * u + np.roll(u, 1))
        ftcs_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        for _ in range(50):
            evolved = palette.compose_and_apply(
                manifold,
                [("physics.thermal_diffusion", 1.0, {"diffusivity": diffusivity, "tau": tau})],
            )
        pal_ms = ((time.perf_counter() - t1) / 50.0) * 1000.0

        u_pal = evolved.slice_manifestation(x)
        rel_err = float(np.linalg.norm(u - u_pal) / np.linalg.norm(u))
        mass_err = abs(evolved.integrate_over_domain() - mass_init)
        speedup = ftcs_ms / max(pal_ms, 1e-6)

        print(
            f"{n:8d} | {steps:10,d} | {ftcs_ms:11.3f} | {pal_ms:16.4f} | "
            f"{speedup:8.1f}x | {rel_err:11.2e} | {mass_err:10.2e}"
        )


def section_2_palette_composition() -> None:
    print("\n" + "=" * 94)
    print(" [SECTION 2] PALETTE COMPOSITION ('팔레트 조색'): PHYSICS + BIOLOGY + OPTICS + CODE")
    print("=" * 94)

    palette = CausalMechanismPalette.with_default_principles()
    m1 = _make_1d(
        global_w=(0.0, 2.0 * math.pi * 3.0, 2.0 * math.pi * 4.5, 2.0 * math.pi * 6.0),
        global_amp=(1.2 + 0.0j, 1.0 + 0.1j, 0.9 - 0.2j, 0.7 + 0.3j),
    )

    m_par = palette.compose_and_apply(
        m1,
        [
            ("physics.thermal_diffusion", 0.55, {"diffusivity": 0.001, "tau": 1.0}),
            ("biology.lateral_inhibition", 0.30, {}),
            ("acoustics.harmonic_resonance", 0.15, {"fundamental": 2.0 * math.pi * 3.0}),
        ],
        mode="parallel",
        balance=SpectralBalanceScale(scale_tilt=0.25, contrast_tilt=0.35),
    )
    prof_par = m_par.spectral_axis_profile()
    print(
        f"Parallel Mix (55% Thermal + 30% Retinal DoG + 15% Acoustic Harmonics): "
        f"K={m_par.total_modes}, L-D Balance={prof_par['light_dark_balance']:+.4f}, "
        f"Chromatic={[round(c, 3) for c in prof_par['chromatic_triad']]}"
    )

    m_seq = palette.compose_and_apply(
        m1,
        [
            ("code.clifford_rotor_twist", 1.0, {"angle": 0.75, "scale": 1.2}),
            ("code.causal_worldline_orbit", 1.0, {"time": 0.8}),
            ("optics.snell_refraction", 1.0, {"index_ratio": 1.15, "angle": 0.0}),
        ],
        mode="sequential",
    )
    prof_seq = m_seq.spectral_axis_profile()
    print(
        f"Sequential Pipeline (Clifford Twist -> Worldline Orbit -> Snell Refraction): "
        f"K={m_seq.total_modes}, MacroE={prof_seq['macro_energy']:.4f}, MicroE={prof_seq['micro_energy']:.4f}"
    )

    # Exact Clifford rewind verification
    tw = palette.get("code.clifford_rotor_twist").apply(m1, {"angle": 0.9, "scale": 1.4})
    rw = palette.get("code.clifford_rotor_twist").apply(tw, {"angle": 0.9, "scale": 1.4, "rewind": True})
    probe = np.linspace(0.0, 1.0, 256)[:, None]
    rw_err = float(np.max(np.abs(m1.slice_manifestation(probe) - rw.slice_manifestation(probe))))
    print(f"CliffordSpacetimeRotor Forward+Rewind Max Reconstruction Error: {rw_err:.2e}")


def _unknown_external_law(manifold: SpectralScaleRotorManifold) -> SpectralScaleRotorManifold:
    bands = []
    for band in manifold.bands:
        bc = band.copy()
        if bc.num_modes > 0:
            w_mag = np.linalg.norm(bc.wavevectors, axis=1)
            gain = 1.0 + 1.8 * np.exp(-0.5 * ((w_mag - 25.0) / 7.0) ** 2) * np.exp(1j * 0.7)
            bc.amplitudes = bc.amplitudes * gain
        bands.append(bc)
    return SpectralScaleRotorManifold(
        dim=manifold.dim, domain_bounds=manifold.domain_bounds, bands=bands, balance=manifold.balance
    )


def section_3_rectification_and_zero_shot_transfer() -> None:
    print("\n" + "=" * 94)
    print(" [SECTION 3] COGNITIVE RECTIFICATION, MEMORY DISTILLATION & ZERO-SHOT TRANSFER")
    print("=" * 94)

    palette = CausalMechanismPalette.with_default_principles()

    # A. Field Rectification: 1st unseen exposure vs 2nd exposure recall
    novel_field = _make_1d(
        global_w=(2.0 * math.pi * 11.0,),
        global_amp=(1.2 + 0.6j,),
        local_w=(2.0 * math.pi * 19.0,),
        local_amp=(0.9 - 0.3j,),
        local_centers=(0.27,),
        local_sigmas=(0.035,),
    )
    rf1 = palette.rectify_field(novel_field, novelty_threshold=0.15)
    print(
        f"Field Exposure #1: from_memory={rf1.from_memory}, friction={rf1.residual_friction:.4f}, "
        f"distilled={rf1.distilled_pigment_id}, time={rf1.elapsed_ms:.3f} ms"
    )

    composite = novel_field.scale_field(1.4).superpose(
        palette.get("archetype.dc_equilibrium_1d").archetype_signature.scale_field(0.6)  # type: ignore[union-attr]
    )
    rf2 = palette.rectify_field(composite, novelty_threshold=0.15)
    top_w = {k: round(v, 3) for k, v in rf2.recipe_weights.items() if abs(v) > 0.05}
    print(
        f"Field Exposure #2 (Composite): from_memory={rf2.from_memory}, explained={rf2.explained_ratio:.4f}, "
        f"friction={rf2.residual_friction:.2e}, recipe={top_w}"
    )

    # B. Transition Rectification: discover known mix (0.7 Thermal + 0.3 Lateral Inhibition)
    before_k = _make_1d(
        global_w=(0.0, 2.0 * math.pi * 3.0, 2.0 * math.pi * 7.0),
        global_amp=(1.0 + 0.0j, 0.9 + 0.2j, -0.6 + 0.4j),
    )
    after_k = palette.compose_and_apply(
        before_k,
        [("physics.thermal_diffusion", 0.7, None), ("biology.lateral_inhibition", 0.3, None)],
        mode="parallel",
    )
    rt_known = palette.rectify_transition(before_k, after_k, novelty_threshold=0.10)
    top_tk = {k: round(v, 3) for k, v in rt_known.recipe_weights.items() if abs(v) > 0.05}
    print(
        f"Transition Discovery (Known Law): from_memory={rt_known.from_memory}, "
        f"explained={rt_known.explained_ratio:.4f}, discovered={top_tk}"
    )

    # C. Unknown Law Distillation + Zero-Shot Transfer to Brand-New Unseen Input
    b1 = _make_1d(
        global_w=(2.0 * math.pi * 1.0, 2.0 * math.pi * 3.0, 2.0 * math.pi * 4.0, 2.0 * math.pi * 6.0),
        global_amp=(1.0 + 0.1j, 0.9 - 0.2j, 1.1 + 0.3j, 0.8 + 0.0j),
        local_w=(),
        local_amp=(),
        local_centers=(),
        local_sigmas=(),
    )
    a1 = _unknown_external_law(b1)
    rt_nov1 = palette.rectify_transition(b1, a1, novelty_threshold=0.10)
    rt_nov2 = palette.rectify_transition(b1, a1, novelty_threshold=0.10)

    b2 = _make_1d(
        global_w=(2.0 * math.pi * 2.0, 2.0 * math.pi * 4.0, 2.0 * math.pi * 5.0),
        global_amp=(0.7 * np.exp(1j * 1.1), 1.3 * np.exp(-1j * 0.8), 0.9 * np.exp(1j * 2.1)),
        local_w=(),
        local_amp=(),
        local_centers=(),
        local_sigmas=(),
    )
    true_a2 = _unknown_external_law(b2)
    pred_a2 = palette.compose_and_apply(b2, [(rt_nov1.distilled_pigment_id, 1.0, {})])
    transfer_res = pred_a2.wave_resonance(true_a2)["normalized_resonance"]

    print(
        f"Unknown Causal Law #1 -> Distilled '{rt_nov1.distilled_pigment_id}' "
        f"(friction={rt_nov1.residual_friction:.4f}); #2 Recall explained={rt_nov2.explained_ratio:.4f}"
    )
    print(f"Zero-Shot Transfer to Unseen Manifold: Normalized L^2 Resonance = {transfer_res:+.4f}")


def section_4_real_world_and_causal_field() -> None:
    print("\n" + "=" * 94)
    print(" [SECTION 4] REAL-WORLD AUDIO/IMAGE RECTIFICATION & CAUSAL FIELD ENGRAM SYNC")
    print("=" * 94)

    palette = CausalMechanismPalette.with_default_principles()
    wav_path = Path(r"c:\Elysia\data\unified_creation_dawn.wav")
    img_path = Path(r"c:\Elysia\demos\real_world_assets\real_human_face_hd.jpg")

    with wave.open(str(wav_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels())[:, 0]
        sr = float(wf.getframerate())
    audio_norm = audio / max(float(np.max(np.abs(audio))), 1.0)
    m_audio = encode_waveform_to_scale_rotors(
        audio_norm, domain=(0.0, len(audio_norm) / sr), max_macro_modes=32, max_levels=2, max_knots_per_level=10
    )

    ra1 = palette.rectify_field(m_audio, novelty_threshold=0.15, memory_label="memory.dawn_audio")
    ra2 = palette.rectify_field(m_audio, novelty_threshold=0.15)
    print(
        f"Audio ({wav_path.name}, K={m_audio.total_modes}): "
        f"1st distilled={ra1.distilled_pigment_id} ({ra1.elapsed_ms:.2f} ms) | "
        f"2nd recall from_memory={ra2.from_memory}, explained={ra2.explained_ratio:.4f} ({ra2.elapsed_ms:.2f} ms)"
    )

    img = Image.open(img_path).convert("L").resize((128, 128), Image.Resampling.BILINEAR)
    img_arr = np.asarray(img, dtype=np.float64) / 255.0
    m_img = encode_surface_to_scale_rotors(
        img_arr, max_macro_modes=48, max_levels=2, max_knots_per_level=12, residual_tol=0.08
    )

    ri1 = palette.rectify_field(m_img, novelty_threshold=0.15, memory_label="memory.face_surface")
    ri2 = palette.rectify_field(m_img, novelty_threshold=0.15)

    t0 = time.perf_counter()
    _ = palette.compose_and_apply(
        m_img,
        [
            ("physics.thermal_diffusion", 1.0, {"diffusivity": 0.0005, "tau": 1.0}),
            ("biology.lateral_inhibition", 1.0, {}),
            ("optics.snell_refraction", 1.0, {"index_ratio": 1.1, "angle": 0.15}),
        ],
        mode="sequential",
    )
    style_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"Image ({img_path.name} @128x128, K={m_img.total_modes}): "
        f"1st distilled={ri1.distilled_pigment_id} ({ri1.elapsed_ms:.2f} ms) | "
        f"2nd recall explained={ri2.explained_ratio:.4f} | 3-Law O(K) Recipe={style_ms:.4f} ms"
    )

    cf = CausalField(dimensions=3)
    sync = palette.sync_to_causal_field(cf, last_result=ri2, prefix="pal")
    cf.step(dt=0.05)
    print(
        f"CausalField Sync: engrams={len(sync['engram_ids'])}, "
        f"voxels={len(cf.voxels)}, beams_linked={sync['beams_linked']}"
    )
    print("=" * 94)


if __name__ == "__main__":
    section_1_heat_pde_palette_escape()
    section_2_palette_composition()
    section_3_rectification_and_zero_shot_transfer()
    section_4_real_world_and_causal_field()
