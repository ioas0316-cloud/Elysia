r"""
ELYSIA Phase 9 Demo: Unit-1 Spectral Mirror & Living Causal-Operator Superposition
==================================================================================

Demonstrates the living mirror principle ("거울원리 / 단위체계 1 정규화 / 결과를
결과화하지 않기") across real visual images, real acoustic waves, unknown coherent
order discovery, broadband rupture rejection, and native code/chemistry spectra:
1. Real HD Face Eye Illusion (`OBSERVER_STRUCTURE_ARTIFACT`) + De-lensing
2. Real HD Face World Medium (Planck 3200K + Cauchy Dispersion + Local Occlusion)
   + Zero-Shot Causal Transfer onto an unseen checkerboard
3. Unknown Coherent Order Discovery + Live Mirror Self-Expansion (`absorb_discovered_order`)
   vs Broadband Non-Causal Rupture Rejection
4. Real WAV Audio Acoustic Medium Superposition + Transfer
5. Native Code & Chemistry Unit-1 Mirror + Phase 8 Trinitarian & `CausalField` Bridge.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import wave

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.physics.causal_field import CausalField
from core.physics.unit_spectral_mirror_protocol import (
    AcousticWaveOperator,
    IdentityCausalOperator,
    OpticalOcularEyeOperator,
    OpticalWorldMediumOperator,
    PlanetaryOrderGlobe,
    StructuralSpectrumOperator,
)


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA UNIT-1 SPECTRAL MIRROR & LIVING CAUSAL SUPERPOSITION (PHASE 9) DEMO")
    print("=" * 92)

    img_path = Path(r"c:\Elysia\demos\real_world_assets\real_human_face_hd.jpg")
    wav_path = Path(r"c:\Elysia\data\unified_creation_dawn.wav")
    face = np.asarray(
        Image.open(img_path).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0

    # 1. Observer Eye Illusion vs World Stability
    _section("1. REAL HD FACE: OCULAR EYE ILLUSION vs STABLE EXTERNAL WORLD")
    face_globe = PlanetaryOrderGlobe("human_face_64", "optics_2d", face)
    skewed_eye = OpticalOcularEyeOperator(
        name="aberrated_retina",
        cone_gain=(1.25, 0.55, 0.85),
        corneal_defocus_sigma=1.2,
        ocular_shift=(1.5, -2.0),
    )
    obs_eye = skewed_eye.apply(face)
    sup_eye = face_globe.superpose(obs_eye, observer_operator=skewed_eye)
    clear_sup = sup_eye.counterfactual_reobserve(IdentityCausalOperator(modality="optics_2d"))
    print(sup_eye.explanation)
    print(
        f"  -> De-lensed world deviation={sup_eye.world_unit_deviation:.5f} | "
        f"Recon Err={sup_eye.reconstruction_error():.5f} | "
        f"Counterfactual clear-eye origin={clear_sup.origin_kind.value}"
    )

    # 2. World Lawful Transformation (Planck 3200K + Cauchy Dispersion + Local Occlusion)
    _section("2. REAL HD FACE: PLANCK 3200K + CAUCHY DISPERSION + LOCAL OCCLUSION & TRANSFER")
    world_med = OpticalWorldMediumOperator(
        name="sunset_prism_shadow",
        illumination_kelvin=3200.0,
        chromatic_dispersion_shifts=((0.0, 0.2), (0.0, 0.8), (0.0, 1.8)),
        occlusion_box=(40, 58, 36, 56),
        occlusion_factor=0.25,
    )
    sup_world = face_globe.superpose(world_med.apply(face), world_operator_hint=world_med)
    yy, xx = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64), indexing="ij")
    unseen = np.stack([yy, xx, 0.5 * np.ones_like(yy)], axis=2)
    transferred = sup_world.transfer_world_cause_to(unseen)
    print(sup_world.explanation)
    print(
        f"  -> Channel Deviations (R/G/B)={sup_world.onset_profile.channel_deviations['R']:.4f} / "
        f"{sup_world.onset_profile.channel_deviations['G']:.4f} / "
        f"{sup_world.onset_profile.channel_deviations['B']:.4f}"
    )
    print(
        f"  -> Recon Err={sup_world.reconstruction_error():.2e} | "
        f"Zero-Shot Transfer delta RMS={float(np.sqrt(np.mean((transferred - unseen) ** 2))):.4f}"
    )

    # 3. Unknown Coherent Order Discovery & Expansion vs Broadband Rupture
    _section("3. UNKNOWN COHERENT ORDER MIRROR EXPANSION vs BROADBAND RUPTURE")
    t = np.linspace(0.0, 1.0, 256, endpoint=False)
    ref_wave = 1.2 * np.cos(2.0 * math.pi * 3.0 * t) + 0.7 * np.sin(2.0 * math.pi * 7.0 * t)
    chord_globe = PlanetaryOrderGlobe("harmonic_chord", "acoustics_1d", ref_wave)

    op_19 = AcousticWaveOperator(name="overtone_19hz", injected_mode=(19.0, 0.6, 0.25))
    sup_unk = chord_globe.superpose(op_19.apply(ref_wave), world_operator_hint=op_19)
    print(sup_unk.explanation)
    sup_expanded = sup_unk.absorb_discovered_order()
    print(f"  -> After absorb_discovered_order(): origin={sup_expanded.origin_kind.value}, "
          f"unreflected={sup_expanded.unreflected_energy_ratio:.2e}, "
          f"lineage_root={sup_expanded.globe.reference_generator.name}")

    rng = np.random.default_rng(42)
    ns = np.zeros(len(ref_wave) // 2 + 1, dtype=np.complex128)
    ns[12:-1] = rng.normal(0.0, 4.0, size=len(ns) - 13) + 1j * rng.normal(0.0, 4.0, size=len(ns) - 13)
    sup_rup = chord_globe.superpose(ref_wave + np.fft.irfft(ns, n=len(ref_wave)))
    print(f"\n[Tamper Rupture] {sup_rup.explanation}")
    print(f"  -> Distilled order operator accepted? {sup_rup.unreflected_order_operator is not None}")

    # 4. Real WAV Audio Acoustic Medium
    _section("4. REAL WAV AUDIO: ACOUSTIC PHASE DELAY + HIGH-FREQ DAMPING")
    with wave.open(str(wav_path), "rb") as wf:
        raw = wf.readframes(min(2048, wf.getnframes()))
        wav_arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        if wf.getnchannels() > 1:
            wav_arr = wav_arr.reshape(-1, wf.getnchannels())[:, 0]
        wav_arr = wav_arr[:2048]
    wav_norm = wav_arr / max(float(np.max(np.abs(wav_arr))), 1.0)
    wav_globe = PlanetaryOrderGlobe("creation_dawn_2048", "acoustics_1d", wav_norm)
    hall_op = AcousticWaveOperator("hall", phase_delay_samples=4.5, damping_alpha=1.2, harmonic_gain=0.85)
    sup_wav = wav_globe.superpose(hall_op.apply(wav_norm), world_operator_hint=hall_op)
    print(sup_wav.explanation)
    print(f"  -> Audio Recon Err={sup_wav.reconstruction_error():.2e}")

    # 5. Native Code & Chemistry Mirrors + Phase 8 / CausalField Bridge
    _section("5. CODE / CHEMISTRY STRUCTURAL MIRRORS + PHASE 8 & CAUSAL FIELD BRIDGE")
    code_bands = ("entry_flow", "state_conservation", "loop_bound", "branch_purity", "exit_parity", "mem_safety", "prov_chain", "scale_sync")
    code_ref = np.ones(8, dtype=np.float64)
    code_globe = PlanetaryOrderGlobe("code_invariants", "code_structure", code_ref, is_structural_basis=True, band_names=code_bands)
    code_op = StructuralSpectrumOperator("loop_unroll", "code_structure", code_bands, [1.0, 1.0, 1.45, 0.85, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.3, -0.1, 0.0, 0.0, 0.0, 0.0])
    sup_code = code_globe.superpose(code_op.apply(code_ref), world_operator_hint=code_op)
    print(sup_code.explanation)
    print(f"  -> Code Recon Err={sup_code.reconstruction_error():.2e}")

    tri_unit = chord_globe.superpose(ref_wave).to_trinitarian_system().observe_and_discern()
    tri_rup = sup_rup.to_trinitarian_system().observe_and_discern()
    cf_unit = chord_globe.superpose(ref_wave).sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="u")
    cf_unk = sup_unk.sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="k")
    cf_rup = sup_rup.sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="r")
    for cfg in (cf_unit, cf_unk, cf_rup):
        cfg["causal_field"].step(dt=0.1)

    print(
        f"Phase 8 Verdict: Unit-1={tri_unit.status.value} vs Rupture={tri_rup.status.value}\n"
        f"CausalField Min Beam: Unit-1={cf_unit['min_beam_strength']:.4f}, "
        f"UnknownOrder(curiosity={cf_unk['curiosity_charge']:.1f})={cf_unk['min_beam_strength']:.4f}, "
        f"Rupture={cf_rup['min_beam_strength']:.4f}"
    )

    print("\n" + "=" * 92)
    print("  ALL PHASE 9 UNIT-1 SPECTRAL MIRROR VERIFICATIONS COMPLETED.")
    print("=" * 92)


if __name__ == "__main__":
    run_demo()
