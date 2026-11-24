"""
Verification suite for ELYSIA Phase 9:
Unit-1 Spectral Mirror & Living Causal-Operator Superposition Protocol.
"""

from __future__ import annotations

import math
from pathlib import Path
import wave

import numpy as np
from PIL import Image
import pytest

from core.physics.causal_field import CausalField
from core.physics.trinitarian_scale_binding_protocol import DiscernmentStatus
from core.physics.unit_spectral_mirror_protocol import (
    AcousticWaveOperator,
    IdentityCausalOperator,
    MirrorOriginKind,
    OpticalOcularEyeOperator,
    OpticalWorldMediumOperator,
    PlanetaryOrderGlobe,
    StructuralSpectrumOperator,
)


_IMG_PATH = Path(r"c:\Elysia\demos\real_world_assets\real_human_face_hd.jpg")
_WAV_PATH = Path(r"c:\Elysia\data\unified_creation_dawn.wav")


def _load_face_64_rgb() -> np.ndarray:
    img = Image.open(_IMG_PATH).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0


def _load_wav_slice(n_samples: int = 2048) -> np.ndarray:
    with wave.open(str(_WAV_PATH), "rb") as wf:
        raw = wf.readframes(min(n_samples, wf.getnframes()))
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        if wf.getnchannels() > 1:
            arr = arr.reshape(-1, wf.getnchannels())[:, 0]
        arr = arr[:n_samples]
    return arr / max(float(np.max(np.abs(arr))), 1.0)


def test_unit_one_perfect_reflection_and_lossless_lineage():
    t = np.linspace(0.0, 1.0, 256, endpoint=False)
    ref_wave = 1.2 * np.cos(2.0 * math.pi * 3.0 * t) + 0.7 * np.sin(2.0 * math.pi * 7.0 * t)

    globe = PlanetaryOrderGlobe(
        name="harmonic_base",
        modality="acoustics_1d",
        reference_signal=ref_wave,
    )
    sup = globe.superpose(ref_wave.copy())

    assert sup.origin_kind == MirrorOriginKind.UNIT_ONE_REFLECTION
    assert sup.raw_unit_deviation < 1e-12
    assert sup.world_unit_deviation < 1e-12
    assert sup.unreflected_energy_ratio < 1e-12
    assert sup.reconstruction_error() < 1e-12

    lineage = sup.trace_causal_lineage()
    assert lineage["globe_name"] == "harmonic_base"
    assert lineage["observer_operator"]["role"] == "observer_eye"
    assert lineage["extracted_world_operator"]["role"] == "world_law"


def test_observer_eye_illusion_vs_world_stability_real_face_image():
    face = _load_face_64_rgb()
    globe = PlanetaryOrderGlobe(
        name="human_face_64",
        modality="optics_2d",
        reference_signal=face,
    )

    skewed_eye = OpticalOcularEyeOperator(
        name="aberrated_retina",
        cone_gain=(1.25, 0.55, 0.85),
        corneal_defocus_sigma=1.2,
        ocular_shift=(1.5, -2.0),
    )
    # World itself did NOT change; only the observer's eye transformed perception.
    observed_through_eye = skewed_eye.apply(face)
    sup = globe.superpose(observed_through_eye, observer_operator=skewed_eye)

    assert sup.raw_unit_deviation > 0.20
    assert sup.origin_kind == MirrorOriginKind.OBSERVER_STRUCTURE_ARTIFACT
    assert sup.world_unit_deviation < 0.03
    assert sup.reconstruction_error() < 0.02

    # Counterfactual re-observation through a clear eye recovers Unit-1 directly
    clear_sup = sup.counterfactual_reobserve(IdentityCausalOperator(modality="optics_2d"))
    assert clear_sup.origin_kind == MirrorOriginKind.UNIT_ONE_REFLECTION


def test_world_lawful_transformation_planck_cauchy_and_spatial_onset():
    face = _load_face_64_rgb()
    globe = PlanetaryOrderGlobe(
        name="human_face_64",
        modality="optics_2d",
        reference_signal=face,
    )

    world_medium = OpticalWorldMediumOperator(
        name="warm_prism_occlusion",
        illumination_kelvin=3200.0,
        chromatic_dispersion_shifts=((0.0, 0.2), (0.0, 0.8), (0.0, 1.8)),
        occlusion_box=(40, 58, 36, 56),
        occlusion_factor=0.25,
    )
    observed = world_medium.apply(face)
    sup = globe.superpose(observed, world_operator_hint=world_medium)

    assert sup.origin_kind == MirrorOriginKind.WORLD_LAWFUL_TRANSFORMATION
    assert sup.onset_profile.channel_deviations["B"] > sup.onset_profile.channel_deviations["R"]
    # Lower-right occlusion box sits around y in [0.625, 0.906], x in [0.5625, 0.875]
    assert sup.onset_profile.onset_coord[0] > 0.5
    assert sup.onset_profile.onset_coord[1] > 0.45
    assert sup.reconstruction_error() < 1e-6

    # Transfer extracted world operator zero-shot onto an unseen synthetic checkerboard
    yy, xx = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64), indexing="ij")
    unseen_img = np.stack([yy, xx, 0.5 * np.ones_like(yy)], axis=2)
    transferred = sup.transfer_world_cause_to(unseen_img)
    assert transferred.shape == unseen_img.shape
    assert np.all(np.isfinite(transferred))
    assert not np.allclose(transferred, unseen_img)


def test_acoustic_wave_unknown_coherent_order_distillation_and_mirror_expansion():
    t = np.linspace(0.0, 1.0, 256, endpoint=False)
    ref_wave = 1.2 * np.cos(2.0 * math.pi * 3.0 * t) + 0.7 * np.sin(2.0 * math.pi * 7.0 * t)
    globe = PlanetaryOrderGlobe(
        name="two_tone_chord",
        modality="acoustics_1d",
        reference_signal=ref_wave,
    )

    new_mode_op = AcousticWaveOperator(
        name="overtone_19hz",
        injected_mode=(19.0, 0.6, 0.25),
    )
    obs_wave = new_mode_op.apply(ref_wave)
    sup = globe.superpose(obs_wave, world_operator_hint=new_mode_op)

    assert sup.origin_kind == MirrorOriginKind.UNKNOWN_COHERENT_EXTERNAL_ORDER
    assert sup.unreflected_entropy < 0.35
    assert sup.unreflected_top_concentration > 0.70
    assert sup.unreflected_order_operator is not None

    # Expand the Planetary Globe around the discovered coherent order!
    expanded_sup = sup.absorb_discovered_order()
    assert expanded_sup.origin_kind == MirrorOriginKind.UNIT_ONE_REFLECTION
    assert expanded_sup.unreflected_energy_ratio < 1e-10
    assert expanded_sup.globe.reference_generator.describe_lineage()["parents"]


def test_non_causal_broadband_rupture_rejected_vs_coherent_order():
    t = np.linspace(0.0, 1.0, 256, endpoint=False)
    ref_wave = 1.2 * np.cos(2.0 * math.pi * 3.0 * t) + 0.7 * np.sin(2.0 * math.pi * 7.0 * t)
    globe = PlanetaryOrderGlobe(
        name="two_tone_chord",
        modality="acoustics_1d",
        reference_signal=ref_wave,
    )

    rng = np.random.default_rng(42)
    noise_spec = np.zeros(len(ref_wave) // 2 + 1, dtype=np.complex128)
    # Inject broadband unstructured noise across high frequency unsupported bins
    bins = slice(12, len(noise_spec) - 1)
    noise_spec[bins] = rng.normal(0.0, 4.0, size=noise_spec[bins].shape) + 1j * rng.normal(
        0.0, 4.0, size=noise_spec[bins].shape
    )
    ruptured = ref_wave + np.fft.irfft(noise_spec, n=len(ref_wave))

    sup = globe.superpose(ruptured)
    assert sup.origin_kind == MirrorOriginKind.NON_CAUSAL_RUPTURE
    assert sup.unreflected_entropy > 0.70
    assert sup.unreflected_order_operator is None


def test_real_wav_audio_acoustic_mirror_phase_and_damping():
    audio = _load_wav_slice(2048)
    globe = PlanetaryOrderGlobe(
        name="creation_dawn_excerpt",
        modality="acoustics_1d",
        reference_signal=audio,
    )

    acoustic_medium = AcousticWaveOperator(
        name="hall_propagation",
        phase_delay_samples=4.5,
        damping_alpha=1.2,
        harmonic_gain=0.85,
    )
    observed = acoustic_medium.apply(audio)
    sup = globe.superpose(observed, world_operator_hint=acoustic_medium)

    assert sup.origin_kind == MirrorOriginKind.WORLD_LAWFUL_TRANSFORMATION
    assert sup.world_phase_shift_rad > 0.10
    assert sup.reconstruction_error() < 1e-6

    t = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    probe = np.cos(2.0 * math.pi * 5.0 * t)
    transferred = sup.transfer_world_cause_to(probe)
    assert transferred.shape == probe.shape
    assert np.all(np.isfinite(transferred))


def test_code_and_chemistry_structural_unit_one_mirror():
    # Code invariant spectrum (no spatial FFT flattening)
    code_bands = ("entry_flow", "state_conservation", "loop_bound", "branch_purity", "exit_parity", "mem_safety", "prov_chain", "scale_sync")
    code_ref = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    code_globe = PlanetaryOrderGlobe(
        name="code_invariants",
        modality="code_structure",
        reference_signal=code_ref,
        is_structural_basis=True,
        band_names=code_bands,
    )
    code_op = StructuralSpectrumOperator(
        name="loop_unroll_shift",
        modality="code_structure",
        band_names=code_bands,
        band_gains=[1.0, 1.0, 1.45, 0.85, 1.0, 1.0, 1.0, 1.0],
        band_phase_rad=[0.0, 0.0, 0.3, -0.1, 0.0, 0.0, 0.0, 0.0],
    )
    sup_code = code_globe.superpose(code_op.apply(code_ref), world_operator_hint=code_op)
    assert sup_code.origin_kind == MirrorOriginKind.WORLD_LAWFUL_TRANSFORMATION
    assert sup_code.reconstruction_error() < 1e-12
    assert sup_code.onset_profile.channel_deviations["loop_bound"] > 0.3

    # Chemistry valence/stoichiometry spectrum (H2 + 0.5 O2 -> H2O balance)
    chem_bands = ("H_donor", "O_acceptor", "bond_order", "enthalpy_norm", "entropy_norm")
    chem_ref = np.array([2.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float64)
    chem_globe = PlanetaryOrderGlobe(
        name="water_stoichiometry",
        modality="chemistry_spectrum",
        reference_signal=chem_ref,
        is_structural_basis=True,
        band_names=chem_bands,
    )
    chem_op = StructuralSpectrumOperator(
        name="catalytic_hydration",
        modality="chemistry_spectrum",
        band_names=chem_bands,
        band_gains=[1.0, 1.0, 1.0, 0.72, 1.15],
    )
    sup_chem = chem_globe.superpose(chem_op.apply(chem_ref), world_operator_hint=chem_op)
    assert sup_chem.origin_kind == MirrorOriginKind.WORLD_LAWFUL_TRANSFORMATION
    assert sup_chem.reconstruction_error() < 1e-12


def test_living_mirror_bridges_to_phase8_trinitarian_and_causal_field():
    t = np.linspace(0.0, 1.0, 256, endpoint=False)
    ref_wave = 1.2 * np.cos(2.0 * math.pi * 3.0 * t) + 0.7 * np.sin(2.0 * math.pi * 7.0 * t)
    globe = PlanetaryOrderGlobe("chord", "acoustics_1d", ref_wave)

    # 1. Unit-1 Reflection -> Harmonious Convergence & strong beam (> 0.95)
    sup_unit = globe.superpose(ref_wave.copy())
    tri_unit = sup_unit.to_trinitarian_system()
    v_unit = tri_unit.observe_and_discern()
    assert v_unit.status == DiscernmentStatus.HARMONIOUS_CONVERGENCE

    cf_unit = sup_unit.sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="m_unit")
    assert cf_unit["min_beam_strength"] > 0.95
    cf_unit["causal_field"].step(dt=0.1)

    # 2. Unknown Coherent Order -> curiosity_charge == 1.0
    obs_new = AcousticWaveOperator(injected_mode=(19.0, 0.6, 0.0)).apply(ref_wave)
    sup_unk = globe.superpose(obs_new)
    cf_unk = sup_unk.sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="m_unk")
    assert cf_unk["curiosity_charge"] == 1.0
    cf_unk["causal_field"].step(dt=0.1)

    # 3. Broadband Rupture -> Phase 8 LocalValidUpperViolation & beam < 0.35
    rng = np.random.default_rng(7)
    ns = np.zeros(len(ref_wave) // 2 + 1, dtype=np.complex128)
    ns[15:-1] = rng.normal(0.0, 5.0, size=len(ns) - 16) + 1j * rng.normal(0.0, 5.0, size=len(ns) - 16)
    sup_rup = globe.superpose(ref_wave + np.fft.irfft(ns, n=len(ref_wave)))
    tri_rup = sup_rup.to_trinitarian_system()
    v_rup = tri_rup.observe_and_discern()
    assert v_rup.status == DiscernmentStatus.LOCAL_VALID_UPPER_VIOLATION

    cf_rup = sup_rup.sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="m_rup")
    assert cf_rup["min_beam_strength"] < 0.35
    cf_rup["causal_field"].step(dt=0.1)
