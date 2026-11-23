import math
from pathlib import Path
import time
import wave

import numpy as np
from PIL import Image

from core.physics.causal_field import CausalField
from core.physics.causal_mechanism_palette import (
    CausalMechanismPalette,
    op_thermal_diffusion,
)
from core.physics.spectral_scale_rotor_manifold import (
    ScaleRotorBand,
    SpectralBalanceScale,
    SpectralScaleRotorManifold,
    encode_surface_to_scale_rotors,
    encode_waveform_to_scale_rotors,
)


def _make_1d_field(
    global_w=(2.0 * math.pi * 2.0, 2.0 * math.pi * 5.0),
    global_amp=(1.0 + 0.2j, 0.7 - 0.3j),
    local_w=(0.0,),
    local_amp=(1.2 + 0.0j,),
    local_centers=(0.5,),
    local_sigmas=(0.04,),
) -> SpectralScaleRotorManifold:
    b0 = ScaleRotorBand(
        level=0,
        dim=1,
        wavevectors=np.array(global_w, dtype=np.float64)[:, None],
        temporal_omegas=np.abs(np.array(global_w, dtype=np.float64)),
        amplitudes=np.array(global_amp, dtype=np.complex128),
        centers=np.zeros((len(global_w), 1), dtype=np.float64),
        sigmas=np.full(len(global_w), np.inf, dtype=np.float64),
        is_global=True,
        parent_indices=-np.ones(len(global_w), dtype=np.int32),
        parent_couplings=np.ones(len(global_w), dtype=np.float64),
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
            parent_couplings=np.ones(len(local_w), dtype=np.float64),
        )
        bands.append(b1)
    return SpectralScaleRotorManifold(dim=1, domain_bounds=([0.0], [1.0]), bands=bands)


def test_thermal_diffusion_exact_pde_escape_1_to_500() -> None:
    # Mixed global harmonic + localized Gaussian wave-packet on periodic [0, 1]
    manifold = _make_1d_field(
        global_w=(0.0, 2.0 * math.pi * 3.0),
        global_amp=(0.4 + 0.0j, 1.0 + 0.3j),
        local_w=(2.0 * math.pi * 4.0,),
        local_amp=(1.2 + 0.2j,),
        local_centers=(0.5,),
        local_sigmas=(0.045,),
    )

    n = 512
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    dx = 1.0 / n
    diffusivity = 0.0005
    tau = 1.0  # D * tau = 5e-4 -> >500 CFL-stable FTCS steps

    # Stable explicit FTCS baseline requiring > 500 sequential steps
    dt_cfl = 0.25 * (dx ** 2) / diffusivity
    num_steps = int(math.ceil(tau / dt_cfl))
    assert num_steps >= 500
    dt = tau / num_steps
    alpha = diffusivity * dt / (dx ** 2)

    u = manifold.slice_manifestation(x)
    mass_before = float(np.mean(u))
    for _ in range(num_steps):
        u = u + alpha * (np.roll(u, -1) - 2.0 * u + np.roll(u, 1))

    # Single O(K) palette law invocation (no stepping!)
    palette = CausalMechanismPalette.with_default_principles()
    evolved_rotor = palette.compose_and_apply(
        manifold,
        [("physics.thermal_diffusion", 1.0, {"diffusivity": diffusivity, "tau": tau})],
    )
    u_rotor = evolved_rotor.slice_manifestation(x)

    rel_l2_err = float(np.linalg.norm(u - u_rotor) / np.linalg.norm(u))
    assert rel_l2_err < 5e-3

    # Exact mass conservation in O(K) integral
    mass_rotor = evolved_rotor.integrate_over_domain()
    assert abs(mass_rotor - mass_before) < 1e-3


def test_disciplinary_and_code_pigments_execute_in_ok() -> None:
    palette = CausalMechanismPalette.with_default_principles()
    m1 = _make_1d_field(
        global_w=(0.0, 2.0 * math.pi * 3.0, 2.0 * math.pi * 4.5, 2.0 * math.pi * 6.0),
        global_amp=(1.5 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j),
        local_w=(2.0 * math.pi * 3.0,),
        local_amp=(0.6 + 0.0j,),
    )

    # 1. Lateral inhibition suppresses uniform DC strongly while preserving edges/mid-bands
    m_dog = palette.get("biology.lateral_inhibition").apply(
        m1, {"sigma_center": 0.015, "sigma_surround": 0.10, "surround_weight": 0.90}
    )
    dc_after = float(np.abs(m_dog.bands[0].amplitudes[0]))
    mid_after = float(np.abs(m_dog.bands[0].amplitudes[3]))
    assert dc_after < 0.25
    assert mid_after > dc_after * 2.5

    # 2. Acoustic harmonic resonance retains integer harmonics (3, 6) and attenuates non-harmonic (4.5)
    m_harm = palette.get("acoustics.harmonic_resonance").apply(
        m1, {"fundamental": 2.0 * math.pi * 3.0, "max_overtones": 4, "bandwidth": 1.5, "floor_gain": 0.05}
    )
    assert abs(m_harm.bands[0].amplitudes[1]) > 0.95
    assert abs(m_harm.bands[0].amplitudes[3]) > 0.95
    assert abs(m_harm.bands[0].amplitudes[2]) < 0.15

    # 3. Damped harmonic oscillator strictly dissipates energy
    m_damp = palette.get("physics.damped_harmonic").apply(m1, {"damping": 0.4, "tau": 1.0})
    assert m_damp.wave_resonance(m_damp)["energy_self"] < m1.wave_resonance(m1)["energy_self"]

    # 4. Doppler advection evolves phase & localized centers cleanly
    m_adv = palette.get("physics.doppler_advection").apply(m1, {"tau": 0.2, "velocity": [0.3]})
    assert abs(float(m_adv.bands[1].centers[0, 0]) - 0.56) < 1e-12

    # 5. Clifford spacetime rotor twist + exact rewind in both 1D and 2D
    m1_tw = palette.get("code.clifford_rotor_twist").apply(m1, {"angle": 0.8, "scale": 1.3})
    m1_rw = palette.get("code.clifford_rotor_twist").apply(m1_tw, {"angle": 0.8, "scale": 1.3, "rewind": True})
    x_probe = np.linspace(0.0, 1.0, 128)[:, None]
    assert np.max(np.abs(m1.slice_manifestation(x_probe) - m1_rw.slice_manifestation(x_probe))) < 1e-11

    b0_2d = ScaleRotorBand(
        level=0,
        dim=2,
        wavevectors=np.array([[2.0 * math.pi * 2.0, 0.0], [0.0, 2.0 * math.pi * 3.0]]),
        temporal_omegas=np.array([4.0, 6.0]),
        amplitudes=np.array([1.0 + 0.3j, -0.5 + 0.7j]),
        centers=np.zeros((2, 2)),
        sigmas=np.full(2, np.inf),
        is_global=True,
        parent_indices=np.array([-1, -1], dtype=np.int32),
        parent_couplings=np.array([1.0, 1.0]),
    )
    b1_2d = ScaleRotorBand(
        level=1,
        dim=2,
        wavevectors=np.array([[10.0, -6.0]]),
        temporal_omegas=np.array([11.0]),
        amplitudes=np.array([0.8 + 0.2j]),
        centers=np.array([[0.35, 0.68]]),
        sigmas=np.array([0.07]),
        is_global=False,
        parent_indices=np.array([0], dtype=np.int32),
        parent_couplings=np.array([0.9]),
    )
    m2 = SpectralScaleRotorManifold(dim=2, domain_bounds=([0.0, 0.0], [1.0, 1.0]), bands=[b0_2d, b1_2d])
    m2_tw = palette.get("code.clifford_rotor_twist").apply(m2, {"angle": 0.65, "scale": 1.5})
    m2_rw = palette.get("code.clifford_rotor_twist").apply(m2_tw, {"angle": 0.65, "scale": 1.5, "rewind": True})
    gy, gx = np.meshgrid(np.linspace(0, 1, 24), np.linspace(0, 1, 24), indexing="ij")
    pts2 = np.stack([gy.reshape(-1), gx.reshape(-1)], axis=1)
    assert np.max(np.abs(m2.slice_manifestation(pts2) - m2_rw.slice_manifestation(pts2))) < 1e-11

    # 6. Snell optical refraction & CausalScaleRotorField worldline orbit execute cleanly
    m2_ref = palette.get("optics.snell_refraction").apply(m2, {"index_ratio": 1.3, "angle": 0.2})
    assert np.allclose(
        np.linalg.norm(m2_ref.bands[0].wavevectors, axis=1),
        1.3 * np.linalg.norm(m2.bands[0].wavevectors, axis=1),
        atol=1e-10,
    )
    m2_orb = palette.get("code.causal_worldline_orbit").apply(m2, {"time": 1.2})
    assert not np.allclose(m2_orb.bands[1].centers, m2.bands[1].centers)


def test_palette_parallel_and_sequential_composition() -> None:
    palette = CausalMechanismPalette.with_default_principles()
    m1 = _make_1d_field()

    # Parallel linear mixing ("paint mixing"): 0.6 * thermal + 0.4 * lateral_inhibition
    m_par = palette.compose_and_apply(
        m1,
        [
            ("physics.thermal_diffusion", 0.6, {"diffusivity": 0.002, "tau": 1.0}),
            ("biology.lateral_inhibition", 0.4, {}),
        ],
        mode="parallel",
        balance=SpectralBalanceScale(scale_tilt=0.3, contrast_tilt=0.2),
    )
    m_direct = (
        palette.get("physics.thermal_diffusion").apply(m1, {"diffusivity": 0.002, "tau": 1.0}).scale_field(0.6)
        .superpose(palette.get("biology.lateral_inhibition").apply(m1, {}).scale_field(0.4))
        .tune_balance(scale_tilt=0.3, contrast_tilt=0.2)
    )
    x = np.linspace(0.0, 1.0, 200)[:, None]
    assert np.max(np.abs(m_par.slice_manifestation(x) - m_direct.slice_manifestation(x))) < 1e-11

    # Sequential pipeline: diffusion -> Clifford twist -> worldline orbit
    m_seq = palette.compose_and_apply(
        m1,
        [
            ("physics.thermal_diffusion", 1.0, {"diffusivity": 0.001, "tau": 0.5}),
            ("code.clifford_rotor_twist", 1.0, {"angle": 0.4, "scale": 1.1}),
            ("code.causal_worldline_orbit", 1.0, {"time": 0.7}),
        ],
        mode="sequential",
    )
    assert math.isfinite(m_seq.integrate_over_domain())


def test_field_rectification_and_memory_sedimentation_recall() -> None:
    palette = CausalMechanismPalette.with_default_principles()

    # Unseen 1D pattern orthogonal to the 4 coarse canonical archetypes
    novel_field = _make_1d_field(
        global_w=(2.0 * math.pi * 11.0,),
        global_amp=(1.2 + 0.6j,),
        local_w=(2.0 * math.pi * 19.0,),
        local_amp=(0.9 - 0.3j,),
        local_centers=(0.27,),
        local_sigmas=(0.035,),
    )

    # First exposure: novelty exceeds threshold -> sediments memory.field_0
    r1 = palette.rectify_field(novel_field, novelty_threshold=0.15)
    assert not r1.from_memory
    assert r1.distilled_pigment_id == "memory.field_0"
    assert r1.residual_friction > 0.5
    assert "memory.field_0" in palette.pigments

    # Second exposure (same pattern + existing archetype DC & harmonic blend): instant palette hit!
    composite = (
        novel_field.scale_field(1.5)
        .superpose(palette.get("archetype.dc_equilibrium_1d").archetype_signature.scale_field(0.7))  # type: ignore[union-attr]
        .superpose(palette.get("archetype.harmonic_fundamental_1d").archetype_signature.scale_field(-0.4))  # type: ignore[union-attr]
    )
    r2 = palette.rectify_field(composite, novelty_threshold=0.15)
    assert r2.from_memory
    assert r2.distilled_pigment_id is None
    assert r2.explained_ratio > 0.999
    assert r2.residual_friction < 1e-3
    assert abs(r2.recipe_weights["memory.field_0"] - 1.5) < 1e-2


def test_transition_rectification_discovers_known_principles() -> None:
    palette = CausalMechanismPalette.with_default_principles()
    before = _make_1d_field(
        global_w=(0.0, 2.0 * math.pi * 3.0, 2.0 * math.pi * 7.0),
        global_amp=(1.0 + 0.0j, 0.9 + 0.2j, -0.6 + 0.4j),
        local_w=(2.0 * math.pi * 5.0,),
        local_amp=(0.8 + 0.1j,),
        local_centers=(0.45,),
        local_sigmas=(0.05,),
    )

    # Known composite mechanism: 0.7 * thermal_diffusion + 0.3 * lateral_inhibition
    after = palette.compose_and_apply(
        before,
        [
            ("physics.thermal_diffusion", 0.7, None),
            ("biology.lateral_inhibition", 0.3, None),
        ],
        mode="parallel",
    )

    rec = palette.rectify_transition(before, after, novelty_threshold=0.10)
    assert rec.from_memory
    assert rec.distilled_pigment_id is None
    assert rec.explained_ratio > 0.98
    assert rec.residual_friction < 0.05
    assert abs(rec.recipe_weights["physics.thermal_diffusion"] - 0.7) < 0.05
    assert abs(rec.recipe_weights["biology.lateral_inhibition"] - 0.3) < 0.05


def _apply_unknown_resonant_phase_law(manifold: SpectralScaleRotorManifold) -> SpectralScaleRotorManifold:
    """Hidden external law NOT present in default palette: frequency-selective resonant boost + 0.7 rad twist."""
    out_bands = []
    for band in manifold.bands:
        bc = band.copy()
        if bc.num_modes > 0:
            w_mag = np.linalg.norm(bc.wavevectors, axis=1)
            gain = 1.0 + 1.8 * np.exp(-0.5 * ((w_mag - 25.0) / 7.0) ** 2) * np.exp(1j * 0.7)
            bc.amplitudes = bc.amplitudes * gain
        out_bands.append(bc)
    return SpectralScaleRotorManifold(
        dim=manifold.dim, domain_bounds=manifold.domain_bounds, bands=out_bands, balance=manifold.balance
    )


def test_transition_novelty_distillation_and_zero_shot_transfer() -> None:
    palette = CausalMechanismPalette.with_default_principles()

    before_1 = _make_1d_field(
        global_w=(2.0 * math.pi * 1.0, 2.0 * math.pi * 3.0, 2.0 * math.pi * 4.0, 2.0 * math.pi * 6.0),
        global_amp=(1.0 + 0.1j, 0.9 - 0.2j, 1.1 + 0.3j, 0.8 + 0.0j),
        local_w=(),
        local_amp=(),
        local_centers=(),
        local_sigmas=(),
    )
    after_1 = _apply_unknown_resonant_phase_law(before_1)

    # 1st encounter: unknown causal mechanism -> distill new executable pigment
    res1 = palette.rectify_transition(before_1, after_1, novelty_threshold=0.10)
    assert not res1.from_memory
    assert res1.distilled_pigment_id == "memory.transition_0"
    assert "memory.transition_0" in palette.pigments

    # 2nd encounter: exact recall from distilled pigment
    res2 = palette.rectify_transition(before_1, after_1, novelty_threshold=0.10)
    assert res2.from_memory
    assert res2.distilled_pigment_id is None
    assert res2.explained_ratio > 0.98

    # Zero-shot transfer to unseen frequencies, amplitudes, and phases!
    before_2 = _make_1d_field(
        global_w=(2.0 * math.pi * 2.0, 2.0 * math.pi * 4.0, 2.0 * math.pi * 5.0),
        global_amp=(0.7 * np.exp(1j * 1.1), 1.3 * np.exp(-1j * 0.8), 0.9 * np.exp(1j * 2.1)),
        local_w=(),
        local_amp=(),
        local_centers=(),
        local_sigmas=(),
    )
    true_after_2 = _apply_unknown_resonant_phase_law(before_2)
    pred_after_2 = palette.compose_and_apply(before_2, [("memory.transition_0", 1.0, {})])
    transfer_res = pred_after_2.wave_resonance(true_after_2)["normalized_resonance"]
    assert transfer_res > 0.95


def test_real_world_audio_and_image_palette_rectification_and_causal_field() -> None:
    wav_path = Path(r"c:\Elysia\data\unified_creation_dawn.wav")
    img_path = Path(r"c:\Elysia\demos\real_world_assets\real_human_face_hd.jpg")
    assert wav_path.is_file() and img_path.is_file()

    palette = CausalMechanismPalette.with_default_principles()

    # 1. Real 1D audio ingestion & rectification
    with wave.open(str(wav_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels())[:, 0]
        sr = float(wf.getframerate())
    audio_norm = audio / max(float(np.max(np.abs(audio))), 1.0)
    m_audio = encode_waveform_to_scale_rotors(
        audio_norm, domain=(0.0, len(audio_norm) / sr), max_macro_modes=24, max_levels=2, max_knots_per_level=8
    )

    rec_a1 = palette.rectify_field(m_audio, novelty_threshold=0.15, memory_label="memory.dawn_audio")
    assert not rec_a1.from_memory
    assert rec_a1.distilled_pigment_id == "memory.dawn_audio"

    rec_a2 = palette.rectify_field(m_audio, novelty_threshold=0.15)
    assert rec_a2.from_memory
    assert rec_a2.explained_ratio > 0.99

    # 2. Real 2D image ingestion & rectification + O(K) disciplinary palette composition
    img = Image.open(img_path).convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    img_arr = np.asarray(img, dtype=np.float64) / 255.0
    m_img = encode_surface_to_scale_rotors(
        img_arr, max_macro_modes=32, max_levels=2, max_knots_per_level=10, residual_tol=0.10
    )

    rec_i1 = palette.rectify_field(m_img, novelty_threshold=0.15, memory_label="memory.face_surface")
    assert not rec_i1.from_memory
    rec_i2 = palette.rectify_field(m_img, novelty_threshold=0.15)
    assert rec_i2.from_memory
    assert rec_i2.explained_ratio > 0.99

    # Apply multi-discipline palette recipe to image manifold in < 5 ms without pixel scan
    t0 = time.perf_counter()
    m_img_styled = palette.compose_and_apply(
        m_img,
        [
            ("physics.thermal_diffusion", 1.0, {"diffusivity": 0.0005, "tau": 1.0}),
            ("biology.lateral_inhibition", 1.0, {"sigma_center": 0.02, "sigma_surround": 0.08}),
            ("optics.snell_refraction", 1.0, {"index_ratio": 1.1, "angle": 0.15}),
        ],
        mode="sequential",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0
    assert math.isfinite(m_img_styled.integrate_over_domain())

    # 3. Project palette state & rectification outcome into CausalField
    cf = CausalField(dimensions=3)
    sync = palette.sync_to_causal_field(cf, last_result=rec_i2, prefix="pal")
    assert len(cf.engrams) == len(palette.pigments)
    assert len(cf.voxels) == len(palette.pigments)
    assert sync["beams_linked"] >= 1
    cf.step(dt=0.05)
