import math
import time
from pathlib import Path
import wave

import numpy as np
from PIL import Image

from core.physics.causal_field import CausalField
from core.physics.spectral_scale_rotor_manifold import (
    ScaleRotorBand,
    SpectralBalanceScale,
    SpectralScaleRotorManifold,
    encode_surface_to_scale_rotors,
    encode_waveform_to_scale_rotors,
)


def _make_1d_test_manifold(
    global_w=(4.0, 11.0),
    global_amp=(1.2 + 0.3j, 0.7 - 0.5j),
    local_w=(25.0,),
    local_amp=(0.9 + 0.4j,),
    local_centers=(0.52,),
    local_sigmas=(0.07,),
    dispersion_speed: float = 2.0,
) -> SpectralScaleRotorManifold:
    w0 = np.array(global_w, dtype=np.float64)[:, None]
    a0 = np.array(global_amp, dtype=np.complex128)
    b0 = ScaleRotorBand(
        level=0,
        dim=1,
        wavevectors=w0,
        temporal_omegas=dispersion_speed * np.abs(w0[:, 0]),
        amplitudes=a0,
        centers=np.zeros((len(global_w), 1), dtype=np.float64),
        sigmas=np.full(len(global_w), np.inf, dtype=np.float64),
        is_global=True,
        parent_indices=-np.ones(len(global_w), dtype=np.int32),
        parent_couplings=np.ones(len(global_w), dtype=np.float64),
    )
    bands = [b0]
    if local_w:
        w1 = np.array(local_w, dtype=np.float64)[:, None]
        a1 = np.array(local_amp, dtype=np.complex128)
        c1 = np.array(local_centers, dtype=np.float64)[:, None]
        s1 = np.array(local_sigmas, dtype=np.float64)
        b1 = ScaleRotorBand(
            level=1,
            dim=1,
            wavevectors=w1,
            temporal_omegas=dispersion_speed * np.abs(w1[:, 0]),
            amplitudes=a1,
            centers=c1,
            sigmas=s1,
            is_global=False,
            parent_indices=np.zeros(len(local_w), dtype=np.int32),
            parent_couplings=np.ones(len(local_w), dtype=np.float64),
        )
        bands.append(b1)
    return SpectralScaleRotorManifold(dim=1, domain_bounds=([0.0], [1.0]), bands=bands)


def test_exact_spacetime_evolution_no_1_to_100_stepping() -> None:
    # 1D multi-scale manifold with drift velocity
    m1 = _make_1d_test_manifold()
    vel_1d = np.array([0.002], dtype=np.float64)
    dt_jump = 100.0

    # Direct O(K) evolution vs 100 sequential steps of dt=1.0
    m_direct = m1.evolve_spacetime(dt=dt_jump, velocity=vel_1d)
    m_stepped = m1
    for _ in range(100):
        m_stepped = m_stepped.evolve_spacetime(dt=1.0, velocity=vel_1d)

    probe_1d = np.linspace(0.0, 1.0, 512, endpoint=False)[:, None]
    y_direct = m_direct.slice_manifestation(probe_1d)
    y_stepped = m_stepped.slice_manifestation(probe_1d)
    assert np.max(np.abs(y_direct - y_stepped)) < 1e-10

    # Large jump dt=10000 and exact rewind dt=-10000 recover original state
    m_far = m1.evolve_spacetime(dt=10000.0, velocity=vel_1d)
    m_rewound = m_far.evolve_spacetime(dt=-10000.0, velocity=vel_1d)
    y_orig = m1.slice_manifestation(probe_1d)
    y_rewound = m_rewound.slice_manifestation(probe_1d)
    assert np.max(np.abs(y_orig - y_rewound)) < 1e-9

    # 2D global + localized rotor evolution & rewind
    b0 = ScaleRotorBand(
        level=0,
        dim=2,
        wavevectors=np.array([[2.0 * math.pi, 0.0], [0.0, 4.0 * math.pi]]),
        temporal_omegas=np.array([3.5, 6.2]),
        amplitudes=np.array([1.0 + 0.2j, -0.6 + 0.8j]),
        centers=np.zeros((2, 2)),
        sigmas=np.full(2, np.inf),
        is_global=True,
        parent_indices=np.array([-1, -1], dtype=np.int32),
        parent_couplings=np.array([1.0, 1.0]),
    )
    b1 = ScaleRotorBand(
        level=1,
        dim=2,
        wavevectors=np.array([[18.0, -12.0]]),
        temporal_omegas=np.array([9.0]),
        amplitudes=np.array([0.5 - 0.3j]),
        centers=np.array([[0.45, 0.55]]),
        sigmas=np.array([0.08]),
        is_global=False,
        parent_indices=np.array([0], dtype=np.int32),
        parent_couplings=np.array([0.8]),
    )
    m2 = SpectralScaleRotorManifold(dim=2, domain_bounds=([0.0, 0.0], [1.0, 1.0]), bands=[b0, b1])
    vel_2d = np.array([0.001, -0.0015])
    m2_fwd = m2.evolve_spacetime(dt=100.0, velocity=vel_2d)
    m2_back = m2_fwd.evolve_spacetime(dt=-100.0, velocity=vel_2d)

    gy, gx = np.meshgrid(np.linspace(0, 1, 32, endpoint=False), np.linspace(0, 1, 32, endpoint=False), indexing="ij")
    pts2 = np.stack([gy.reshape(-1), gx.reshape(-1)], axis=1)
    assert np.max(np.abs(m2.slice_manifestation(pts2) - m2_back.slice_manifestation(pts2))) < 1e-11


def test_exact_definite_integral_1d_and_2d() -> None:
    # 1D mixed global + localized manifold
    m1 = _make_1d_test_manifold(
        global_w=(0.0, 5.3, 14.7),
        global_amp=(0.4 + 0.0j, 1.1 + 0.6j, -0.8 + 0.3j),
        local_w=(12.0, -8.5),
        local_amp=(0.9 - 0.2j, 0.6 + 0.5j),
        local_centers=(0.35, 0.72),
        local_sigmas=(0.05, 0.06),
    )
    analytic_1d = m1.integrate_over_domain()

    n_quad = 20000
    x_quad = np.linspace(0.0, 1.0, n_quad, endpoint=False) + 0.5 / n_quad
    numeric_1d = float(np.mean(m1.slice_manifestation(x_quad[:, None])))
    assert abs(analytic_1d - numeric_1d) < 1e-3

    # Also test sub-domain integration [0.2, 0.85]
    analytic_sub = m1.integrate_over_domain(bounds=([0.2], [0.85]))
    x_sub = np.linspace(0.2, 0.85, n_quad, endpoint=False) + 0.65 / (2 * n_quad)
    numeric_sub = float(np.mean(m1.slice_manifestation(x_sub[:, None])) * 0.65)
    assert abs(analytic_sub - numeric_sub) < 1e-3

    # 2D mixed global + localized manifold
    b0 = ScaleRotorBand(
        level=0,
        dim=2,
        wavevectors=np.array([[0.0, 0.0], [3.7, -5.2], [9.1, 4.4]]),
        temporal_omegas=np.array([0.0, 6.4, 10.1]),
        amplitudes=np.array([0.3 + 0.0j, 0.8 - 0.4j, -0.5 + 0.7j]),
        centers=np.zeros((3, 2)),
        sigmas=np.full(3, np.inf),
        is_global=True,
        parent_indices=np.array([-1, -1, -1], dtype=np.int32),
        parent_couplings=np.array([1.0, 1.0, 1.0]),
    )
    b1 = ScaleRotorBand(
        level=1,
        dim=2,
        wavevectors=np.array([[10.0, -7.0], [-4.0, 11.0]]),
        temporal_omegas=np.array([12.2, 11.7]),
        amplitudes=np.array([1.3 + 0.4j, -0.7 + 0.9j]),
        centers=np.array([[0.40, 0.60], [0.65, 0.35]]),
        sigmas=np.array([0.07, 0.06]),
        is_global=False,
        parent_indices=np.array([1, 2], dtype=np.int32),
        parent_couplings=np.array([0.8, 0.9]),
    )
    m2 = SpectralScaleRotorManifold(dim=2, domain_bounds=([0.0, 0.0], [1.0, 1.0]), bands=[b0, b1])
    analytic_2d = m2.integrate_over_domain()

    n2 = 128
    dy = 1.0 / n2
    ys = np.linspace(0.0, 1.0, n2, endpoint=False) + 0.5 * dy
    xs = np.linspace(0.0, 1.0, n2, endpoint=False) + 0.5 * dy
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pts2 = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
    numeric_2d = float(np.mean(m2.slice_manifestation(pts2)))
    assert abs(analytic_2d - numeric_2d) < 1e-3


def test_exact_wave_resonance_all_pair_types_no_aliasing() -> None:
    def _num_inner(ma: SpectralScaleRotorManifold, mb: SpectralScaleRotorManifold, n: int = 16000) -> float:
        x = (np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 / n)[:, None]
        return float(np.mean(ma.slice_manifestation(x) * mb.slice_manifestation(x)))

    # (a) Global x Global (asymmetric non-harmonic frequencies)
    g_a = _make_1d_test_manifold(
        global_w=(3.3, 10.7), global_amp=(1.2 + 0.4j, -0.6 + 0.9j), local_w=(), local_amp=(), local_centers=(), local_sigmas=()
    )
    g_b = _make_1d_test_manifold(
        global_w=(5.1, 10.7), global_amp=(0.5 - 0.8j, 1.0 + 0.2j), local_w=(), local_amp=(), local_centers=(), local_sigmas=()
    )
    assert abs(g_a.wave_resonance(g_b)["raw_overlap"] - _num_inner(g_a, g_b)) < 1e-3

    # (b) Local x Local (different centers, sigmas, complex amplitudes, frequencies)
    l_a = SpectralScaleRotorManifold(
        dim=1,
        domain_bounds=([0.0], [1.0]),
        bands=[
            ScaleRotorBand(
                level=1,
                dim=1,
                wavevectors=np.array([[14.0], [-7.5]]),
                temporal_omegas=np.array([14.0, 7.5]),
                amplitudes=np.array([1.1 + 0.5j, 0.7 - 0.4j]),
                centers=np.array([[0.42], [0.61]]),
                sigmas=np.array([0.06, 0.05]),
                is_global=False,
                parent_indices=np.array([-1, -1], dtype=np.int32),
                parent_couplings=np.array([1.0, 1.0]),
            )
        ],
    )
    l_b = SpectralScaleRotorManifold(
        dim=1,
        domain_bounds=([0.0], [1.0]),
        bands=[
            ScaleRotorBand(
                level=1,
                dim=1,
                wavevectors=np.array([[11.0], [22.0]]),
                temporal_omegas=np.array([11.0, 22.0]),
                amplitudes=np.array([0.8 - 0.3j, -0.9 + 0.2j]),
                centers=np.array([[0.46], [0.58]]),
                sigmas=np.array([0.07, 0.045]),
                is_global=False,
                parent_indices=np.array([-1, -1], dtype=np.int32),
                parent_couplings=np.array([1.0, 1.0]),
            )
        ],
    )
    assert abs(l_a.wave_resonance(l_b)["raw_overlap"] - _num_inner(l_a, l_b)) < 1e-3

    # (c) Global x Local (and symmetry Local x Global)
    assert abs(g_a.wave_resonance(l_b)["raw_overlap"] - _num_inner(g_a, l_b)) < 1e-3
    assert abs(l_b.wave_resonance(g_a)["raw_overlap"] - g_a.wave_resonance(l_b)["raw_overlap"]) < 1e-12

    # (d) Mixed multi-level manifolds
    m_a = _make_1d_test_manifold()
    m_b = _make_1d_test_manifold(
        global_w=(4.0, 13.0),
        global_amp=(0.9 + 0.1j, 0.6 + 0.4j),
        local_w=(22.0,),
        local_amp=(0.7 - 0.5j,),
        local_centers=(0.55,),
        local_sigmas=(0.06,),
    )
    assert abs(m_a.wave_resonance(m_b)["raw_overlap"] - _num_inner(m_a, m_b)) < 1e-3

    # (e) Anti-aliasing check: 120-degree phase shift must NEVER wrap back to +1.0!
    base = _make_1d_test_manifold(
        global_w=(2.0 * math.pi * 4.0,),
        global_amp=(1.0 + 0.0j,),
        local_w=(),
        local_amp=(),
        local_centers=(),
        local_sigmas=(),
    )
    shifted_120 = _make_1d_test_manifold(
        global_w=(2.0 * math.pi * 4.0,),
        global_amp=(np.exp(1j * (2.0 * math.pi / 3.0)),),
        local_w=(),
        local_amp=(),
        local_centers=(),
        local_sigmas=(),
    )
    res_120 = base.wave_resonance(shifted_120)["normalized_resonance"]
    assert abs(res_120 - (-0.5)) < 1e-4
    # Scaling granularity must monotonically adjust rather than wrapping cos(delta*scale) back to +1
    res_finer = base.tune_balance(granularity_scale=0.33).wave_resonance(shifted_120)["normalized_resonance"]
    assert res_finer < 0.0


def test_adaptive_scale_splitting_smooth_vs_event_knot() -> None:
    # 1D smooth 2-sine signal -> 0 lower rotors spawned
    n = 1024
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    smooth_1d = 1.4 * np.cos(2.0 * math.pi * 3.0 * x) + 0.8 * np.sin(2.0 * math.pi * 7.0 * x)
    m_smooth = encode_waveform_to_scale_rotors(smooth_1d, domain=(0.0, 1.0), residual_tol=0.02)
    assert len(m_smooth.bands) == 1
    assert m_smooth.bands[0].role == "macro_carrier"
    assert m_smooth.bands[0].num_modes <= 3

    # Add a sharp localized shock pulse at x = 0.63 -> lower rotors spawn near 0.63
    shock_pos = 0.63
    shock_1d = smooth_1d + 3.0 * np.exp(-0.5 * ((x - shock_pos) / 0.018) ** 2)
    m_macro_only = encode_waveform_to_scale_rotors(
        shock_1d, domain=(0.0, 1.0), max_macro_modes=8, max_levels=1
    )
    m_shock = encode_waveform_to_scale_rotors(
        shock_1d, domain=(0.0, 1.0), max_macro_modes=8, max_levels=3, max_knots_per_level=6, residual_tol=0.03
    )
    assert len(m_shock.bands) >= 2
    lower_centers = np.concatenate([b.centers[:, 0] for b in m_shock.bands[1:]])
    assert np.min(np.abs(lower_centers - shock_pos)) < 0.04

    err_macro_only = float(np.linalg.norm(shock_1d - m_macro_only.slice_manifestation(x)))
    err_with_knots = float(np.linalg.norm(shock_1d - m_shock.slice_manifestation(x)))
    assert err_with_knots < err_macro_only * 0.5

    # 2D smooth surface vs surface + sharp localized defect
    ny, nx = 64, 64
    ys = np.linspace(0.0, 1.0, ny, endpoint=False)
    xs = np.linspace(0.0, 1.0, nx, endpoint=False)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    smooth_2d = np.cos(2.0 * math.pi * 2.0 * yy) + 0.6 * np.sin(2.0 * math.pi * 3.0 * xx)
    m2_smooth = encode_surface_to_scale_rotors(smooth_2d, residual_tol=0.02)
    assert len(m2_smooth.bands) == 1
    assert m2_smooth.bands[0].role == "macro_surface"

    knot_yx = (0.31, 0.72)
    shock_2d = smooth_2d + 2.5 * np.exp(
        -(((yy - knot_yx[0]) ** 2 + (xx - knot_yx[1]) ** 2) / (2.0 * 0.03 ** 2))
    )
    m2_shock = encode_surface_to_scale_rotors(
        shock_2d, max_macro_modes=8, max_levels=3, max_knots_per_level=6, residual_tol=0.05
    )
    assert len(m2_shock.bands) >= 2
    pts2 = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
    m2_macro = encode_surface_to_scale_rotors(shock_2d, max_macro_modes=8, max_levels=1)
    err2_macro = float(np.linalg.norm(shock_2d.reshape(-1) - m2_macro.slice_manifestation(pts2)))
    err2_knots = float(np.linalg.norm(shock_2d.reshape(-1) - m2_shock.slice_manifestation(pts2)))
    assert err2_knots < err2_macro * 0.6


def test_spectral_balance_scale_light_dark_and_macro_micro() -> None:
    # Construct manifold with bright (+1 alignment) & dark (-1 alignment) modes across levels
    b0 = ScaleRotorBand(
        level=0,
        dim=1,
        wavevectors=np.array([[2.0 * math.pi * 2.0], [2.0 * math.pi * 5.0]]),
        temporal_omegas=np.array([2.0, 5.0]),
        amplitudes=np.array([1.0 + 0.0j, -0.8 + 0.0j]),  # one bright (0), one dark (pi)
        centers=np.zeros((2, 1)),
        sigmas=np.full(2, np.inf),
        is_global=True,
        parent_indices=np.array([-1, -1], dtype=np.int32),
        parent_couplings=np.array([1.0, 1.0]),
    )
    b1 = ScaleRotorBand(
        level=1,
        dim=1,
        wavevectors=np.array([[2.0 * math.pi * 12.0], [2.0 * math.pi * 19.0]]),
        temporal_omegas=np.array([12.0, 19.0]),
        amplitudes=np.array([0.8 + 0.0j, -0.8 + 0.0j]),
        centers=np.array([[0.35], [0.70]]),
        sigmas=np.array([0.05, 0.05]),
        is_global=False,
        parent_indices=np.array([0, 1], dtype=np.int32),
        parent_couplings=np.array([0.7, 0.7]),
    )
    m = SpectralScaleRotorManifold(dim=1, domain_bounds=([0.0], [1.0]), bands=[b0, b1])

    # Neutral balance scale must be exact identity
    assert np.allclose(m.effective_amplitudes(b0), b0.amplitudes, atol=1e-15)
    assert np.allclose(m.effective_amplitudes(b1), b1.amplitudes, atol=1e-15)

    # Sweep scale_tilt: micro-to-macro energy ratio increases strictly monotonically
    scale_tilts = [-1.0, -0.5, 0.0, 0.5, 1.0]
    micro_macro_ratios = [
        m.tune_balance(scale_tilt=st).spectral_axis_profile()["micro_to_macro_ratio"]
        for st in scale_tilts
    ]
    assert all(a < b for a, b in zip(micro_macro_ratios, micro_macro_ratios[1:]))

    # Sweep contrast_tilt: bright-to-dark ratio increases strictly monotonically
    contrast_tilts = [-1.0, -0.5, 0.0, 0.5, 1.0]
    bright_dark_ratios = [
        m.tune_balance(contrast_tilt=ct).spectral_axis_profile()["bright_to_dark_ratio"]
        for ct in contrast_tilts
    ]
    assert all(a < b for a, b in zip(bright_dark_ratios, bright_dark_ratios[1:]))

    # Granularity scale > 1.0 smoothly attenuates micro band while preserving macro band
    p_fine = m.tune_balance(granularity_scale=1.0).spectral_axis_profile()
    p_coarse = m.tune_balance(granularity_scale=2.5).spectral_axis_profile()
    assert abs(p_coarse["macro_energy"] - p_fine["macro_energy"]) < 1e-12
    assert p_coarse["micro_energy"] < p_fine["micro_energy"] * 0.2


def test_bidirectional_emit_observe_coupling_order_independent() -> None:
    # Constructive pair (small phase offset 0.45 rad)
    emitter = _make_1d_test_manifold(
        global_w=(2.0 * math.pi * 3.0, 2.0 * math.pi * 7.0),
        global_amp=(1.0 + 0.0j, 0.6 + 0.0j),
        local_w=(24.0,),
        local_amp=(0.5 + 0.0j,),
        local_centers=(0.5,),
        local_sigmas=(0.06,),
    )
    observer = _make_1d_test_manifold(
        global_w=(2.0 * math.pi * 3.0, 2.0 * math.pi * 7.0),
        global_amp=(np.exp(1j * 0.45), 0.6 * np.exp(1j * 0.45)),
        local_w=(24.0,),
        local_amp=(0.5 * np.exp(1j * 0.45),),
        local_centers=(0.5,),
        local_sigmas=(0.06,),
    )

    res_before = emitter.wave_resonance(observer)["normalized_resonance"]
    e_after, o_after, metrics = emitter.bidirectional_couple(observer, coupling_gain=0.35, inhibition_gain=0.25)
    assert metrics["resonance_after"] > res_before
    # Both emitter (recoil) and observer (alignment) mutually shifted phase
    assert not np.allclose(e_after.bands[0].amplitudes, emitter.bands[0].amplitudes)
    assert not np.allclose(o_after.bands[0].amplitudes, observer.bands[0].amplitudes)

    # Order independence: permuting mode order inside bands yields identical coupled field
    perm_obs = SpectralScaleRotorManifold(
        dim=1,
        domain_bounds=observer.domain_bounds,
        bands=[
            ScaleRotorBand(
                level=0,
                dim=1,
                wavevectors=observer.bands[0].wavevectors[::-1],
                temporal_omegas=observer.bands[0].temporal_omegas[::-1],
                amplitudes=observer.bands[0].amplitudes[::-1],
                centers=observer.bands[0].centers[::-1],
                sigmas=observer.bands[0].sigmas[::-1],
                is_global=True,
                parent_indices=observer.bands[0].parent_indices[::-1],
                parent_couplings=observer.bands[0].parent_couplings[::-1],
            ),
            observer.bands[1],
        ],
    )
    e_perm, o_perm, _ = emitter.bidirectional_couple(perm_obs, coupling_gain=0.35, inhibition_gain=0.25)
    probe = np.linspace(0.0, 1.0, 256)[:, None]
    assert np.max(np.abs(e_after.slice_manifestation(probe) - e_perm.slice_manifestation(probe))) < 1e-12
    assert np.max(np.abs(o_after.slice_manifestation(probe) - o_perm.slice_manifestation(probe))) < 1e-12

    # Destructive pair (anti-phase pi) triggers continuous inhibitory damping
    anti_obs = _make_1d_test_manifold(
        global_w=(2.0 * math.pi * 3.0, 2.0 * math.pi * 7.0),
        global_amp=(-1.0 + 0.0j, -0.6 + 0.0j),
        local_w=(24.0,),
        local_amp=(-0.5 + 0.0j,),
        local_centers=(0.5,),
        local_sigmas=(0.06,),
    )
    e_inh, o_inh, inh_metrics = emitter.bidirectional_couple(anti_obs, coupling_gain=0.3, inhibition_gain=0.4)
    assert inh_metrics["destructive_overlap"] < 0.0
    assert inh_metrics["observer_damping_sum"] > 0.0
    assert e_inh.wave_resonance(e_inh)["energy_self"] < emitter.wave_resonance(emitter)["energy_self"]
    assert o_inh.wave_resonance(o_inh)["energy_self"] < anti_obs.wave_resonance(anti_obs)["energy_self"]


def test_causal_field_integration() -> None:
    x = np.linspace(0.0, 1.0, 512, endpoint=False)
    sig = np.cos(2.0 * math.pi * 4.0 * x) + 2.5 * np.exp(-0.5 * ((x - 0.5) / 0.02) ** 2)
    manifold = encode_waveform_to_scale_rotors(sig, max_macro_modes=6, max_levels=3, residual_tol=0.03)
    tuned = manifold.tune_balance(scale_tilt=0.4, contrast_tilt=0.3)

    cf = CausalField(dimensions=3)
    sync_info = tuned.sync_to_causal_field(cf, prefix="test_rotor")

    assert len(cf.voxels) == len(tuned.bands)
    assert len(cf.beams) >= len(tuned.bands) - 1
    assert sync_info["beams_linked"] == len(tuned.bands) - 1
    assert abs(sum(sync_info["chromatic_triad"]) - 1.0) < 1e-9

    # Mobility conservation: injected balance flux produced non-zero velocity on voxels
    vel_norms = [float(np.linalg.norm(v.velocity)) for v in cf.voxels.values()]
    assert all(vn > 0.0 for vn in vel_norms)

    # Ensure CausalField.step runs without error on the projected voxels & beams
    cf.step(dt=0.05)


def test_scale_invariance_wall_clock_vs_point_array() -> None:
    # Build two identical-complexity manifolds encoded from N=100 and N=1_000_000 grids
    # and verify in-manifold operators run in O(K) independent of N.
    n_small = 100
    n_large = 1_000_000

    x_small = np.linspace(0.0, 1.0, n_small, endpoint=False)
    x_large = np.linspace(0.0, 1.0, n_large, endpoint=False)

    f_small = 1.5 * np.cos(2.0 * math.pi * 3.0 * x_small) + 2.0 * np.exp(-0.5 * ((x_small - 0.6) / 0.025) ** 2)
    f_large = 1.5 * np.cos(2.0 * math.pi * 3.0 * x_large) + 2.0 * np.exp(-0.5 * ((x_large - 0.6) / 0.025) ** 2)

    m_small = encode_waveform_to_scale_rotors(f_small, max_macro_modes=8, max_levels=2, max_knots_per_level=6)
    m_large = encode_waveform_to_scale_rotors(f_large, max_macro_modes=8, max_levels=2, max_knots_per_level=6)
    other = m_large.evolve_spacetime(dt=0.5)

    def _time_rotor_ops(manifold: SpectralScaleRotorManifold, partner: SpectralScaleRotorManifold, reps: int = 30) -> float:
        t0 = time.perf_counter()
        for _ in range(reps):
            ev = manifold.evolve_spacetime(dt=100.0).tune_balance(scale_tilt=0.3, contrast_tilt=0.2)
            _ = ev.integrate_over_domain()
            _ = ev.wave_resonance(partner)
        return time.perf_counter() - t0

    # Warmup
    _ = _time_rotor_ops(m_small, other, reps=3)
    dt_small = _time_rotor_ops(m_small, other, reps=30)
    dt_large = _time_rotor_ops(m_large, other, reps=30)

    # 10,000x increase in point resolution must yield virtually flat O(K) wall time
    ratio = dt_large / max(dt_small, 1e-9)
    assert ratio < 5.0
    assert m_large.nbytes < f_large.nbytes * 0.01


def test_real_world_audio_and_image_ingestion() -> None:
    wav_path = Path(r"c:\Elysia\data\unified_creation_dawn.wav")
    img_path = Path(r"c:\Elysia\demos\real_world_assets\real_human_face_hd.jpg")
    assert wav_path.is_file()
    assert img_path.is_file()

    # 1. Real 1D audio waveform
    with wave.open(str(wav_path), "rb") as wf:
        raw_frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float64)
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels())[:, 0]
        framerate = float(wf.getframerate())

    audio_norm = audio / max(float(np.max(np.abs(audio))), 1.0)
    duration = len(audio_norm) / framerate
    m_audio = encode_waveform_to_scale_rotors(
        audio_norm,
        domain=(0.0, duration),
        max_macro_modes=32,
        max_levels=3,
        max_knots_per_level=12,
        residual_tol=0.10,
    )
    assert m_audio.total_modes < len(audio_norm) * 0.05
    t_grid = np.linspace(0.0, duration, len(audio_norm), endpoint=False)[:, None]
    recon_audio = m_audio.slice_manifestation(t_grid)
    audio_corr = float(np.corrcoef(audio_norm, recon_audio)[0, 1])
    assert audio_corr > 0.70

    # Pure O(K) operations work cleanly on real audio manifold
    audio_tuned = m_audio.tune_balance(scale_tilt=0.4, contrast_tilt=0.3)
    assert math.isfinite(audio_tuned.integrate_over_domain())

    # 2. Real 2D image surface (downsampled to 64x64 for fast unit testing)
    img = Image.open(img_path).convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    img_arr = np.asarray(img, dtype=np.float64) / 255.0
    m_img = encode_surface_to_scale_rotors(
        img_arr,
        domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        max_macro_modes=48,
        max_levels=3,
        max_knots_per_level=14,
        residual_tol=0.08,
    )
    assert m_img.total_modes < img_arr.size * 0.05

    ys = np.linspace(0.0, 1.0, 64, endpoint=False)
    xs = np.linspace(0.0, 1.0, 64, endpoint=False)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pts2 = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
    recon_img = m_img.slice_manifestation(pts2)
    img_corr = float(np.corrcoef(img_arr.reshape(-1), recon_img)[0, 1])
    assert img_corr > 0.75

    # Light/dark balance tilt modifies bright vs dark ratio smoothly in O(K)
    r_dark = m_img.tune_balance(contrast_tilt=-0.8).spectral_axis_profile()["bright_to_dark_ratio"]
    r_bright = m_img.tune_balance(contrast_tilt=+0.8).spectral_axis_profile()["bright_to_dark_ratio"]
    assert r_bright > r_dark


def test_analytic_differentiation_and_superposition() -> None:
    # 2D global rotor field: f(y, x) = Re(a0 * exp(i*(wy0*y + wx0*x)) + a1 * exp(i*(wy1*y + wx1*x)))
    wy0, wx0 = 2.0 * math.pi * 2.0, -2.0 * math.pi * 3.0
    wy1, wx1 = -2.0 * math.pi * 1.0, 2.0 * math.pi * 4.0
    a0, a1 = 1.1 + 0.4j, -0.7 + 0.9j

    m2 = SpectralScaleRotorManifold(
        dim=2,
        domain_bounds=([0.0, 0.0], [1.0, 1.0]),
        bands=[
            ScaleRotorBand(
                level=0,
                dim=2,
                wavevectors=np.array([[wy0, wx0], [wy1, wx1]]),
                temporal_omegas=np.array([4.0, 5.0]),
                amplitudes=np.array([a0, a1]),
                centers=np.zeros((2, 2)),
                sigmas=np.full(2, np.inf),
                is_global=True,
                parent_indices=np.array([-1, -1], dtype=np.int32),
                parent_couplings=np.array([1.0, 1.0]),
            )
        ],
    )
    gy, gx = np.meshgrid(
        np.linspace(0.0, 1.0, 24, endpoint=False),
        np.linspace(0.0, 1.0, 24, endpoint=False),
        indexing="ij",
    )
    pts = np.stack([gy.reshape(-1), gx.reshape(-1)], axis=1)

    df_dy_exact = np.real(
        1j * wy0 * a0 * np.exp(1j * (wy0 * pts[:, 0] + wx0 * pts[:, 1]))
        + 1j * wy1 * a1 * np.exp(1j * (wy1 * pts[:, 0] + wx1 * pts[:, 1]))
    )
    df_dx_exact = np.real(
        1j * wx0 * a0 * np.exp(1j * (wy0 * pts[:, 0] + wx0 * pts[:, 1]))
        + 1j * wx1 * a1 * np.exp(1j * (wy1 * pts[:, 0] + wx1 * pts[:, 1]))
    )

    assert np.max(np.abs(m2.differentiate(axis=0).slice_manifestation(pts) - df_dy_exact)) < 1e-11
    assert np.max(np.abs(m2.differentiate(axis=1).slice_manifestation(pts) - df_dx_exact)) < 1e-11

    # Superposition linearity & parent index offset preservation on 1D multi-level manifolds
    m_a = _make_1d_test_manifold(
        global_w=(3.0, 7.0), global_amp=(1.0 + 0.2j, 0.5 - 0.3j), local_w=(19.0,), local_amp=(0.6 + 0.1j,)
    )
    m_b = _make_1d_test_manifold(
        global_w=(5.0,), global_amp=(0.8 + 0.4j,), local_w=(29.0,), local_amp=(-0.4 + 0.5j,), local_centers=(0.38,)
    )
    m_c = _make_1d_test_manifold(global_w=(7.0, 11.0), global_amp=(0.9 - 0.1j, 0.4 + 0.6j))

    m_sum = m_a.superpose(m_b)
    # m_b's level-1 parent index (0) must shift by m_a's level-0 mode count (2) -> 2
    assert list(m_sum.bands[1].parent_indices) == [0, 2]

    probe_1d = np.linspace(0.0, 1.0, 200)[:, None]
    assert np.max(
        np.abs(m_sum.slice_manifestation(probe_1d) - (m_a.slice_manifestation(probe_1d) + m_b.slice_manifestation(probe_1d)))
    ) < 1e-12

    ov_sum = m_sum.wave_resonance(m_c)["raw_overlap"]
    ov_parts = m_a.wave_resonance(m_c)["raw_overlap"] + m_b.wave_resonance(m_c)["raw_overlap"]
    assert abs(ov_sum - ov_parts) < 1e-12

