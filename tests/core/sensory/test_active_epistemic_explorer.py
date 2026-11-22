"""
test_active_epistemic_explorer.py

End-to-end pytest verification of MechanoAcousticOrgan, HiddenMultiPhysicsWorld,
Real4KYosemiteWorld, EpistemicAttentionField, blank-slate CrossModalCorrelationGraph,
Prototype Mitosis on deceptive counter-examples, and motor-visual shadow resolution.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.sensory.active_epistemic_explorer import ActiveEpistemicExplorer
from core.sensory.mechano_acoustic_organ import MechanoAcousticOrgan
from simulators.hidden_multiphysics_world import (
    HiddenMultiPhysicsWorld,
    Real4KYosemiteWorld,
    YOSEMITE_4K_PATH,
)


def _make_force_disp_curve(
    k: float,
    damping: float,
    force_max: float = 15.0,
    yield_force: float = 100.0,
    plastic_soft: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    n_half = 40
    t_up = np.linspace(0.0, 1.0, n_half)
    t_dn = np.linspace(1.0, 0.0, n_half)
    force_t = np.concatenate([t_up, t_dn]) * force_max

    disp = []
    plastic = 0.0
    max_f = 0.0
    for idx, f_val in enumerate(force_t):
        is_load = idx < n_half
        if f_val <= yield_force:
            x_el = f_val / k
        else:
            excess = f_val - yield_force
            x_el = (yield_force / k) + (excess / (0.14 * k))
            if is_load and f_val > max_f:
                plastic = max(plastic, plastic_soft * (excess / (0.14 * k)))
        max_f = max(max_f, f_val)
        phase_frac = (idx / float(n_half)) if is_load else ((2 * n_half - 1 - idx) / float(n_half))
        hyst = (0.5 * damping * (force_max / k) * phase_frac) if not is_load else (-0.2 * damping * (force_max / k) * phase_frac)
        if not is_load:
            x_val = (f_val / k) + plastic + max(0.0, hyst)
        else:
            x_val = max(0.0, x_el + hyst)
        disp.append(x_val)
    return force_t.astype(np.float32), np.maximum(0.0, np.asarray(disp, dtype=np.float32))


def _make_tap_waveform(freq_hz: float, decay_s: float, q_hint: float) -> np.ndarray:
    sr = 16000
    n = 4096
    t = np.arange(n, dtype=np.float64) / float(sr)
    rng = np.random.default_rng(7)
    sig = 0.8 * np.exp(-decay_s * t) * np.sin(2.0 * np.pi * freq_hz * t)
    sig += np.exp(-150.0 * t) * rng.normal(0.0, 0.25 / max(1.0, q_hint), size=n)
    return sig.astype(np.float32)


def test_mechano_acoustic_organs_separate_rigid_compliant_plastic():
    organ = MechanoAcousticOrgan()

    f_r, x_r = _make_force_disp_curve(k=12000.0, damping=0.08, force_max=15.0, yield_force=80.0)
    f_c, x_c = _make_force_disp_curve(k=650.0, damping=0.65, force_max=15.0, yield_force=40.0)
    f_p, x_p = _make_force_disp_curve(k=500.0, damping=0.70, force_max=15.0, yield_force=8.0, plastic_soft=0.45)

    m_r = organ.observe_push(f_r, x_r)
    m_c = organ.observe_push(f_c, x_c)
    m_p = organ.observe_push(f_p, x_p)

    assert 10000.0 <= m_r.effective_stiffness_n_m <= 14500.0
    assert not m_r.yield_detected
    assert m_r.mechanical_regime == "ELASTIC_RIGID"

    assert 500.0 <= m_c.effective_stiffness_n_m <= 900.0
    assert not m_c.yield_detected
    assert m_c.mechanical_regime == "COMPLIANT_SOFT"

    assert m_p.yield_detected
    assert m_p.residual_plastic_ratio > 0.25
    assert m_p.mechanical_regime == "PLASTIC_YIELDING"

    w_ring = _make_tap_waveform(freq_hz=2400.0, decay_s=18.0, q_hint=14.0)
    w_thud = _make_tap_waveform(freq_hz=320.0, decay_s=95.0, q_hint=2.0)
    a_ring = organ.observe_tap(w_ring)
    a_thud = organ.observe_tap(w_thud)

    assert abs(a_ring.peak_frequency_hz - 2400.0) < 120.0
    assert a_ring.acoustic_regime == "SHARP_HIGH_RING"
    assert abs(a_thud.peak_frequency_hz - 320.0) < 60.0
    assert a_thud.acoustic_regime == "DAMPED_LOW_THUD"


def test_real_4k_yosemite_autonomous_foveal_exploration():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = Real4KYosemiteWorld(YOSEMITE_4K_PATH)
    explorer = ActiveEpistemicExplorer()
    trace = explorer.explore(world, num_steps=6)

    assert trace.steps == 6
    assert all(d.chosen.recommended_action == "FOVEAL_FIXATE" for d in trace.decisions)
    distinct_locs = {(round(d.chosen.y_norm, 2), round(d.chosen.x_norm, 2)) for d in trace.decisions}
    assert len(distinct_locs) >= 4
    assert len(trace.prototypes) == 0  # Strictly optical world -> no cross-modal probes fabricated


def test_blank_slate_cross_modal_correlation_discovery_and_generalization():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = HiddenMultiPhysicsWorld(YOSEMITE_4K_PATH, seed=42)
    explorer = ActiveEpistemicExplorer()
    trace = explorer.explore(world, num_steps=16)

    # Discovered multiple distinct cross-modal prototypes from completely blank slate
    assert len(trace.prototypes) >= 3

    # Verify cross-modal generalization on confirmed prototype (compliant_canopy_C & D share same prototype!)
    confirmed_protos = [p for p in trace.prototypes if p.confirmation_count >= 2]
    assert len(confirmed_protos) >= 1

    confirmed_followups = [
        ep
        for ep in trace.episodes
        if ep.action_type == "MECHANICAL_AND_ACOUSTIC_PROBE"
        and ep.predicted_mech_vec is not None
        and ep.prediction_surprise < 0.50
    ]
    assert len(confirmed_followups) >= 2


def test_deceptive_counterexample_triggers_surprise_attention_capture_and_mitosis():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = HiddenMultiPhysicsWorld(YOSEMITE_4K_PATH, seed=42)
    explorer = ActiveEpistemicExplorer()
    trace = explorer.explore(world, num_steps=16)

    assert len(trace.refinements) >= 1
    ref = trace.refinements[0]
    assert ref.surprise_error > 0.75

    # Immediately following step must capture attention onto the exact surprising region for foveation!
    idx = ref.step - 1
    assert idx + 1 < len(trace.decisions)
    surprise_decision = trace.decisions[idx]
    capture_decision = trace.decisions[idx + 1]
    assert capture_decision.chosen.target_id == surprise_decision.chosen.target_id
    assert capture_decision.chosen.surprise_score > 0.0
    assert capture_decision.chosen.recommended_action == "FOVEAL_FIXATE"

    # Both parent and split child prototypes acquire foveal microstructure separating deceptive foam vs rigid granite
    proto_map = {p.prototype_id: p for p in trace.prototypes}
    parent_p = proto_map[ref.parent_prototype_id]
    child_p = proto_map[ref.new_prototype_id]
    assert parent_p.visual_foveal_center is not None
    assert child_p.visual_foveal_center is not None
    assert abs(float(parent_p.visual_foveal_center[0]) - float(child_p.visual_foveal_center[0])) > 0.15


def test_motor_visual_push_resolves_shadowed_rigid_slab():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = HiddenMultiPhysicsWorld(YOSEMITE_4K_PATH, seed=42)
    explorer = ActiveEpistemicExplorer()
    trace = explorer.explore(world, num_steps=16)

    assert len(trace.resolved_shadow_boundaries) >= 1
    assert any(
        verdict == "CONFIRMED_RIGID_UNIFIED_BODY_UNDER_SHADOW"
        for verdict in trace.resolved_shadow_boundaries.values()
    )

    # Check CausalField preserved a strong unified beam across every resolved rigid shadow pair
    for (ra, rb), verdict in trace.resolved_shadow_boundaries.items():
        if verdict != "CONFIRMED_RIGID_UNIFIED_BODY_UNDER_SHADOW":
            continue
        pair_names = {(f"region_{ra}", f"region_{rb}"), (f"region_{rb}", f"region_{ra}")}
        matching_beams = [
            b for b in trace.final_causal_field.beams if (b.source_id, b.target_id) in pair_names
        ]
        assert len(matching_beams) >= 1
        assert all(b.strength >= 2.5 and not b.is_broken for b in matching_beams)


def test_empirical_correlation_matrix_emerges_from_experience():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = HiddenMultiPhysicsWorld(YOSEMITE_4K_PATH, seed=42)
    explorer = ActiveEpistemicExplorer()
    trace = explorer.explore(world, num_steps=16)

    cm = trace.correlation_matrix
    assert cm["sample_count"] >= 5
    assert cm["matrix"] is not None
    assert cm["stiffness_vs_freq_r"] > 0.65


def test_existing_ocular_tests_still_pass_and_causal_field_finite():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = HiddenMultiPhysicsWorld(YOSEMITE_4K_PATH, seed=42)
    explorer = ActiveEpistemicExplorer()
    trace = explorer.explore(world, num_steps=16)

    field = trace.final_causal_field
    proto_voxels = [v for k, v in field.voxels.items() if k.startswith("proto_")]
    region_voxels = [v for k, v in field.voxels.items() if k.startswith("region_")]
    assert len(proto_voxels) == len(trace.prototypes)
    assert len(region_voxels) >= 4
    assert all(v.scale_level == 8.0 for v in proto_voxels)

    cross_modal_beams = [
        b
        for b in field.beams
        if b.source_id.startswith("region_") and b.target_id.startswith("proto_")
    ]
    assert len(cross_modal_beams) >= 4

    for v in field.voxels.values():
        assert np.all(np.isfinite(v.position))
        assert np.all(np.isfinite(v.velocity))
        assert np.all(np.isfinite(v.tensor))
