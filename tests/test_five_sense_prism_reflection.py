r"""
Verification suite for ELYSIA Five-Sense White-Light Prism Rotor & Contrastive Reflection:
Tests multi-band 5-sense dispersion -> RGB-style Cl(3,0) Gestalt convergence,
recreation boundary testing (Can vs. Cannot self-diagnosis), contrastive correlation
expansion bridging missing sensory bands & cross-sense couplings, and persistence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.embodiment import (
    AgencyMode,
    FiveSensePrismCortex,
    ScaleRotorEmbodiedMind,
    ScaleRotorHierarchy,
    ScreenRetinaSensor,
    UniversalMotorActuator,
    batch_clifford_docking_affinity,
)


def test_five_sense_white_light_dispersion_and_rgb_combinatorial_synthesis():
    """Disperses distinct physical objects across 5 senses and converges into distinct Cl(3,0) rotors."""
    cortex = FiveSensePrismCortex()

    stocking = cortex.disperse_and_synthesize(
        "sheer_elastic_stocking",
        {
            "visual.sheerness": 0.82,
            "visual.weave_density": 0.72,
            "tactile.softness": 0.75,
            "tactile.elasticity": 0.88,
            "tactile.tear_stress": 0.65,
        },
    )
    steel = cortex.disperse_and_synthesize(
        "cold_rigid_steel",
        {
            "visual.gloss": 0.90,
            "auditory.sharpness": 0.85,
            "tactile.tension": 0.95,
            "tactile.temperature": 0.08,
            "tactile.softness": 0.02,
        },
    )
    silk = cortex.disperse_and_synthesize(
        "warm_sunlit_silk",
        {
            "visual.wavelength": 0.88,
            "visual.luminance": 0.92,
            "tactile.softness": 0.95,
            "tactile.temperature": 0.82,
        },
    )

    assert set(stocking.sense_bands.keys()) >= {"visual", "tactile"}
    assert "visual:sheerness__x__tactile:elasticity" in stocking.cross_couplings
    assert len(stocking.unified_rotor) == 8

    # Distinct White-Light Gestalts occupy clearly distinct Cl(3,0) orientations
    r_steel = np.asarray(steel.unified_rotor, dtype=np.float64)[None, :]
    r_silk = np.asarray(silk.unified_rotor, dtype=np.float64)
    aff, _tq = batch_clifford_docking_affinity(r_steel, r_silk)
    assert float(aff[0]) < 0.80


def test_recreation_contrasts_what_it_can_and_cannot_do_then_expands_correlations():
    """
    Reconstructs a compatible target (`can_recreate=True`), honestly detects inability on an
    uncoupled target (`can_recreate=False`), contrasts both to expand missing correlations,
    and unlocks successful re-creation on retry.
    """
    cortex = FiveSensePrismCortex()
    cortex.register_observables_prototype(
        "smooth_silk",
        {"visual.luminance": 0.80, "visual.gloss": 0.70, "tactile.softness": 0.90},
    )
    cortex.register_observables_prototype(
        "sheer_veil",
        {"visual.sheerness": 0.85, "visual.gloss": 0.50, "tactile.softness": 0.50},
    )
    cortex.register_observables_prototype(
        "elastic_band",
        {"tactile.elasticity": 0.90, "tactile.tension": 0.60},
    )

    # 1. Recreable composite using already-understood bands & cross-sense couplings
    ok_attempt = cortex.attempt_recreation(
        "sheer_silk_ribbon",
        {
            "visual.luminance": 0.42,
            "visual.gloss": 0.62,
            "visual.sheerness": 0.44,
            "tactile.softness": 0.72,
        },
        fidelity_threshold=0.82,
    )
    assert ok_attempt.can_recreate is True
    assert ok_attempt.fidelity >= 0.82
    assert ok_attempt.missing_bands == []
    assert ok_attempt.missing_couplings == []

    # 2. Target requiring unlearned cross-sense coupling + unseen tear/weave bands -> honestly cannot recreate yet
    fail_attempt = cortex.attempt_recreation(
        "stocking_tear",
        {
            "visual.sheerness": 0.80,
            "visual.weave_density": 0.70,
            "tactile.elasticity": 0.85,
            "tactile.tear_stress": 0.78,
        },
        fidelity_threshold=0.82,
    )
    assert fail_attempt.can_recreate is False
    assert "tactile:tear_stress" in fail_attempt.missing_bands
    assert "visual:weave_density" in fail_attempt.missing_bands
    assert any("sheerness__x__tactile" in ck for ck in fail_attempt.missing_couplings)

    # 3. Contrast successes vs failures -> extract missing correlations -> unlock re-creation
    report = cortex.contrast_and_expand_correlations(fidelity_threshold=0.82)
    assert "sheer_silk_ribbon" in report.recreable_ids
    assert "stocking_tear" in report.non_recreable_ids
    assert "tactile:tear_stress" in report.identified_missing_bands
    assert len(report.identified_missing_couplings) >= 2
    assert report.expanded_prototypes >= 1
    assert report.expanded_couplings >= 2
    assert report.after_avg_fidelity > report.before_avg_fidelity + 0.15
    assert "stocking_tear" in report.newly_unlocked_ids

    retry = cortex.attempt_recreation(
        "stocking_tear",
        {
            "visual.sheerness": 0.80,
            "visual.weave_density": 0.70,
            "tactile.elasticity": 0.85,
            "tactile.tear_stress": 0.78,
        },
        fidelity_threshold=0.82,
    )
    assert retry.can_recreate is True
    assert retry.fidelity >= 0.82

    # 4. Also generalizes combinatorially to a novel silk+stocking blend (RGB synthesis: 0.3*silk + 0.6*stocking)
    blend = cortex.attempt_recreation(
        "silky_elastic_blend",
        {
            "visual.luminance": 0.24,
            "visual.gloss": 0.21,
            "tactile.softness": 0.27,
            "visual.sheerness": 0.48,
            "visual.weave_density": 0.42,
            "tactile.elasticity": 0.51,
            "tactile.tear_stress": 0.468,
        },
        fidelity_threshold=0.82,
    )
    assert blend.can_recreate is True
    assert blend.fidelity >= 0.82


def test_prism_cortex_persists_inside_unified_hierarchy(tmp_path: Path):
    """FiveSensePrismCortex prototypes and expanded couplings persist inside `ScaleRotorHierarchy`."""
    uni = ScaleRotorHierarchy()
    uni.prism_cortex.register_observables_prototype(
        "warm_fabric",
        {"visual.wavelength": 0.85, "tactile.softness": 0.80, "tactile.temperature": 0.75},
    )
    save_path = tmp_path / "uni_prism.json"
    uni.save(save_path)

    reloaded = ScaleRotorHierarchy.load(save_path)
    topo = reloaded.topology_summary()
    assert topo["prism_prototypes"] == 1
    assert topo["prism_couplings"] >= 1
    assert "warm_fabric" in reloaded.prism_cortex.prototypes


def test_embodied_mind_drives_prism_and_spectrum_expansion_from_live_pixels(tmp_path: Path):
    """Real pixel arrays + foreground window titles drive live FiveSensePrism & SelfExpandingSpectrum expansion."""
    # Alternate genuinely distinct pixel frames: dark cool code pane -> bright warm sunlit canvas -> striped motion
    dark_code = np.tile(np.array([18, 26, 48], dtype=np.uint8), (24, 32, 1))
    bright_web = np.tile(np.array([245, 220, 185], dtype=np.uint8), (24, 32, 1))
    high_edge = np.zeros((24, 32, 3), dtype=np.uint8)
    high_edge[::2, ::2] = [250, 90, 40]
    high_edge[1::2, 1::2] = [20, 40, 210]

    frames = [dark_code, dark_code, bright_web, bright_web, high_edge, high_edge, dark_code]
    titles = [
        "Visual Studio Code - main.py",
        "Visual Studio Code - main.py",
        "YouTube - Browser",
        "YouTube - Browser",
        "Unity 6 Preview",
        "Unity 6 Preview",
        "Visual Studio Code - main.py",
    ]
    idx = {"i": 0}
    retina = ScreenRetinaSensor(frame_provider=lambda: frames[idx["i"]])
    actuator = UniversalMotorActuator(dry_run=True)

    mind = ScaleRotorEmbodiedMind(tmp_path / "live_prism_state")
    last_rec = {}
    for k in range(len(frames)):
        idx["i"] = k
        snap = retina.capture()
        last_rec = mind.perceive_and_act_tick(
            frame_snap=snap,
            user_idle_s=1.0,
            agency_mode=AgencyMode.OBSERVE_AND_CHAT,
            idle_threshold_s=15.0,
            actuator=actuator,
            foreground_window=titles[k],
        )

    assert last_rec["foreground_window"] == "Visual Studio Code - main.py"
    assert last_rec["prism_prototypes"] >= 1
    assert last_rec["prism_couplings"] >= 1
    assert last_rec["discovered_axes"] >= 1

    reloaded = ScaleRotorEmbodiedMind(tmp_path / "live_prism_state")
    assert len(reloaded.universe.prism_cortex.prototypes) == last_rec["prism_prototypes"]
    assert reloaded.universe.spectrum_cortex.discovered_axis_count() == last_rec["discovered_axes"]
