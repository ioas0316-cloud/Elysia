r"""
Verification suite for ELYSIA Cl(3,0) Geometric Rotor & Protein-Docking Self-Assembly:
Tests exact geometric algebra ascent (Point -> Line Bivector -> Volume -> Spacetime Rotor),
exact reversible rotor sandwich transformations, innate sensory-spectral coupling without
cosine dot-product similarity or emotion dictionaries, protein-like autonomous clustering,
and persistence / legacy state migration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.embodiment import (
    ScaleRotorHierarchy,
    ScaleRotorNode,
    batch_clifford_docking_affinity,
    clifford_geometric_product,
    clifford_reverse,
    compose_dimensional_rotor,
    extract_unit_rotor,
    rotor_sandwich,
    sensory_spectrum_to_point,
    wedge_points_to_line,
    wedge_three_to_volume,
)


def test_clifford_ascent_point_line_volume_and_reversible_rotor_sandwich():
    """Point x Point -> Line bivector, 3 points -> Volume, and reversible R X R~."""
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])

    # Anti-symmetric Point x Point -> Line bivector
    b12 = wedge_points_to_line(e1, e2)
    b21 = wedge_points_to_line(e2, e1)
    assert np.allclose(b12, [1.0, 0.0, 0.0])
    assert np.allclose(b21, -b12)

    # Line x Point -> Oriented Volume (Grade-3 pseudoscalar)
    assert np.isclose(wedge_three_to_volume(e1, e2, e3), 1.0)
    assert np.isclose(wedge_three_to_volume(e2, e1, e3), -1.0)

    # Ascend non-collinear child points into a multi-grade Cl(3,0) rotor
    mvs = [
        sensory_spectrum_to_point([0.9, 0.3, 0.2, 0.8], warm_cool_ratio=1.8, motion_flux=0.04),
        sensory_spectrum_to_point([0.5, 0.5, 0.6, 0.2], warm_cool_ratio=1.1, motion_flux=0.35),
        sensory_spectrum_to_point([0.2, 0.7, 0.3, 0.7], warm_cool_ratio=0.5, motion_flux=0.05),
    ]
    rotor_mv = compose_dimensional_rotor(mvs, scale=4)
    assert rotor_mv.shape == (8,)
    assert np.isclose(np.linalg.norm(rotor_mv), 1.0, atol=1e-6)
    assert np.linalg.norm(rotor_mv[4:7]) > 1e-3  # non-zero emergent Line bivector

    # Check exact reversible sandwich via both Cl(3,0) product & CliffordMultivector
    probe_pt = np.array([0.0, 0.7, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    transformed_mv = rotor_sandwich(rotor_mv, probe_pt)
    unit_r = extract_unit_rotor(rotor_mv)
    transformed_vec = unit_r.transform_vector(probe_pt[1:4])
    assert np.allclose(transformed_mv[1:4], transformed_vec, atol=1e-8)

    restored_vec = unit_r.reverse_transform_vector(transformed_vec)
    assert np.allclose(restored_vec, probe_pt[1:4], atol=1e-8)

    # Reversing an even-odd chain length of identical-polarity children stays locked
    same_pt = sensory_spectrum_to_point([0.85, 0.2, 0.2, 0.75], warm_cool_ratio=1.7)
    r_even = compose_dimensional_rotor([same_pt, same_pt], scale=3)
    r_odd = compose_dimensional_rotor([same_pt, same_pt, same_pt], scale=3)
    aff_eo, tq_eo = batch_clifford_docking_affinity(r_even[None, :], r_odd)
    assert float(aff_eo[0]) > 0.90
    assert float(tq_eo[0]) < 0.20


def test_innate_sensory_spectral_docking_without_emotion_dictionaries():
    """Warm long-wave vs cool short-wave phenomena separate/align geometrically from physics alone."""
    warm_vis = sensory_spectrum_to_point(
        [0.82, 0.40, 0.30, 0.75],
        warm_cool_ratio=1.85,
        brightness=0.80,
        motion_flux=0.06,
    )
    warm_lang = sensory_spectrum_to_point([0.80, 0.45, 0.25, 0.80])
    cool_storm = sensory_spectrum_to_point(
        [0.15, 0.35, 0.85, -0.75],
        warm_cool_ratio=0.35,
        brightness=0.20,
        motion_flux=0.45,
        contrast_std=0.32,
    )

    # Reverse product of warm visuals & warm language exhibits high Clifford phase-lock & low torque
    rev_check = clifford_geometric_product(warm_vis, clifford_reverse(warm_lang))
    assert float(rev_check[0]) > 0.70

    affs, torques = batch_clifford_docking_affinity(
        np.stack([warm_lang, cool_storm], axis=0),
        warm_vis,
    )
    assert float(affs[0]) >= 0.72
    assert float(torques[0]) <= 0.45
    assert float(affs[1]) < 0.0
    assert float(torques[1]) > float(torques[0])


def test_protein_like_self_assembly_emerges_distinct_clusters_automatically():
    """Feeding distinct physical episodes self-assembles `docks` bonds & distinct `cluster::*` nodes."""
    uni = ScaleRotorHierarchy()

    # 1. Warm luminous sunset episode -> nucleates cluster::c_1
    warm_spec = [0.82, 0.55, 0.25, 0.80]
    for i, phrase in enumerate(("햇살이 따스하다.", "노을이 포근하다.")):
        vw = uni.differentiate_visual(
            [0.92, 0.45, 0.18] * 16,
            motion_flux=0.03,
            contrast_std=0.10,
            warm_cool_ratio=1.85,
            spectrum=warm_spec,
        )
        uw = uni.differentiate_text(phrase, spectrum=warm_spec)
        aw = uni.differentiate_audio({"rms_energy": 0.06, "zero_cross_rate": 0.05}, spectrum=warm_spec)
        uni.integrate_lived_scene(f"warm_scene_{i}", vw, aw, uw, "observe_attend", warm_spec, visual_change=0.02)

    clusters_after_warm = [nid for nid in uni.nodes if nid.startswith("cluster::")]
    assert len(clusters_after_warm) == 1

    # 2. Additional matching warm input reinforces/docks into existing cluster rather than duplicating
    vw2 = uni.differentiate_visual(
        [0.90, 0.50, 0.20] * 16,
        motion_flux=0.04,
        contrast_std=0.09,
        warm_cool_ratio=1.80,
        spectrum=[0.80, 0.55, 0.28, 0.78],
    )
    uw2 = uni.differentiate_text("빛이 온화하다.", spectrum=[0.80, 0.55, 0.28, 0.78])
    uni.integrate_lived_scene("warm_scene_extra", vw2, None, uw2, "observe_attend", [0.80, 0.55, 0.28, 0.78])
    assert len([nid for nid in uni.nodes if nid.startswith("cluster::")]) == 1

    # 3. Qualitatively opposite dark/tense/turbulent episode -> automatically nucleates a 2nd cluster
    cool_spec = [0.14, 0.28, 0.88, -0.78]
    for i, phrase in enumerate(("밤바다가 어둡다.", "바람이 매섭다.")):
        vc = uni.differentiate_visual(
            [0.08, 0.14, 0.75] * 16,
            motion_flux=0.45,
            contrast_std=0.30,
            warm_cool_ratio=0.35,
            spectrum=cool_spec,
        )
        uc = uni.differentiate_text(phrase, spectrum=cool_spec)
        ac = uni.differentiate_audio({"rms_energy": 0.28, "zero_cross_rate": 0.25}, spectrum=cool_spec)
        uni.integrate_lived_scene(f"cool_scene_{i}", vc, ac, uc, "click_salient", cool_spec, visual_change=0.35)

    topo = uni.topology_summary()
    assert topo["docked_links"] >= 4
    assert topo["emergent_clusters"] >= 2

    # Warm and cool clusters reside on opposite sides of physical e1 / e3 axes
    c_nodes = [nd for nid, nd in sorted(uni.nodes.items()) if nid.startswith("cluster::")]
    c_aff, _c_tq = batch_clifford_docking_affinity(
        np.asarray(c_nodes[1].multivector, dtype=np.float64)[None, :],
        np.asarray(c_nodes[0].multivector, dtype=np.float64),
    )
    assert float(c_aff[0]) < 0.35


def test_multivector_persistence_roundtrip_and_legacy_node_migration(tmp_path: Path):
    """Persists 8-D Cl(3,0) multivectors + docks + clusters and migrates legacy dicts cleanly."""
    legacy_node = ScaleRotorNode.from_dict(
        {
            "node_id": "unit::온기",
            "scale": 2,
            "surface": "온기",
            "spectrum": [0.85, 0.45, 0.20, 0.75],
            "signature_vec": [0.1] * 32,
            "mass": 2.0,
        }
    )
    assert len(legacy_node.multivector) == 8
    assert legacy_node.multivector[1] > 0.2  # Migrated cleanly onto +e1 warm wavelength pole

    uni = ScaleRotorHierarchy()
    warm_spec = [0.82, 0.55, 0.25, 0.80]
    vw = uni.differentiate_visual([0.9, 0.4, 0.2] * 16, 0.04, 0.12, 1.80, warm_spec)
    uw = uni.differentiate_text("봄빛이 따뜻하다.", spectrum=warm_spec)
    uni.integrate_lived_scene("persist_ep", vw, None, uw, "observe_attend", warm_spec)

    save_path = tmp_path / "scale_rotor_universe.json"
    uni.save(save_path)
    reloaded = ScaleRotorHierarchy.load(save_path)

    assert reloaded.topology_summary() == uni.topology_summary()
    for nid, nd in uni.nodes.items():
        assert np.allclose(reloaded.nodes[nid].multivector, nd.multivector, atol=1e-4)
