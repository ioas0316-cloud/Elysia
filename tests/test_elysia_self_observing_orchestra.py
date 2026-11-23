from pathlib import Path

import numpy as np

from core.orchestration.elysia_self_observing_orchestra import (
    ElysiaSelfObservingOrchestra,
)
from core.physics.causal_field import CausalField
from core.physics.spectral_scale_rotor_manifold import SpectralBalanceScale


def test_upper_self_rotor_past_present_future_continuity(tmp_path: Path) -> None:
    slab = tmp_path / "self_continuity.slab"
    with ElysiaSelfObservingOrchestra(slab_path=slab, slice_duration_s=0.4) as director:
        res = director.experience_and_direct(t_now=2.5, horizon=1.25, manifest=False)

        assert res.t_past < res.t_now < res.t_future
        assert res.self_rotor_angles[0] < res.self_rotor_angles[1] < res.self_rotor_angles[2]
        assert len(res.process_trace.palette_mechanisms) >= 3
        assert "vision" in res.process_trace.sensory_outcomes
        assert "hearing" in res.process_trace.sensory_outcomes
        assert "narrative" in res.process_trace.sensory_outcomes
        assert "touch" in res.process_trace.sensory_outcomes
        assert "과거" in res.self_reflection_ko and "현재" in res.self_reflection_ko and "미래" in res.self_reflection_ko
        assert res.four_axis.epistemic_boundary != ""


def test_polyphonic_composer_and_narrative_drama_coherence(tmp_path: Path) -> None:
    slab = tmp_path / "poly_drama.slab"
    with ElysiaSelfObservingOrchestra(slab_path=slab, slice_duration_s=0.35) as director:
        times = [0.0, 2.5, 5.0, 7.5]
        expected_chords = ["C_major", "F_lydian", "A_minor", "C_add9"]
        acts_seen = set()

        for t_val, exp_chord in zip(times, expected_chords):
            res = director.experience_and_direct(
                t_now=t_val, manifest=True, resolution=48, auto_revise_criteria=False
            )
            aud = res.present_slice.performances["hearing.polyphonic_composer"]
            drm = res.present_slice.performances["narrative.drama_director"]
            vis = res.present_slice.performances["vision.optical_prism"]

            assert aud.causal_state["chord_name"] == exp_chord
            assert aud.causal_state["voice_count"] == 3
            assert aud.manifestation is not None
            assert np.all(np.isfinite(aud.manifestation))
            assert float(np.std(aud.manifestation)) > 0.05
            assert vis.manifestation is not None and vis.manifestation.shape == (48, 48, 3)
            acts_seen.add(drm.causal_state["act_title"])

        assert len(acts_seen) == 4

        # Tilting contrast changes dramatic tension coherently across hearing & narrative
        calm = director.experience_and_direct(
            t_now=5.0,
            balance=SpectralBalanceScale(contrast_tilt=-0.6),
            auto_revise_criteria=False,
            manifest=False,
        )
        tense = director.experience_and_direct(
            t_now=5.0,
            balance=SpectralBalanceScale(contrast_tilt=+0.8),
            auto_revise_criteria=False,
            manifest=False,
        )
        t_calm = calm.present_slice.performances["narrative.drama_director"].causal_state["dramatic_tension"]
        t_tense = tense.present_slice.performances["narrative.drama_director"].causal_state["dramatic_tension"]
        assert t_tense > t_calm


def test_counterfactual_future_and_criteria_self_revision(tmp_path: Path) -> None:
    slab = tmp_path / "self_revision.slab"
    with ElysiaSelfObservingOrchestra(slab_path=slab, slice_duration_s=0.3) as director:
        extreme = SpectralBalanceScale(scale_tilt=0.95, contrast_tilt=0.95)
        res = director.experience_and_direct(
            t_now=5.0,
            horizon=1.25,
            balance=extreme,
            auto_revise_criteria=True,
            manifest=False,
        )
        assert res.four_axis.criteria_revised is True
        assert abs(res.four_axis.revised_balance.contrast_tilt) < abs(extreme.contrast_tilt)
        assert abs(res.four_axis.revised_balance.scale_tilt) < abs(extreme.scale_tilt)
        assert "스스로 저울을 조율" in res.self_reflection_ko


def test_ice_water_gas_memory_sedimentation_and_causal_field_self_sync(tmp_path: Path) -> None:
    slab = tmp_path / "self_cf_sync.slab"
    with ElysiaSelfObservingOrchestra(slab_path=slab, slice_duration_s=0.3, max_gas_bytes=32 * 1024) as director:
        res = director.experience_and_direct(t_now=2.5, horizon=1.25, manifest=False)
        assert director.hw_runtime.ice.num_crystals >= 2
        assert res.hardware_metrics["ssd_read_bytes_delta"] == 0
        assert res.hardware_metrics["gas_bytes"] <= 32 * 1024

        cf = CausalField(dimensions=3)
        sync = director.sync_to_causal_field(res, cf, prefix="self_test")
        assert len(sync["self_voxels"]) == 3
        # 3 self + 3 globe tiers + 4 instruments + 3 hw phases = 13 voxels
        assert sync["total_voxels"] == 13
        assert sync["total_beams"] >= 14
        cf.step(dt=0.05)
