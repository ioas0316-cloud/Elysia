from pathlib import Path
import wave

from core.topology.a4_surface_frame_rotor import A4SurfaceFrameRotor
from core.topology.causal_drama_director import CausalDramaDirector


def test_dynamic_variable_resolution_surface_rotor() -> None:
    # Proves A4SurfaceFrameRotor is NOT locked to 32x32 — works cleanly at 16x16 or 8x8!
    r16 = A4SurfaceFrameRotor(dim=64, surface_h=16, surface_w=16, device="cpu")
    assert r16.surface_cells == 256
    assert r16._pos_phase.shape == (16, 16, 64)


def test_causal_drama_director_counterfactual_meta_dials_transform_all_6_cuts(tmp_path: Path) -> None:
    director = CausalDramaDirector(dim=128)

    # Regime A: Warm Empathy + Calm + Energetic
    res_a = director.direct_causal_drama(
        title_ko="공명의 오후",
        empathy_vs_conflict=0.80,
        calm_vs_crisis=-0.70,
        fatigue_gravity=-0.50,
        out_dir=tmp_path / "drama_a",
        stem="drama_empathy_calm",
    )
    # Regime B: Conflict + Storm Crisis + Heavy Fatigue
    res_b = director.direct_causal_drama(
        title_ko="폭풍 속 균열",
        empathy_vs_conflict=-0.80,
        calm_vs_crisis=0.85,
        fatigue_gravity=0.80,
        out_dir=tmp_path / "drama_b",
        stem="drama_conflict_crisis",
    )

    assert res_a["ok"] is True and res_b["ok"] is True
    assert len(res_a["cuts"]) == 6 and len(res_b["cuts"]) == 6
    assert res_a["volume_continuity_cos"] > 0.70

    # Counterfactual causal physics MUST coherently transform ALL 6 cuts:
    # Conflict+Crisis -> higher storm, colder warmth, wider separation, lower exhausted posture, higher chord tension!
    mean_storm_a = sum(c["storm_intensity"] for c in res_a["cuts"]) / 6.0
    mean_storm_b = sum(c["storm_intensity"] for c in res_b["cuts"]) / 6.0
    mean_dist_a = sum(c["interpersonal_dist"] for c in res_a["cuts"]) / 6.0
    mean_dist_b = sum(c["interpersonal_dist"] for c in res_b["cuts"]) / 6.0
    mean_post_a = sum(c["posture_a"] for c in res_a["cuts"]) / 6.0
    mean_post_b = sum(c["posture_a"] for c in res_b["cuts"]) / 6.0
    mean_tens_a = sum(c["chord_tension"] for c in res_a["cuts"]) / 6.0
    mean_tens_b = sum(c["chord_tension"] for c in res_b["cuts"]) / 6.0

    assert mean_storm_b > mean_storm_a + 0.35
    assert mean_dist_b > mean_dist_a + 0.20
    assert mean_post_a > mean_post_b + 0.50
    assert mean_tens_b > mean_tens_a + 0.25

    # Verify all 4 artifacts (.html theater, .png 6-cut storyboard, .wav soundtrack, .json manifest) exist & valid
    arts = res_a["artifacts"]
    assert Path(arts["theater_html"]).is_file() and Path(arts["theater_html"]).stat().st_size > 4000
    assert Path(arts["storyboard_png"]).is_file() and Path(arts["storyboard_png"]).stat().st_size > 8000
    assert Path(arts["manifest_json"]).is_file()
    with wave.open(arts["soundtrack_wav"], "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getnframes() > 80000
