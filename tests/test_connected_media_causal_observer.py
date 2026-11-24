"""
Verification suite for ELYSIA v3.2:
Generative O(K) Scale-Rotor Connected Media Observer, Recreator & Self-Improver.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.orchestration.connected_media_causal_observer import (
    AgencyMode,
    ConnectedMediaCausalObserver,
    NarrativeCausalityMode,
    build_audiovisual_narrative_scene,
    run_pixel_rendered_interactive_game,
)
from core.orchestration.elysia_unified_living_cycle import ElysiaUnifiedLivingCycle
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.physics.causal_field import CausalField

_REAL_MP4 = Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")
_REAL_GAME_GIF = Path(r"c:\Elysia\docs\assets\causal_game_trajectory_playback.gif")
_REAL_SPEECH_WAV = Path(
    r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data_archive\S1_Body\L1_Foundation\external\CosyVoice\asset\cross_lingual_prompt.wav"
)
_REAL_SYMPHONY_WAV = Path(r"c:\Elysia\data\elysia_drama_symphony.wav")


def test_real_external_mp4_video_and_game_gif_pixel_flow_perception():
    """Watches real MP4 video and real game GIF via vectorized Fourier Phase-Shift Scale-Rotors."""
    observer = ConnectedMediaCausalObserver()

    rep_mp4 = observer.observe_episode(
        episode_id="real_nature_mp4",
        description="실제 외부 자연 동영상(MP4)",
        video_source=_REAL_MP4,
        max_frames=30,
    )
    assert rep_mp4.visual_trace is not None
    assert rep_mp4.visual_trace.scale_rotor_spectrum is not None
    assert rep_mp4.visual_trace.frame_count >= 20
    assert float(rep_mp4.visual_trace.motion_energy.mean()) > 0.10
    assert rep_mp4.present_modalities == ["visual"]
    assert set(rep_mp4.absent_modalities) == {"audio", "language", "control"}
    assert rep_mp4.causality_mode == NarrativeCausalityMode.UNIMODAL_PHYSICAL_FLOW
    assert len(rep_mp4.beats) >= 3

    rep_gif = observer.observe_episode(
        episode_id="real_game_gif",
        description="실제 게임 궤적 GIF",
        video_source=_REAL_GAME_GIF,
        max_frames=39,
    )
    assert rep_gif.visual_trace is not None
    assert rep_gif.visual_trace.frame_count >= 25
    assert float(rep_gif.visual_trace.active_object_speed.max()) > 0.25
    assert float(rep_mp4.visual_trace.active_motion_area.mean()) > float(
        rep_gif.visual_trace.active_motion_area.mean()
    )


def test_real_human_speech_wav_vs_continuous_symphony_emergent_discernment():
    """Spontaneously distinguishes real human speech phrasing from continuous symphony via FFT Wiener-Khinchin."""
    observer = ConnectedMediaCausalObserver()

    rep_speech = observer.observe_episode(
        episode_id="real_human_speech",
        description="실제 사람 음성 녹음(WAV)",
        audio_source=_REAL_SPEECH_WAV,
    )
    rep_music = observer.observe_episode(
        episode_id="real_drama_symphony",
        description="연속 교향곡 음향(WAV)",
        audio_source=_REAL_SYMPHONY_WAV,
    )

    assert rep_speech.audio_trace is not None
    assert rep_speech.audio_trace.scale_rotor_spectrum is not None
    assert rep_music.audio_trace is not None
    assert rep_speech.audio_trace.acoustic_character == "speech_like_phrasing"
    assert rep_speech.audio_trace.pause_fraction >= 0.12
    assert rep_speech.audio_trace.phrase_count >= 3
    assert rep_speech.audio_trace.voiced_fraction >= 0.40

    assert rep_music.audio_trace.acoustic_character == "continuous_harmonic"
    assert rep_music.audio_trace.pause_fraction < 0.10
    # Vectorized FFT audio perception must be fast (< 250ms for 13.75s speech)
    assert rep_speech.audio_trace.elapsed_ms < 250.0


def test_interactive_game_vs_passive_video_self_agency_discernment():
    """Same blind observer distinguishes playable game from recorded video and distills causal operator H."""
    observer = ConnectedMediaCausalObserver()

    rep_game = observer.observe_bundle(run_pixel_rendered_interactive_game(mode="interactive_play"))
    rep_video = observer.observe_bundle(run_pixel_rendered_interactive_game(mode="passive_video_playback"))

    assert rep_game.agency_mode == AgencyMode.SELF_AGENCY_INTERACTIVE
    assert rep_game.causality_mode == NarrativeCausalityMode.INTERACTIVE_CONTROL_DRIVEN
    ctrl_links = [lk for lk in rep_game.causal_links if lk.relation_kind == "self_agency_control_to_motion"]
    assert len(ctrl_links) == 1
    assert ctrl_links[0].peak_correlation >= 0.38
    assert ctrl_links[0].transfer_operator is not None
    assert ctrl_links[0].transfer_operator.train_r2 > 0.35

    assert rep_video.agency_mode == AgencyMode.EXTERNAL_AGENCY_OBSERVED
    assert not any(lk.relation_kind == "self_agency_control_to_motion" for lk in rep_video.causal_links)


def test_audiovisual_physical_synchrony_verified_and_falsified_by_desync_counterfactual():
    """True synchronized impacts resonate across video & audio; shifted soundtrack breaks sync."""
    observer = ConnectedMediaCausalObserver()

    rep_sync = observer.observe_bundle(run_pixel_rendered_interactive_game(mode="interactive_play"))
    rep_desync = observer.observe_bundle(run_pixel_rendered_interactive_game(mode="desynchronized_audio"))

    sync_av = [lk for lk in rep_sync.causal_links if lk.relation_kind == "audiovisual_physical_synchrony"]
    desync_av = [lk for lk in rep_desync.causal_links if lk.relation_kind == "audiovisual_physical_synchrony"]

    assert len(sync_av) == 1
    assert sync_av[0].peak_correlation >= 0.35
    if desync_av:
        assert sync_av[0].peak_correlation > desync_av[0].peak_correlation + 0.12


def test_vocabulary_free_volitional_vs_reactive_narrative_and_cut_bridges():
    """Distinguishes language->motion vs motion->language even on completely unknown alien words + detects cut bridges."""
    observer = ConnectedMediaCausalObserver()

    # Test both Korean AND completely unknown/unseen vocabulary ("Zorblax venn-ta!" / "Quorath! Vex?")
    for unknown_lang in (False, True):
        rep_vol = observer.observe_bundle(
            build_audiovisual_narrative_scene("volitional", use_unknown_language=unknown_lang)
        )
        rep_rea = observer.observe_bundle(
            build_audiovisual_narrative_scene("reactive", use_unknown_language=unknown_lang)
        )

        assert rep_vol.causality_mode == NarrativeCausalityMode.VOLITIONAL_LANGUAGE_LEADS_MOTION
        assert any(lk.relation_kind == "volitional_language_to_motion" and lk.best_lag_steps > 0 for lk in rep_vol.causal_links)

        assert rep_rea.causality_mode == NarrativeCausalityMode.REACTIVE_MOTION_LEADS_LANGUAGE
        assert any(lk.relation_kind == "reactive_motion_to_language" and lk.best_lag_steps > 0 for lk in rep_rea.causal_links)

        assert len(rep_vol.continuity_bridges) >= 1
        assert any(br.audio_continuity_preserved or br.language_continuity_preserved for br in rep_vol.continuity_bridges)
        assert len(rep_rea.continuity_bridges) >= 1
        assert any(br.audio_continuity_preserved or br.language_continuity_preserved for br in rep_rea.continuity_bridges)


def test_closed_loop_recreation_and_autonomous_self_improvement_on_game_and_real_media():
    """
    Core Generative Principle Proof:
    Elysia recreates real RGB frames & PCM audio strictly from her Scale-Rotor Spectrum,
    re-observes her own creation, harvests residuals, and strictly self-improves across cycles
    on BOTH interactive gameplay AND real disk MP4 + real human speech WAV!
    """
    observer = ConnectedMediaCausalObserver()

    # 1. Interactive audiovisual game closed-loop self-improvement
    bundle = run_pixel_rendered_interactive_game(mode="interactive_play")
    imp_game = observer.learn_recreate_and_self_improve(
        episode_id="self_improve_game",
        video_source=bundle.frames,
        video_fps=bundle.fps,
        audio_source=(bundle.audio_pcm, bundle.sample_rate),
        max_frames=len(bundle.frames),
        max_cycles=3,
    )
    assert imp_game.strictly_improved is True
    assert len(imp_game.cycles) == 3
    assert imp_game.cycles[0].visual_mse > imp_game.cycles[1].visual_mse > imp_game.cycles[2].visual_mse
    assert imp_game.cycles[0].audio_mse > imp_game.cycles[1].audio_mse > imp_game.cycles[2].audio_mse
    assert imp_game.visual_error_drop_ratio > 0.65
    assert imp_game.audio_error_drop_ratio > 0.90
    assert imp_game.cycles[-1].motion_flow_correlation > 0.70
    assert imp_game.cycles[-1].audio_envelope_correlation > 0.95
    assert imp_game.recreated_frames is not None and imp_game.recreated_frames.shape == bundle.frames.shape
    assert imp_game.recreated_pcm is not None and len(imp_game.recreated_pcm) == len(bundle.audio_pcm)
    assert imp_game.cycles[-1].visual_compression_ratio > 5.0

    # 2. Real external MP4 video + Real human speech WAV closed-loop self-improvement
    imp_real = observer.learn_recreate_and_self_improve(
        episode_id="self_improve_real_mp4_and_speech",
        video_source=_REAL_MP4,
        audio_source=_REAL_SPEECH_WAV,
        max_frames=24,
        max_cycles=3,
    )
    assert imp_real.strictly_improved is True
    assert imp_real.cycles[0].visual_mse > imp_real.cycles[-1].visual_mse
    assert imp_real.cycles[0].audio_mse > imp_real.cycles[-1].audio_mse
    assert imp_real.cycles[-1].audio_envelope_correlation > 0.90


def test_novel_counterfactual_world_creation_from_learned_causal_principles():
    """
    Creates a brand-new RGB pixel video + synchronized PCM audio from a NEVER-SEEN
    control rhythm purely via learned Causal Transfer Operators H + Scale-Rotor ground,
    and verifies the created world preserves self-agency & audiovisual synchrony!
    """
    observer = ConnectedMediaCausalObserver()
    origin = observer.observe_bundle(run_pixel_rendered_interactive_game(mode="interactive_play"))

    # Brand-new unseen control rhythm (different timing & pulse count from origin!)
    novel_ctrl = np.zeros(40, dtype=np.float64)
    novel_ctrl[[3, 4, 13, 14, 24, 25, 33, 34]] = 1.0

    created = observer.create_novel_world_from_principles(origin, novel_ctrl)
    assert created.created_frames.shape == (40, 80, 128, 3)
    assert len(created.created_pcm) > 10000
    assert created.agency_preserved is True
    assert created.synchrony_preserved is True
    # Must generate + self-verify in fast O(K) milliseconds (< 500ms)
    assert created.elapsed_ms < 500.0


def test_cross_media_narrative_isomorphism_without_modality_flattening():
    """Compares narrative structure across interactive game, audiovisual story, and silent MP4 without collapsing them."""
    observer = ConnectedMediaCausalObserver()

    rep_game = observer.observe_bundle(run_pixel_rendered_interactive_game(mode="interactive_play"))
    rep_story = observer.observe_bundle(build_audiovisual_narrative_scene("volitional"))
    rep_mp4 = observer.observe_episode(episode_id="nature_mp4", video_source=_REAL_MP4, max_frames=24)

    cmp_game_story = observer.compare_causal_narratives(rep_game, rep_story)
    assert cmp_game_story.narrative_arc_isomorphism > 0.50
    assert "self_agency_interactive" in cmp_game_story.agency_divergence
    assert "control" in cmp_game_story.modality_divergence["only_in_a"]

    cmp_story_mp4 = observer.compare_causal_narratives(rep_story, rep_mp4)
    assert "audio" in cmp_story_mp4.modality_divergence["only_in_a"]
    assert "language" in cmp_story_mp4.modality_divergence["only_in_a"]
    assert "visual" in cmp_story_mp4.modality_divergence["shared_modalities"]


def test_v32_sphere_hierarchy_4d_causal_field_ssd_and_20_organ_self_audit(tmp_path: Path):
    """Full v3.2 absorption into OpenVirtualRelationalMemory, 4D CausalField, SSD Ice, and 20-organ audit."""
    slab_path = tmp_path / "elysia_v32_media.ice"
    with IceWaterGasRotorRuntime(slab_path=slab_path) as mem_rt:
        cycle = ElysiaUnifiedLivingCycle(memory_runtime=mem_rt)
        observer = ConnectedMediaCausalObserver(living_cycle=cycle)

        observer.observe_episode(episode_id="real_mp4", video_source=_REAL_MP4, max_frames=20)
        observer.observe_bundle(run_pixel_rendered_interactive_game(mode="interactive_play"))
        observer.observe_bundle(build_audiovisual_narrative_scene("volitional"))

        topo = observer.memory.topology_metrics()
        assert set(topo["scale_tiers_occupied"]) == {"earth", "water", "sky"}
        assert topo["warp_links"] > 0
        assert topo["weft_links"] > 0
        assert mem_rt.ice.num_crystals >= 3
        assert cycle.verify_autobiographical_integrity() is True

        field = CausalField(dimensions=3)
        sync = observer.sync_to_causal_field(causal_field=field)
        assert sync["beams_linked"] > 0
        for vox in field.voxels.values():
            assert tuple(vox.tensor.shape) == (4,)
        field.step(dt=0.1)

        audit = cycle.introspect_own_architecture()
        assert audit.total_organ_files >= 20
        assert "orchestration/connected_media_causal_observer.py" in [
            m["relative_path"] for m in audit.verified_organ_modules
        ]


def test_graceful_boundary_friction_absorption_on_missing_or_corrupt_media():
    """Corrupt or missing media paths never crash; absorbed as BoundaryFrictionEvent on the sphere."""
    observer = ConnectedMediaCausalObserver()
    rep = observer.observe_episode(
        episode_id="broken_boundary_probe",
        video_source=Path(r"c:\Elysia\does_not_exist_clip.mp4"),
        audio_source=Path(r"c:\Elysia\does_not_exist_sound.wav"),
    )
    assert rep.present_modalities == []
    assert set(rep.absent_modalities) == {"visual", "audio", "language", "control"}
    assert len(observer.memory.boundary_events) == 2
