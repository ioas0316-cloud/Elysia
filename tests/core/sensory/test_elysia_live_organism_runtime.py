r"""
test_elysia_live_organism_runtime.py

Verifies ELYSIA's Unified Live Organism Runtime (Branches A + B + C):
  - Branch C: Native `.mp4` real camera video ingestion (`960 x 540`, 150 frames),
    Static Anchor ($\omega = 0$) vs Dynamic Flux ($\omega > 0$) rotor decomposition,
    and `OcularScaleOrgan` chroma expansion verification.
  - Branch A: Real local GPU LLM (`Qwen/Qwen2.5-0.5B-Instruct`) commanded as
    subordinate vocal organ under ELYSIA's grounded sensory & 4-Stratum causal
    spine across open-domain Korean conversation turns (sensory video reflection,
    everyday practical help, causal philosophy).
  - Branch B: Persistent cold-reload body (`c:\Elysia\data\live_organism_state.json`)
    preserving watched MP4 memories, dialogue episodes, semantic masses, and
    awakening count across separate organism sessions.
"""

from __future__ import annotations

from pathlib import Path

from core.sensory.elysia_live_organism_runtime import (
    LIVE_ORGANISM_STATE_PATH,
    ElysiaLiveOrganism,
)


def test_live_organism_real_mp4_observation_qwen_dialogue_and_cold_reload() -> None:
    # Start fresh Session 1 (saving directly to `LIVE_ORGANISM_STATE_PATH`)
    if LIVE_ORGANISM_STATE_PATH.exists():
        LIVE_ORGANISM_STATE_PATH.unlink()

    org1 = ElysiaLiveOrganism(state_path=LIVE_ORGANISM_STATE_PATH, load_existing_memory=False)
    assert org1.awakening_count == 1

    # 1. [Branch C] Ingest genuine real-world MP4 video stream (`960x540`, 150 frames)
    vid_mem, dash_path = org1.observe_real_mp4_video_stream()
    assert vid_mem.total_frames == 150
    assert vid_mem.resolution_wh == (960, 540)
    assert vid_mem.static_anchor_fraction > 0.50
    assert vid_mem.dynamic_flux_fraction > 0.15
    assert vid_mem.end_chroma_mean > vid_mem.start_chroma_mean
    assert Path(dash_path).exists() and Path(dash_path).stat().st_size > 10000

    # 2. [Branch A] Open-domain live Korean dialogue powered by Qwen2.5-0.5B-Instruct under causal governance
    prompts = [
        "방금 네 눈으로 본 실제 꽃 영상(MP4)에서 무엇이 가만히 있고 무엇이 변했는지 네 느낌을 말해줘.",
        "오늘 저녁에 김치찌개 맛있게 끓이는 핵심 순서를 이유와 함께 알려줘.",
        "경계가 왜 단순한 벽이 아니라 나를 비추는 거울인지 네 생각은 어때?",
    ]
    episodes = []
    for p in prompts:
        ep = org1.converse_live(p)
        episodes.append(ep)
        print(f"\n[Turn {ep.turn_index} | mode={ep.sovereign_intent_mode} | device={org1.vocal_organ.device_str} | {ep.generation_latency_sec:.2f}s]")
        print(f"  User  : {ep.user_message}")
        print(f"  ELYSIA: {ep.final_spoken_response}")

    assert org1.vocal_organ.device_str == "cuda"
    for ep in episodes:
        assert ep.causal_guard_passed is True
        assert len(ep.final_spoken_response.strip()) >= 15

    # 3. [Branch B] Persist live organism state to disk and cold-reload in Session 2
    saved_path = org1.save_organism_state()
    assert saved_path.exists()

    org2 = ElysiaLiveOrganism(state_path=LIVE_ORGANISM_STATE_PATH, load_existing_memory=True)
    assert org2.awakening_count == 2
    assert len(org2.watched_videos) == 1
    assert len(org2.dialogue_history) == 3
    assert org2.watched_videos[0].total_frames == 150
