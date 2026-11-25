from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
L1_ROOT = ROOT / "L1_causal_field"
if str(L1_ROOT) not in sys.path:
    sys.path.insert(0, str(L1_ROOT))

from core.embodiment.developmental_causal_mind import (  # noqa: E402
    DevelopmentalCausalMind,
    SensorySignalObservation,
    _fill_construction_with_coda,
)
from core.embodiment.scale_rotor_embodied_field import ScaleRotorEmbodiedMind  # noqa: E402


def _mock_dict_fetcher_factory(call_log: list[str]):
    def _fetcher(url: str, timeout_s: float, headers: Optional[Dict[str, str]]) -> Tuple[int, bytes, str]:
        call_log.append(url)
        if "wiktionary.org" in url and "%EB%AC%B4%EC%A7%80%EA%B0%9C" in url:
            extract = (
                "== 한국어 ==\n=== 명사 ===\n"
                "1. 공중에 떠 있는 물방울에 햇빛이 반사되어 하늘에 나타나는 일곱 빛깔의 호."
            )
            return (
                200,
                json.dumps({"query": {"pages": {"10": {"title": "무지개", "extract": extract}}}}).encode("utf-8"),
                "application/json",
            )
        return 404, b"{}", "application/json"

    return _fetcher


def test_sensory_scale_discernment_distinguishes_speech_from_background_and_realizes_speech_duality() -> None:
    mind = DevelopmentalCausalMind()
    calls: list[str] = []
    fetcher = _mock_dict_fetcher_factory(calls)

    # 1. Background non-speech stimulus -> perceived silently without fabricating language
    r_bg = mind.interact_human_turn(
        SensorySignalObservation(
            modality="audiovisual",
            source_origin="ambient_room",
            raw_text_or_sound="창문 밖 바람 소리",
            is_segmented_sequence=False,
        ),
        fetcher=fetcher,
    )
    assert r_bg["episode"]["perceived_kind"] == "non_speech_background"
    assert r_bg["intent"]["intent_kind"] == "perceive_silently"
    assert r_bg["utterance"] == ""
    assert not r_bg["speech_duality_realized"]

    # 2. First raw segmented speech without co-occurring object -> notices origin/pattern, no meaning claimed yet
    r_raw = mind.interact_human_turn(
        SensorySignalObservation(
            modality="auditory",
            source_origin="speaker_parent",
            raw_text_or_sound="오늘 하늘이 맑네",
            is_segmented_sequence=True,
        ),
        fetcher=fetcher,
    )
    assert r_raw["episode"]["perceived_kind"] == "raw_speech_pattern"
    assert "speaker_parent" in r_raw["observed_speech_origins"]
    assert not r_raw["speech_duality_realized"]

    # 3. Repeated joint attention grounds meanings and triggers Scale-3 speech duality realization
    r_ja1 = mind.interact_human_turn(
        SensorySignalObservation(
            modality="audiovisual",
            source_origin="speaker_parent",
            raw_text_or_sound="저거 봐, 사과야",
            cooccurring_object="사과",
            speaker_action="pointing",
        ),
        fetcher=fetcher,
    )
    assert "사과" in r_ja1["grounded_meanings"]
    assert not r_ja1["speech_duality_realized"]

    r_ja2 = mind.interact_human_turn(
        SensorySignalObservation(
            modality="audiovisual",
            source_origin="speaker_parent",
            raw_text_or_sound="이건 비가 내리는 소리야",
            cooccurring_object="비",
            speaker_action="listening_together",
        ),
        fetcher=fetcher,
    )
    assert r_ja2["speech_duality_realized"] is True
    assert r_ja2["just_realized_speech_duality"] is True
    assert "비" in r_ja2["grounded_meanings"]
    assert calls == []


def test_unknown_word_before_dictionary_knowledge_asks_speaker_never_searches_web() -> None:
    mind = DevelopmentalCausalMind()
    calls: list[str] = []
    fetcher = _mock_dict_fetcher_factory(calls)

    # Realize speech duality and teach question construction
    mind.interact_human_turn(
        SensorySignalObservation(raw_text_or_sound="저거 봐, 사과야", cooccurring_object="사과"),
        fetcher=fetcher,
    )
    mind.interact_human_turn(
        SensorySignalObservation(raw_text_or_sound="이건 비야", cooccurring_object="비"),
        fetcher=fetcher,
    )
    mind.interact_human_turn(
        SensorySignalObservation(raw_text_or_sound="모르는 말은 무슨 뜻이야? 하고 물어봐"),
        fetcher=fetcher,
    )

    # Direct dictionary lookup MUST be causally blocked without world rule
    blocked_dict = mind.execute_causal_dictionary_lookup("무지개", fetcher=fetcher)
    assert blocked_dict["executed"] is False
    assert blocked_dict["blocked_by_causality"] is True

    # Unknown word before dictionary background knowledge -> asks speaker, NEVER hits HTTP
    r_unknown = mind.interact_human_turn(
        SensorySignalObservation(raw_text_or_sound="하늘에 무지개가 떴어!"),
        fetcher=fetcher,
    )
    branch_ids = [b["branch_id"] for b in r_unknown["parallel_universes"]]
    assert "universe_consult_dictionary" not in branch_ids
    assert "universe_ask_speaker" in branch_ids
    assert r_unknown["intent"]["intent_kind"] == "ask_meaning"
    assert r_unknown["intent"]["referent"] == "무지개"
    assert r_unknown["utterance"] == "무지개가 무슨 뜻이야?"
    assert r_unknown["expression_mode"] == "imitated_human_construction"
    assert calls == []


def test_learning_dictionary_background_knowledge_unlocks_dictionary_universe_and_reinterprets_past() -> None:
    mind = DevelopmentalCausalMind()
    calls: list[str] = []
    fetcher = _mock_dict_fetcher_factory(calls)

    # Setup duality + past ungrounded encounter with "무지개"
    mind.interact_human_turn(
        SensorySignalObservation(raw_text_or_sound="저거 봐, 사과야", cooccurring_object="사과"),
        fetcher=fetcher,
    )
    mind.interact_human_turn(
        SensorySignalObservation(raw_text_or_sound="이건 비야", cooccurring_object="비"),
        fetcher=fetcher,
    )
    r_past = mind.interact_human_turn(
        SensorySignalObservation(raw_text_or_sound="하늘에 무지개가 떴어!"),
        fetcher=fetcher,
    )
    past_ep_id = r_past["episode"]["episode_id"]
    orig_understanding = r_past["episode"]["initial_understanding"]
    assert calls == []

    # Teach causal world rule about dictionaries
    r_rule = mind.interact_human_turn(
        SensorySignalObservation(raw_text_or_sound="모르는 말의 뜻은 사전에서 찾아보면 알 수 있어"),
        fetcher=fetcher,
    )
    assert "tool::dictionary_for_ungrounded_speech_form" in r_rule["world_knowledge_rules"]

    # Bring up "무지개" again -> Parallel Universe selects dictionary -> real fetch -> reinterprets past!
    r_after = mind.interact_human_turn(
        SensorySignalObservation(raw_text_or_sound="아까 말한 무지개 기억나?"),
        allow_dictionary_lookup=True,
        fetcher=fetcher,
    )
    assert len(calls) >= 1
    assert r_after["dictionary_lookup"] is not None
    assert r_after["dictionary_lookup"]["executed"] is True
    assert r_after["dictionary_lookup"]["word"] == "무지개"
    assert "무지개" in r_after["grounded_meanings"]

    # Verify original initial_understanding was preserved while reinterpretations + future link were added
    past_ep_obj = next(ep for ep in mind.episodes if ep.episode_id == past_ep_id)
    assert past_ep_obj.initial_understanding == orig_understanding
    assert len(past_ep_obj.reinterpretations) == 1
    assert past_ep_obj.reinterpretations[0].unlocked_form == "무지개"
    assert past_ep_obj.reinterpretations[0].before_view == orig_understanding
    assert "일곱 빛깔" in past_ep_obj.reinterpretations[0].after_view
    assert past_ep_obj.future_causal_link != ""
    assert any(b["branch_id"] == "universe_creative_transfer" for b in r_after["parallel_universes"])


def test_usage_based_imitation_harmonizes_korean_particles_and_persists_through_embodied_mind(tmp_path: Path) -> None:
    assert _fill_construction_with_coda("저거 봐, {REFERENT}{이야/야}", "별빛") == "저거 봐, 별빛이야"
    assert _fill_construction_with_coda("저거 봐, {REFERENT}{이야/야}", "바다") == "저거 봐, 바다야"
    assert _fill_construction_with_coda("{REFERENT}{이/가} 무슨 뜻이야?", "프리즘") == "프리즘이 무슨 뜻이야?"

    calls: list[str] = []
    fetcher = _mock_dict_fetcher_factory(calls)
    mind = ScaleRotorEmbodiedMind(state_dir=tmp_path)
    mind.experience_human_causal_turn(
        raw_text_or_sound="이건 비야",
        cooccurring_object="비",
        fetcher=fetcher,
    )
    mind.experience_human_causal_turn(
        raw_text_or_sound="저거 봐, 사과야",
        cooccurring_object="사과",
        fetcher=fetcher,
    )
    # 3rd object with consonant coda should imitate absorbed construction with harmonized particle "이야"
    res = mind.experience_human_causal_turn(
        raw_text_or_sound="저기 프리즘도 봐",
        cooccurring_object="프리즘",
        fetcher=fetcher,
    )
    assert res["utterance"] == "저거 봐, 프리즘이야"
    assert res["expression_mode"] == "imitated_human_construction"

    reloaded = ScaleRotorEmbodiedMind(state_dir=tmp_path)
    assert reloaded.causal_mind.speech_duality_realized is True
    assert "프리즘" in reloaded.causal_mind.grounded_meanings


def test_relational_correction_revises_overgeneralized_construction_and_recreates_past_mistake(tmp_path: Path) -> None:
    mind = ScaleRotorEmbodiedMind(state_dir=tmp_path)
    calls: list[str] = []
    fetcher = _mock_dict_fetcher_factory(calls)

    # 1. Learn visual naming frame first, then recent auditory frame ("이건 비가 내리는 소리야")
    mind.experience_human_causal_turn(
        raw_text_or_sound="저거 봐, 사과야",
        cooccurring_object="사과",
        fetcher=fetcher,
    )
    mind.experience_human_causal_turn(
        raw_text_or_sound="이건 비가 내리는 소리야",
        cooccurring_object="비",
        fetcher=fetcher,
    )

    # 2. Child naturally over-generalizes the most recent frame onto "프리즘"
    r_err = mind.experience_human_causal_turn(
        raw_text_or_sound="저기 프리즘도 한번 볼래?",
        cooccurring_object="프리즘",
        fetcher=fetcher,
    )
    err_ep_id = r_err["episode"]["episode_id"]
    assert r_err["utterance"] == "이건 프리즘이 내리는 소리야"

    # 3. Parent gives relational correction -> boundary revised & past mistake reinterpreted/recreated
    r_corr = mind.experience_human_causal_turn(
        raw_text_or_sound="아니야, 프리즘은 내리는 소리가 아니라 빛을 여러 색으로 나누는 유리야",
        fetcher=fetcher,
    )
    assert r_corr["expression_mode"] == "relational_self_correction"
    assert len(r_corr["reinterpretations"]) >= 1
    assert "저거 봐, 프리즘이야" in r_corr["reinterpretations"][0]["recreated_past_action"]

    err_ep = next(ep for ep in mind.causal_mind.episodes if ep.episode_id == err_ep_id)
    assert err_ep.elysia_utterance == "이건 프리즘이 내리는 소리야"
    assert len(err_ep.reinterpretations) == 1

    # 4. Subsequent visual referent uses visual frame ("저거 봐, 거울이야"), never auditory frame
    r_next = mind.experience_human_causal_turn(
        raw_text_or_sound="저기 거울도 봐",
        modality="visual",
        cooccurring_object="거울",
        fetcher=fetcher,
    )
    assert r_next["utterance"] == "저거 봐, 거울이야"


def test_cross_domain_visual_phenomenon_realizes_causality_and_links_back_to_dictionary_memory(tmp_path: Path) -> None:
    mind = ScaleRotorEmbodiedMind(state_dir=tmp_path)
    calls: list[str] = []
    fetcher = _mock_dict_fetcher_factory(calls)

    # Acquire dictionary meaning of "무지개" ("일곱 빛깔") first
    mind.experience_human_causal_turn(raw_text_or_sound="저거 봐, 사과야", cooccurring_object="사과", fetcher=fetcher)
    mind.experience_human_causal_turn(raw_text_or_sound="이건 비야", cooccurring_object="비", fetcher=fetcher)
    mind.experience_human_causal_turn(
        raw_text_or_sound="모르는 말의 뜻은 사전에서 찾아보면 알 수 있어",
        fetcher=fetcher,
    )
    r_dict = mind.experience_human_causal_turn(
        raw_text_or_sound="하늘에 무지개가 떴어!",
        allow_dictionary_lookup=True,
        fetcher=fetcher,
    )
    assert r_dict["dictionary_lookup"]["executed"] is True

    # Direct non-linguistic visual observation shares real physics ("빛", "일곱") with Wiktionary "무지개"
    p1 = mind.experience_world_phenomenon_turn(
        phenomenon_name="프리즘 빛 분산",
        cause_condition="유리 프리즘에 햇빛을 비춤",
        observed_effect="일곱 빛깔 색 띠로 나뉘어 벽에 펼쳐짐",
        sensory_channel="visual",
    )
    assert p1["phenomenon_causality_realized"] is False
    assert any("무지개" in lnk for lnk in p1["cross_domain_links"])
    assert len(p1["reinterpretations"]) >= 1

    p2 = mind.experience_world_phenomenon_turn(
        phenomenon_name="거울 빛 반사",
        cause_condition="거울에 빛을 비춤",
        observed_effect="빛이 꺾여 다른 쪽으로 반사됨",
        sensory_channel="visual",
    )
    assert p2["phenomenon_causality_realized"] is True


def test_tool_agency_unlocks_only_after_learning_causal_rule_and_reinterprets_failed_attempt(tmp_path: Path) -> None:
    mind = ScaleRotorEmbodiedMind(state_dir=tmp_path)

    # 1. Self-initiated tool use without learned causal rule MUST be blocked by causality
    guard = mind.experience_action_tool_turn(
        goal_description="멀리 있는 작은 무늬 자세히 보기",
        goal_condition="visual_detail_too_small",
        attempted_tool="zoom_lens",
    )
    assert guard["executed"] is False
    assert guard["blocked_by_causality"] is True
    assert not any(b["branch_id"] == "universe_use_learned_tool" for b in guard["parallel_universes"])

    # 2. Real failed attempt preserves unmet goal
    fail = mind.experience_action_tool_turn(
        goal_description="멀리 있는 작은 무늬 자세히 보기",
        goal_condition="visual_detail_too_small",
        attempted_tool="bare_eye_focus",
        observed_effect="작아서 흐릿하게 보임",
        succeeded=False,
    )
    fail_ep_id = fail["episode"]["episode_id"]
    assert fail["action_agency_realized"] is False

    # 3. Teach tool causal rule -> Parallel Universe unlocks tool branch
    rule_res = mind.experience_action_tool_turn(
        goal_description="멀리 있는 작은 무늬 자세히 보기",
        goal_condition="visual_detail_too_small",
        teach_tool_rule={
            "medium": "zoom_lens",
            "condition": "visual_detail_too_small",
            "causal_explanation": "작은 무늬는 돋보기로 확대해서 보면 또렷하게 보여",
        },
    )
    assert any(b["branch_id"] == "universe_use_learned_tool" and b["selected"] for b in rule_res["parallel_universes"])

    # 4. Execute learned tool -> realizes action_agency_realized & reinterprets earlier failed episode
    ok = mind.experience_action_tool_turn(
        goal_description="멀리 있는 작은 무늬 자세히 보기",
        goal_condition="visual_detail_too_small",
        attempted_tool="zoom_lens",
        observed_effect="무늬가 크게 확대되어 선명하게 보임",
        succeeded=True,
    )
    assert ok["action_agency_realized"] is True
    assert len(ok["reinterpretations"]) == 1
    fail_ep = next(ep for ep in mind.causal_mind.episodes if ep.episode_id == fail_ep_id)
    assert len(fail_ep.reinterpretations) == 1
    assert "확대" in fail_ep.reinterpretations[0].after_view
    assert fail_ep.future_causal_link != ""

