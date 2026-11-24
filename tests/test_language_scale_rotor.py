r"""
Verification suite for ELYSIA 4-Scale Generative Language Rotor:
Tests Scale 0 Hangul Jamo & coda operator harmony (`이/가`, `은/는`, `을/를`, `으로/로`),
Scale 1->2->3 unsupervised induction producing genuinely novel non-verbatim Korean speech,
and self-detected ExpressiveDeficit triggering autonomous world inquiry before replying.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.embodiment import (
    LanguageScaleRotor,
    ScaleRotorEmbodiedMind,
    bind_korean_operator,
    compose_hangul_jamo,
    decompose_hangul_jamo,
    has_jongseong,
)


def test_hangul_jamo_axis_principle_and_coda_binding():
    """Scale 0 cleanly decomposes/composes Hangul jamo and harmonizes Korean operators across codas."""
    assert decompose_hangul_jamo("한") == ("ㅎ", "ㅏ", "ㄴ")
    assert compose_hangul_jamo("ㅎ", "ㅏ", "ㄴ") == "한"
    assert has_jongseong("마음") is True
    assert has_jongseong("바다") is False

    assert bind_korean_operator("마음", "이/가") == "마음이"
    assert bind_korean_operator("바다", "이/가") == "바다가"
    assert bind_korean_operator("하늘", "을/를") == "하늘을"
    assert bind_korean_operator("소리", "을/를") == "소리를"
    assert bind_korean_operator("하늘", "으로/로") == "하늘로"
    assert bind_korean_operator("마음", "으로/로") == "마음으로"


def test_few_shot_axis_induction_generates_novel_non_verbatim_utterance():
    """Induces stems/operators/schemas from 12 sentences and composes a brand-new Korean utterance."""
    training = [
        ("퇴근길 비가 내렸다.", "발걸음이 느려졌다.", "마음이 너무 무거워서 걷기가 힘들어요."),
        ("바람이 불어왔다.", "창문을 열었다.", "마음을 조금 내려놓으면 숨결이 편해져요."),
        ("하늘이 개었다.", "구름이 흩어졌다.", "바람이 불어도 하늘은 여전히 맑네요."),
        ("밤늦도록 일했다.", "한숨이 깊어졌다.", "혼자서 버티느라 얼마나 지쳤을까?"),
        ("조용한 벤치에 앉았다.", "햇살이 비췄다.", "잠시 창밖을 바라보며 쉬어가자."),
        ("친구와 마주 앉았다.", "이야기를 꺼냈다.", "서로 이야기를 나누면 걱정이 가벼워진다."),
        ("숨이 가빠졌다.", "눈을 감았다.", "지금은 천천히 호흡을 가다듬고 있어요."),
        ("비가 그쳤다.", "바람이 불었다.", "비가 그치면 들꽃 향기가 올라오지?"),
        ("걱정이 깊어 보였다.", "조용히 말을 건넸다.", "네 마음이 조금이라도 편안해지면 좋겠어."),
        ("하루가 저물었다.", "불빛이 켜졌다.", "오늘 하루도 묵묵히 버텨냈구나!"),
        ("겨울 바람이 불었다.", "손을 잡았다.", "바람은 차갑지만 손끝의 온기는 남아있어."),
        ("머릿속이 복잡했다.", "대화를 시작했다.", "걱정이 쌓일수록 말로 꺼내놓는 게 도움이 돼."),
    ]
    samples = [{"situation": s, "cause": c, "utterance": u} for s, c, u in training]
    corpus_set = {u for _, _, u in training}

    rotor = LanguageScaleRotor()
    stats = rotor.induce_from_utterances(samples)

    assert stats["stems_total"] >= 8
    assert stats["operators_total"] >= 4
    assert stats["schemas_total"] >= 2

    composed, meta = rotor.compose_novel_utterance(
        "일이 쌓여서 마음이 무겁고 지치네",
        situation_text="오랜 작업 뒤에 피로가 남았다.",
        chroma=(0.25, 0.45, 0.70),
    )
    assert composed
    assert composed not in corpus_set
    assert meta["grounded_in_real_world"] is True
    assert meta["novel_vs_corpus"] is True
    assert len(meta["assembled_stems"]) >= 2


def test_expressive_deficit_triggers_autonomous_world_inquiry_and_resolves(tmp_path: Path):
    """
    Empty ScaleRotorEmbodiedMind detects its own ExpressiveDeficit from user input,
    autonomously queries external texts derived from user terms, induces schemas/axes,
    and replies with a genuinely novel utterance (no developer templates or verbatim copy).
    """
    drama_prose = (
        "비 내리는 밤 작업실에 앉아 있었다. "
        "“요즘 일이 풀리지 않아 마음이 무척 지치네.”\n"
        "동료가 따뜻한 차를 건네며 물었다. "
        "“지칠 때는 잠시 멈춰서 숨을 고르는 게 낫지 않아?”\n"
        "나는 창문을 열고 깊은 숨을 내쉬었다. "
        "“말을 꺼내놓고 나니까 마음이 조금 가벼워져.”\n"
        "그가 미소 지으며 자리에서 일어났다. "
        "“우리 같이 창밖 바람이나 쐬면서 쉬자.”\n"
        "바람이 얼굴을 스치자 긴장이 서서히 풀렸다. "
        "“하루가 버거울수록 천천히 걸어가면 괜찮아진다.”\n"
    )

    def _mock_wiki_fetcher(url: str, timeout: float, headers=None):
        if "list=search" in url:
            payload = {"query": {"search": [{"title": "작업실의_밤"}]}}
            return 200, json.dumps(payload).encode("utf-8"), "application/json"
        if "prop=extracts" in url:
            payload = {"query": {"pages": {"1": {"title": "작업실의_밤", "extract": drama_prose}}}}
            return 200, json.dumps(payload).encode("utf-8"), "application/json"
        return 404, b"", "text/plain"

    mind = ScaleRotorEmbodiedMind(tmp_path / "deficit_inquiry_mind")
    assert len(mind.language_rotor.schemas) == 0

    res = mind.converse_with_user(
        "일이 안 풀려서 마음이 지치네",
        allow_autonomous_inquiry=True,
        fetcher=_mock_wiki_fetcher,
    )

    assert res["deficit"]["needs_world_inquiry"] is True
    assert len(res["deficit"]["inquiry_terms"]) >= 1
    assert res["autonomous_inquiry"] is not None
    assert res["autonomous_inquiry"]["added_count"] >= 4
    assert res["language_axes"]["schemas"] >= 2
    assert res["recombination"]["grounded_in_real_world"] is True
    assert res["recombination"]["novel_vs_corpus"] is True
    assert res["reply_ko"]
    assert res["reply_ko"] not in (
        "요즘 일이 풀리지 않아 마음이 무척 지치네.",
        "지칠 때는 잠시 멈춰서 숨을 고르는 게 낫지 않아?",
        "말을 꺼내놓고 나니까 마음이 조금 가벼워져.",
        "우리 같이 창밖 바람이나 쐬면서 쉬자.",
        "하루가 버거울수록 천천히 걸어가면 괜찮아진다.",
    )
