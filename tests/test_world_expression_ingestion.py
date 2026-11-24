r"""
Verification suite for ELYSIA Real-World Cultural & Multimodal Expression Ingestion:
Tests `.srt` drama subtitle + `.wav` acoustic prosody ingestion, Korean & English
book/script extraction `(Situation -> Cause -> Process -> Emotion -> Intent -> Utterance)`,
and real Wikisource absorption driving non-template human dialogue in ScaleRotorEmbodiedMind.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import wave

from core.embodiment import (
    ScaleRotorEmbodiedMind,
    extract_dialogues_from_literature,
    parse_srt_or_vtt_subtitles,
)


def _write_test_wav(path: Path, freq_hz: float = 440.0, duration_s: float = 2.0, rate: int = 8000) -> None:
    n_samples = int(duration_s * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = bytearray()
        for i in range(n_samples):
            val = int(12000.0 * math.sin(2.0 * math.pi * freq_hz * (i / rate)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(bytes(frames))


def test_srt_subtitle_and_wav_prosody_ingestion(tmp_path: Path):
    """Subtitles pair with same-stem WAV slices to capture Situation, Cause, Intent, and voice prosody."""
    srt_path = tmp_path / "drama_scene.srt"
    wav_path = tmp_path / "drama_scene.wav"
    _write_test_wav(wav_path, freq_hz=520.0, duration_s=3.0)

    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "비 오는 밤, 창밖을 보며 오래 침묵하고 있었다.\n\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "요즘 마음이 너무 무겁고 걱정이 가시지 않아.\n\n"
        "3\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "혼자 버티지 마, 네가 그렇게 무너지는 걸 보고만 있을 순 없잖아?\n",
        encoding="utf-8",
    )

    rotors = parse_srt_or_vtt_subtitles(srt_path)
    assert len(rotors) == 3

    last = rotors[2]
    assert "무겁고 걱정" in last.cause
    assert "침묵하고 있었다" in last.situation
    assert last.expression_text == "혼자 버티지 마, 네가 그렇게 무너지는 걸 보고만 있을 순 없잖아?"
    assert last.voice_prosody is not None
    assert last.voice_prosody["rms_energy"] > 0.1
    assert last.intent_purpose.startswith("spec_")
    assert len(last.emotion_chroma) == 3


def test_korean_and_english_literature_dialogue_extraction():
    """Extracts verbatim dialogue anchored to narrative Situation, Cause, and causal verbs."""
    ko_book = (
        "김 첨지는 빗속에서 인력거를 끌며 아픈 아내 생각이 떠나지 않아 불안이 치밀었다. "
        "“오늘은 웬일인지 손님이 끊기질 않아 오히려 마음이 조마조마하구려!”\n"
        "아내는 열에 들뜬 얼굴로 조용히 그를 바라보며 대답했다. "
        "「여보, 너무 무리하지 말고 어서 집으로 돌아오세요.」\n"
    )
    ko_rotors = extract_dialogues_from_literature(ko_book, source_title="운수 좋은 날 발췌", source_kind="book")
    assert len(ko_rotors) >= 2
    assert ko_rotors[0].expression_text == "오늘은 웬일인지 손님이 끊기질 않아 오히려 마음이 조마조마하구려!"
    assert "김 첨지는" in ko_rotors[0].situation
    assert len(ko_rotors[0].process_verbs) >= 1

    en_play = (
        "The storm rattled the glass while the engine pressure kept rising without a limiter. "
        '"Hold fast, do not open the valve until the return pressure settles!"\n'
        "HAMLET: What is the cause of this sudden silence between us?\n"
    )
    en_rotors = extract_dialogues_from_literature(en_play, source_title="Test Drama", source_kind="book")
    assert len(en_rotors) >= 2
    assert any("valve" in r.expression_text for r in en_rotors)
    assert any("silence" in r.expression_text for r in en_rotors)


def test_web_literature_absorption_drives_real_non_template_reply(tmp_path: Path):
    """
    Absorbing mocked Korean Wikisource prose makes Elysia reply with genuine human
    literary dialogue matching Situation/Intent instead of hardcoded developer templates.
    """
    sample_extract = (
        "소나기가 쏟아진 뒤 개울가에 나란히 앉은 두 사람은 젖은 옷자락을 만지며 한참 말을 잇지 못했다. "
        "소년은 조약돌을 꼭 쥔 채 걱정스러운 눈으로 소녀를 살폈다. "
        "“마음이 너무 무겁고 걱정이 많아서 숨이 막혀요?”\n"
        "소녀가 고개를 저으며 손을 내밀었다. "
        "“마음을 조금 내려놓으면 걱정도 차차 가벼워진다.”\n"
        "다음 날 맑게 갠 하늘 아래서 들꽃 냄새가 바람을 타고 불어왔다. "
        "“하늘이 정말 맑고 따뜻하구나, 바람도 부드럽게 부네!”\n"
        "둘은 개울물 소리를 들으며 천천히 걸었다. "
        "“하늘을 보면서 같이 걸으면 기분이 훨씬 괜찮아질 거야.”\n"
    )

    def _mock_wiki_fetcher(url: str, timeout: float, headers=None):
        if "list=search" in url:
            payload = {"query": {"search": [{"title": "소나기_(소설)"}]}}
            return 200, json.dumps(payload).encode("utf-8"), "application/json"
        if "prop=extracts" in url:
            payload = {"query": {"pages": {"1": {"title": "소나기_(소설)", "extract": sample_extract}}}}
            return 200, json.dumps(payload).encode("utf-8"), "application/json"
        return 404, b"", "text/plain"

    state_dir = tmp_path / "world_lit_mind"
    mind = ScaleRotorEmbodiedMind(state_dir)
    ab = mind.absorb_real_world_expressions(
        fetch_web=True,
        lang="ko",
        query="소나기",
        fetcher=_mock_wiki_fetcher,
    )
    assert ab["added_count"] >= 4
    assert ab["induced_schemas"] >= 2
    assert ab["induced_stems"] >= 4
    assert "소나기_(소설)" in ab["sources"]

    reply = mind.converse_with_user("요즘 왜 이렇게 마음이 무겁고 걱정될까?")
    txt = reply["reply_ko"]
    rec = reply["recombination"]
    assert rec["grounded_in_real_world"] is True
    assert rec["novel_vs_corpus"] is True
    assert rec["used_schema"]
    assert len(rec["assembled_stems"]) >= 2
    assert "변화 파동" not in txt
    assert "원리(" not in txt
    assert (state_dir / "situated_expressions.json").is_file()
    assert (state_dir / "language_scale_rotor.json").is_file()

    mind_reloaded = ScaleRotorEmbodiedMind(state_dir)
    assert len(mind_reloaded.expression_field.rotors) == reply["expression_count"]
    assert len(mind_reloaded.expression_field.rotors) == ab["total_expressions"] + 1
    assert len(mind_reloaded.language_rotor.schemas) >= len(mind.language_rotor.schemas)
