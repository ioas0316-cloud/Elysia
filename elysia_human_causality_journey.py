r"""
ELYSIA Human Developmental Causality Journey Runner
==================================================

Demonstrates the human-developmental causal cognition & language pipeline:
  1. Sensory Scale Discernment (background noise vs. speech origin/cadence)
  2. Joint Attention & Speech Duality Realization (speech = form + referent)
  3. Unknown Word BEFORE Dictionary Knowledge (asks speaker; zero API shortcut)
  4. Acquiring World Causal Rule -> Parallel Universe -> Live Dictionary Lookup
  5. Present -> Past Reinterpretation & Recreation -> Future Parallel Universes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parent
L1_ROOT = ROOT / "L1_causal_field"
if str(L1_ROOT) not in sys.path:
    sys.path.insert(0, str(L1_ROOT))

from core.embodiment.scale_rotor_embodied_field import ScaleRotorEmbodiedMind  # noqa: E402
from core.orchestration.external_web_bridge import HttpFetcher  # noqa: E402


def _offline_dictionary_fetcher(url: str, timeout_s: float, headers: Optional[Dict[str, str]]) -> Tuple[int, bytes, str]:
    if "wiktionary.org" in url and "%EB%AC%B4%EC%A7%80%EA%B0%9C" in url:
        extract = (
            "== 한국어 ==\n=== 명사 ===\n"
            "1. 공중에 떠 있는 물방울에 햇빛이 반사·굴절되어 하늘에 나타나는 일곱 빛깔의 둥근 띠."
        )
        return (
            200,
            json.dumps({"query": {"pages": {"1": {"title": "무지개", "extract": extract}}}}).encode("utf-8"),
            "application/json",
        )
    if "wikipedia.org" in url and "%EB%AC%B4%EC%A7%80%EA%B0%9C" in url:
        return (
            200,
            json.dumps(
                {
                    "title": "무지개",
                    "description": "광학 현상",
                    "extract": "무지개는 대기 중 물방울에 의해 빛이 굴절되고 반사되어 나타나는 다색의 스펙트럼 호이다.",
                    "content_urls": {"desktop": {"page": "https://ko.wikipedia.org/wiki/무지개"}},
                }
            ).encode("utf-8"),
            "application/json",
        )
    return 404, b"{}", "application/json"


def _print_turn(label: str, speaker_input: str, res: Dict[str, Any]) -> None:
    ep = res["episode"]
    intent = res["intent"]
    print(f"\n  [{label}]")
    print(f"    입력({ep['source_origin']}, {ep['modality']}): {speaker_input or '(무음/배경 자극)'}")
    print(f"    감각분별(Scale {ep['scale_tier']}): kind={ep['perceived_kind']} | 초기이해={ep['initial_understanding']}")
    print(f"    말의 이중성 깨달음: {res['speech_duality_realized']} (근거 {len(res['grounded_meanings'])}개)")
    print(f"    의도(Intent) 먼저 형성: {intent['intent_kind']} | 가리키는 대상='{intent['referent']}' | 이유={intent['reason_ko']}")
    print(f"    표현방식: {res['expression_mode']}")
    print(f"    엘리시아 발화: {res['utterance'] or '(조용히 바라봄)'}")

    if res.get("parallel_universes"):
        print("    [평행우주 상상]")
        for b in res["parallel_universes"]:
            mark = "★선택" if b["selected"] else "  후보"
            print(f"      {mark} {b['title']} -> 예측: {b['predicted_outcome']}")

    if res.get("dictionary_lookup"):
        dl = res["dictionary_lookup"]
        if dl.get("executed"):
            print(f"    [실제 사전 탐색] ({dl['dictionary_source']}) '{dl['word']}' -> {dl['url']}")
            print(f"      사전 뜻: {dl['definition_summary']}")
        else:
            print(f"    [사전 탐색 차단/실패] {dl.get('reason')}")

    if res.get("reinterpretations"):
        print("    [현재 -> 과거 재해석 및 재창조]")
        for r in res["reinterpretations"]:
            print(f"      * 이전 시각: {r['before_view']}")
            print(f"      * 바뀐 재해석: {r['after_view']}")
            print(f"      * 재창조된 과거 행동: {r['recreated_past_action']}")
            print(f"      * 스케일 연결: {r['scale_bridge_summary']}")


def run_journey(state_dir: Path, fresh: bool, live_web: bool) -> int:
    if fresh and state_dir.exists():
        shutil.rmtree(state_dir, ignore_errors=True)
    mind = ScaleRotorEmbodiedMind(state_dir=state_dir)
    fetcher: Optional[HttpFetcher] = None if live_web else _offline_dictionary_fetcher

    print("=" * 92)
    print("  STAGE 1 · 감각 계층의 1차 분별 (배경 vs 사람 말 — 아직 말의 이중성을 알기 전)")
    print("=" * 92)
    r1 = mind.experience_human_causal_turn(
        raw_text_or_sound="창문 밖 바람 소리와 정지 화면",
        source_origin="ambient_room",
        modality="audiovisual",
        is_segmented_sequence=False,
        fetcher=fetcher,
    )
    _print_turn("1-A 배경 자극", r1["episode"]["raw_input"], r1)

    r2 = mind.experience_human_causal_turn(
        raw_text_or_sound="오늘 하늘이 참 맑네",
        source_origin="speaker_parent",
        modality="auditory",
        is_segmented_sequence=True,
        cooccurring_object=None,
        speaker_action="미소 지으며 말함",
        fetcher=fetcher,
    )
    _print_turn("1-B 사람 소리 첫 지각", "오늘 하늘이 참 맑네", r2)

    print("\n" + "=" * 92)
    print("  STAGE 2 · 공동주의(함께 바라봄)로 말의 이중성(소리+뜻) 깨닫기 & 말틀 모방")
    print("=" * 92)
    r3 = mind.experience_human_causal_turn(
        raw_text_or_sound="저거 봐, 사과야",
        source_origin="speaker_parent",
        modality="audiovisual",
        cooccurring_object="사과",
        speaker_action="손가락으로 가리킴",
        fetcher=fetcher,
    )
    _print_turn("2-A 첫 공동주의", "저거 봐, 사과야", r3)

    r4 = mind.experience_human_causal_turn(
        raw_text_or_sound="이건 비가 내리는 소리야",
        source_origin="speaker_parent",
        modality="audiovisual",
        cooccurring_object="비",
        speaker_action="빗소리를 들려줌",
        fetcher=fetcher,
    )
    _print_turn("2-B 말의 이중성 깨달음", "이건 비가 내리는 소리야", r4)

    r5 = mind.experience_human_causal_turn(
        raw_text_or_sound="모르는 말은 무슨 뜻이야? 하고 물어봐",
        source_origin="speaker_parent",
        modality="audiovisual",
        speaker_action="질문하는 법을 보여줌",
        fetcher=fetcher,
    )
    _print_turn("2-C 질문 틀 흡수", "모르는 말은 무슨 뜻이야? 하고 물어봐", r5)

    print("\n" + "=" * 92)
    print("  STAGE 3 · 사전 지식 없이 모르는 말 '무지개' 조우 -> 웹 탐색 없이 화자에게 묻기")
    print("=" * 92)
    r6 = mind.experience_human_causal_turn(
        raw_text_or_sound="하늘에 무지개가 떴어!",
        source_origin="speaker_parent",
        modality="audiovisual",
        speaker_action="창밖을 가리킴",
        fetcher=fetcher,
    )
    _print_turn("3 모르는 말 조우(사전 지식 전)", "하늘에 무지개가 떴어!", r6)

    print("\n" + "=" * 92)
    print("  STAGE 4 · 세상 배경지식 습득 -> 사전 탐색 평행우주 개방 -> 실제 사전 탐색")
    print("=" * 92)
    r7 = mind.experience_human_causal_turn(
        raw_text_or_sound="모르는 말의 뜻은 사전에서 찾아보면 알 수 있어",
        source_origin="speaker_parent",
        modality="audiovisual",
        speaker_action="사전의 역할을 알려줌",
        fetcher=fetcher,
    )
    _print_turn("4-A 배경지식 습득", "모르는 말의 뜻은 사전에서 찾아보면 알 수 있어", r7)

    r8 = mind.experience_human_causal_turn(
        raw_text_or_sound="아까 말한 무지개 기억나?",
        source_origin="speaker_parent",
        modality="audiovisual",
        speaker_action="되물어봄",
        allow_dictionary_lookup=True,
        fetcher=fetcher,
    )
    _print_turn("4-B 사전 탐색 실행 & 과거 재해석", "아까 말한 무지개 기억나?", r8)

    print("\n" + "=" * 92)
    print("  STAGE 5 · 과거-현재-미래 인과 고리 확인 및 새로운 대상('프리즘') 호응")
    print("=" * 92)
    past_ep = next((ep for ep in mind.causal_mind.episodes if "무지개" in ep.raw_input and ep.reinterpretations), None)
    if past_ep is not None:
        print(f"  [과거 기억 보존 확인] {past_ep.episode_id} 초기이해: {past_ep.initial_understanding}")
        print(f"  [과거->현재->미래 고리] {past_ep.future_causal_link}")

    r9 = mind.experience_human_causal_turn(
        raw_text_or_sound="저기 프리즘도 한번 볼래?",
        source_origin="speaker_parent",
        modality="audiovisual",
        cooccurring_object="프리즘",
        speaker_action="유리 프리즘을 비춰줌",
        fetcher=fetcher,
    )
    _print_turn("5 새로운 대상 공동주의 모방 발화", "저기 프리즘도 한번 볼래?", r9)

    print("\n" + "=" * 92)
    print("  STAGE 6 · 관계적 피드백으로 과잉일반화 자기 교정 & 경계 수정 검증")
    print("=" * 92)
    r10 = mind.experience_human_causal_turn(
        raw_text_or_sound="아니야, 프리즘은 내리는 소리가 아니라 빛을 여러 색으로 나누는 유리야",
        source_origin="speaker_parent",
        modality="audiovisual",
        speaker_action="고개를 저으며 설명해줌",
        fetcher=fetcher,
    )
    _print_turn("6-A 관계적 교정 및 과거 실수 재창조", "아니야, 프리즘은 내리는 소리가 아니라 빛을 여러 색으로 나누는 유리야", r10)

    r11 = mind.experience_human_causal_turn(
        raw_text_or_sound="저기 거울도 봐",
        source_origin="speaker_parent",
        modality="visual",
        cooccurring_object="거울",
        speaker_action="거울을 비춰줌",
        fetcher=fetcher,
    )
    _print_turn("6-B 경계 수정 후 다른 시각 대상 호응", "저기 거울도 봐", r11)

    print("\n" + "=" * 92)
    print("  STAGE 7 · 언어 밖 배움 (시각·물리 현상 관찰 + 교차 공명 & 도구 인과 배움)")
    print("=" * 92)
    p1 = mind.experience_world_phenomenon_turn(
        phenomenon_name="프리즘 빛 분산",
        cause_condition="유리 프리즘에 햇빛을 비춤",
        observed_effect="일곱 색깔 빛 띠로 나뉘어 벽에 펼쳐짐",
        sensory_channel="visual",
    )
    print("\n  [7-A1 물리 현상 직접 관찰 · 프리즘 빛 분산]")
    print(f"    엘리시아: {p1['utterance']}")
    if p1.get("cross_domain_links"):
        print(f"    [교차 도메인 연결] {p1['cross_domain_links']}")
    if p1.get("reinterpretations"):
        for r in p1["reinterpretations"]:
            print(f"      * 과거 재해석({r['scale_bridge_summary']}): {r['after_view']}")

    p2 = mind.experience_world_phenomenon_turn(
        phenomenon_name="거울 빛 반사",
        cause_condition="거울에 손전등 빛을 비춤",
        observed_effect="빛이 꺾여 반대쪽 벽으로 반사되어 나아감",
        sensory_channel="visual",
    )
    print("\n  [7-A2 물리 현상 인과 원리 깨달음 · 거울 빛 반사]")
    print(f"    현상 인과 원리 깨달음: {p2['phenomenon_causality_realized']} | 엘리시아: {p2['utterance']}")

    t_guard = mind.experience_action_tool_turn(
        goal_description="멀리 있는 작은 무늬 자세히 보기",
        goal_condition="visual_detail_too_small",
        attempted_tool="zoom_lens",
    )
    t_fail = mind.experience_action_tool_turn(
        goal_description="멀리 있는 작은 무늬 자세히 보기",
        goal_condition="visual_detail_too_small",
        attempted_tool="bare_eye_focus",
        observed_effect="작아서 흐릿하고 선이 뭉개져 보임",
        succeeded=False,
    )
    print("\n  [7-B 도구 배경지식 전 · 막힌 목표]")
    print(f"    인과 차단 확인(미학습 도구 실행 거부): blocked={t_guard['blocked_by_causality']} | 사유={t_guard['reason']}")
    print(f"    맨눈 시도: {t_fail['utterance']}")
    print("    [평행우주(도구 규칙 없음)] -> " + " / ".join(b["title"] for b in t_fail["parallel_universes"]))

    t_rule = mind.experience_action_tool_turn(
        goal_description="멀리 있는 작은 무늬 자세히 보기",
        goal_condition="visual_detail_too_small",
        teach_tool_rule={
            "medium": "zoom_lens",
            "condition": "visual_detail_too_small",
            "causal_explanation": "작아서 흐릿한 무늬는 돋보기로 확대해서 보면 또렷하게 보여",
        },
    )
    t_ok = mind.experience_action_tool_turn(
        goal_description="멀리 있는 작은 무늬 자세히 보기",
        goal_condition="visual_detail_too_small",
        attempted_tool="zoom_lens",
        observed_effect="작은 무늬의 경계와 선이 크게 확대되어 또렷이 보임",
        succeeded=True,
    )
    print("\n  [7-C 도구 배경지식 습득 -> 평행우주 개방 -> 도구 실행 & 실패 기억 재해석]")
    print(f"    배운 후 열린 평행우주: " + " / ".join(b["title"] for b in t_rule["parallel_universes"]))
    print(f"    행동 주체성 깨달음: {t_ok['action_agency_realized']} | 엘리시아: {t_ok['utterance']}")
    if t_ok.get("reinterpretations"):
        for r in t_ok["reinterpretations"]:
            print(f"      * 이전 실패: {r['before_view']}")
            print(f"      * 바뀐 재해석: {r['after_view']}")
            print(f"      * 재창조된 행동: {r['recreated_past_action']}")
    print("=" * 92)
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Elysia Human Developmental Causality Journey")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=ROOT / "data" / "elysia_human_causality_state",
        help="State persistence directory",
    )
    parser.add_argument("--fresh", action="store_true", help="Start from a blank developmental slate")
    parser.add_argument("--live-web", action="store_true", help="Use live Wiktionary/Wikipedia HTTP lookups")
    args = parser.parse_args()
    return run_journey(args.state_dir, fresh=args.fresh, live_web=args.live_web)


if __name__ == "__main__":
    raise SystemExit(main())
