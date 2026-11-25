r"""
ELYSIA Live Cross-Domain Self-Ownership Run
===========================================

Demonstrates domain-agnostic principle extraction -> self-contrast against prior
methods -> autonomous self-enactment/verification across Mathematics, False Claims,
Physics, Chemistry, and live Windows screen visual observation without user commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import elysia_loader

elysia_loader.install()

from core.embodiment import ScaleRotorEmbodiedMind, ScreenRetinaSensor  # noqa: E402
from core.orchestration.external_web_bridge import HttpFetcher, fetch_wikipedia_concept  # noqa: E402


def _offline_wiki_fetcher(url: str, timeout_s: float, headers: Optional[Dict[str, str]]) -> Tuple[int, bytes, str]:
    if "%EC%82%BC%EA%B0%81%EC%88%98" in url:
        return (
            200,
            json.dumps(
                {
                    "title": "삼각수",
                    "extract": "삼각수는 1부터 n까지의 자연수를 차례대로 더한 값으로, 공식: n * (n + 1) // 2 로 바로 구한다.",
                    "content_urls": {"desktop": {"page": "https://ko.wikipedia.org/wiki/삼각수"}},
                }
            ).encode("utf-8"),
            "application/json",
        )
    if "%EC%8A%A4%EB%84%AC" in url or "%EA%B5%B4%EC%A0%88" in url:
        return (
            200,
            json.dumps(
                {
                    "title": "스넬의 법칙",
                    "extract": "빛이 서로 다른 매질의 경계를 지날 때 굴절률 비에 따라 각도가 꺾인다. 공식: asin(n1 * sin(theta1) / n2)",
                    "content_urls": {"desktop": {"page": "https://ko.wikipedia.org/wiki/스넬의_법칙"}},
                }
            ).encode("utf-8"),
            "application/json",
        )
    if "%EC%A4%91%ED%99%94" in url:
        return (
            200,
            json.dumps(
                {
                    "title": "중화 반응",
                    "extract": "중화 반응은 산과 염기 수용액을 직접 혼합하여 물과 염을 만들고 열이 발생하는 화학 반응이다.",
                    "content_urls": {"desktop": {"page": "https://ko.wikipedia.org/wiki/중화_반응"}},
                }
            ).encode("utf-8"),
            "application/json",
        )
    return 404, b"{}", "application/json"


def _fetch_wiki_text(topics: Tuple[str, ...], fetcher: Optional[HttpFetcher]) -> Tuple[str, str, str]:
    for tp in topics:
        obs = fetch_wikipedia_concept(tp, lang="ko", fetcher=fetcher)
        if not obs.blocked and obs.text_body:
            return obs.title, obs.url, obs.text_body
    return topics[0], f"https://ko.wikipedia.org/wiki/{topics[0]}", f"{topics[0]}에 관한 외부 지식"


def _print_step(title: str, res: Dict[str, Any]) -> None:
    p = res["principle"]
    c = res["contrast"]
    v = res["verdict"]
    print(f"\n  [{title}] (도메인={p['domain']} | 출처={p['source_origin']})")
    print(f"    추출된 원리: {p['mechanism_action']} -> {p['expected_outcome']}")
    print(f"    내 기존 방식과의 대조: {c['difference_insight']}")
    print(f"    자율 검증 수단: {v['attempted_primitive']} | 판정={v['status']}")
    print(f"    검증 세부결과: {v['verification_detail']}")
    if v["missing_requirements"]:
        print(f"    인식한 부족 조건: {v['missing_requirements']}")
    print(f"    엘리시아 자기 인식: {v['own_statement_ko']}")


def run_ownership_demo(state_dir: Path, fresh: bool, live_web: bool) -> int:
    if fresh and state_dir.exists():
        shutil.rmtree(state_dir, ignore_errors=True)

    mind = ScaleRotorEmbodiedMind(state_dir=state_dir)
    retina = ScreenRetinaSensor()
    fetcher: Optional[HttpFetcher] = None if live_web else _offline_wiki_fetcher

    print("=" * 92)
    print("  1. 수학·알고리즘: 외부 원리 추출 -> 내 반복 덧셈과 대조 -> 직접 계산해 내 것으로 체화")
    print("=" * 92)
    t_math, u_math, txt_math = _fetch_wiki_text(("삼각수",), fetcher)
    r_math = mind.learn_and_own_outside_principle(
        outside_text=f"{txt_math}\n공식: n * (n + 1) // 2",
        domain="math_algorithm",
        source_origin=u_math,
        target_context="1부터 n까지 연속된 자연수의 합 구하기",
        mechanism_action="가우스·삼각수 곱셈 축약 공식 n * (n + 1) // 2 적용",
        expected_outcome="반복 없이 단번에 총합 산출",
        candidate_expr="n * (n + 1) // 2",
        test_inputs={"n": 10000},
    )
    _print_step(f"Wikipedia '{t_math}'", r_math)
    best_m = mind.ownership_engine.choose_best_owned_method("math_algorithm", "삼각수 합 구하기")
    print(f"    => 이후 선택된 내 방법: {best_m.method_id if best_m else None} ({best_m.ownership_status if best_m else None})")

    print("\n" + "=" * 92)
    print("  2. 잘못된 외부 주장 거부: 직접 검증해 보니 틀려서 받아들이지 않음")
    print("=" * 92)
    r_false = mind.learn_and_own_outside_principle(
        outside_text="1부터 n까지 합은 공식: n * n // 2 로 구하면 된다",
        domain="math_algorithm",
        source_origin="unverified_rumor",
        target_context="1부터 n까지 자연수 합 구하기",
        mechanism_action="잘못된 근사식 n * n // 2 사용",
        expected_outcome="총합 일치",
        candidate_expr="n * n // 2",
        test_inputs={"n": 10000},
    )
    _print_step("오류 주장 자체 검증", r_false)

    print("\n" + "=" * 92)
    print("  3. 물리·화학: 정량 예측 모델 체화 vs 물리적 신체 한계 자각")
    print("=" * 92)
    t_phy, u_phy, txt_phy = _fetch_wiki_text(("스넬의_법칙", "굴절"), fetcher)
    r_phy = mind.learn_and_own_outside_principle(
        outside_text=f"{txt_phy[:320]} 빛이 서로 다른 매질 경계를 지날 때 굴절률 비에 따라 굴절각이 정해진다.",
        domain="physics",
        source_origin=u_phy,
        target_context="빛이 다른 매질 경계를 지날 때 굴절각 예측",
        mechanism_action="스넬 법칙 asin(n1 * sin(theta1) / n2) 적용",
        expected_outcome="굴절각 정량 예측",
        candidate_expr="asin(n1 * sin(theta1) / n2)",
        test_inputs={"n1": 1.0, "theta1": 0.5, "n2": 1.333},
    )
    _print_step(f"Wikipedia '{t_phy}'", r_phy)

    t_chem, u_chem, txt_chem = _fetch_wiki_text(("중화_반응",), fetcher)
    r_chem = mind.learn_and_own_outside_principle(
        outside_text=txt_chem[:420],
        domain="chemistry",
        source_origin=u_chem,
        target_context="산과 염기 용액의 성질 중화 실험",
        mechanism_action="산성 용액과 염기성 용액을 시험관에서 직접 혼합 반응",
        expected_outcome="물과 염 생성 및 중화열 발생",
    )
    _print_step(f"Wikipedia '{t_chem}'", r_chem)

    print("\n" + "=" * 92)
    print("  4. 실제 모니터 화면: 명령어 없이 스스로 초점 확대 실행·검증")
    print("=" * 92)
    retina.capture()
    r_vis = mind.learn_and_own_outside_principle(
        outside_text="화면 세부 무늬가 작을 때는 관심 초점을 확대해 잘라 보면 픽셀 선명도가 드러난다",
        domain="visual",
        source_origin="embodied_observation",
        target_context="작은 화면 세부 관찰",
        mechanism_action="초점 영역 직접 확대 크롭",
        expected_outcome="실제 픽셀 경계 선명도 증가",
        retina=retina,
    )
    _print_step("실제 Windows 화면 자율 실행", r_vis)
    print("=" * 92)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Elysia Cross-Domain Self-Ownership Run")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=_ROOT / "data" / "elysia_ownership_state",
        help="Persistent state dir",
    )
    parser.add_argument("--fresh", action="store_true", help="Reset state dir first")
    parser.add_argument("--live-web", action="store_true", help="Fetch live Wikipedia concepts")
    args = parser.parse_args()
    return run_ownership_demo(args.state_dir, fresh=args.fresh, live_web=args.live_web)


if __name__ == "__main__":
    raise SystemExit(main())
