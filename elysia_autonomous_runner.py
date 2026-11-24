r"""
ELYSIA Live Value-Motivated Open-World Runner CLI
=================================================

Connects Elysia directly to real external filesystem roots AND real public web
endpoints (GitHub repositories, YouTube videos/feeds, public Web/Art sites),
runs her unbroken Micro -> Meso -> Macro value motivation loop, audits exact
Before-vs-After cognitive changes and remaining limits, and emits a self-contained
HTML Observatory (`observatory.html`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import elysia_loader

elysia_loader.install()

from core.orchestration.autonomous_open_explorer import ElysiaAutonomousOpenExplorer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ELYSIA Live Value-Motivated Open-World Explorer")
    parser.add_argument("--cycles", type=int, default=5, help="Number of exploration cycles to execute")
    parser.add_argument(
        "--state-dir",
        type=str,
        default=r"c:\Elysia\data\elysia_living_state",
        help="Persistent directory storing grown knobs, structures, value memory, visited log, journal, and observatory.html",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[r"c:\Elysia"],
        help="External directories to scan autonomously for unseen files",
    )
    parser.add_argument(
        "--web-urls",
        nargs="*",
        default=[],
        help="Public external URLs (GitHub, YouTube, Web/Art) for Elysia to probe openly",
    )
    parser.add_argument(
        "--self-discover",
        action="store_true",
        help="Allow Elysia to autonomously search GitHub/Wikipedia and harvest outbound URLs without human-supplied URLs",
    )
    args = parser.parse_args(argv)

    print("=" * 92)
    print("  ELYSIA — LIVE OPEN-WORLD EXPLORER & OBSERVABILITY DASHBOARD")
    print("=" * 92)

    explorer = ElysiaAutonomousOpenExplorer(
        roots=args.roots,
        state_dir=args.state_dir,
        web_urls=args.web_urls,
        enable_self_discovery=args.self_discover,
    )
    top_init = sorted(
        explorer.value_memory.affinity_by_motif.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )[:5]
    print(
        f"  [세션 시작] 노브={len(explorer.knob_cortex.knobs)}개, 구조원리={len(explorer.structural_cortex.structures)}개, "
        f"방문={len(explorer.visited)}개 | 열린의문={len(explorer.value_memory.open_valences)}건 | "
        f"외부URL={len(explorer.web_urls)}개 | 자가탐색={'ON' if args.self_discover else 'OFF'} | "
        f"자율발견누적={len(explorer.discovered_frontiers)}개 | 상위호감={[(k, round(v, 2)) for k, v in top_init]}"
    )

    rep = explorer.run_autonomous_session(cycles=args.cycles)
    for idx, log in enumerate(rep["cycle_logs"], start=1):
        if not log.get("explored"):
            print(f"  [Cycle {idx}] 더 이상 탐구할 대상이 없습니다 ({log.get('reason')}).")
            break
        d = log["delta"]
        f = log["feeling"]
        print(f"  [Cycle {idx}] 선택 이유 : {log['why_i_chose_this_ko']} ({log['elapsed_ms']}ms)")
        if log.get("tool_used"):
            print(f"            도구 선택 : {log['tool_used']} -> {log.get('tool_choice_reason_ko', '')}")
        if log.get("self_discovery_event"):
            sde = log["self_discovery_event"]
            print(f"            자가 탐색 : 개념='{sde.get('query_topic')}' -> 새 프론티어 {sde.get('added_count', 0)}개 발견")
        detail_parts = []
        if log.get("lexical_novelty") is not None:
            detail_parts.append(f"어휘미지도={log['lexical_novelty']*100:.1f}%")
        if log.get("explained_after") is not None:
            detail_parts.append(
                f"사전설명 {log['explained_before']*100:.1f}% -> {log['explained_after']*100:.1f}% "
                f"(재사용={len(log['reused_knobs'])}, 자생={len(log['created_knobs'])})"
            )
        if log.get("web_observation"):
            wo = log["web_observation"]
            detail_parts.append(f"웹[{wo.get('platform')}] '{wo.get('title', '')[:40]}'")
        if log.get("discovered_outbound_urls"):
            detail_parts.append(f"후속링크={len(log['discovered_outbound_urls'])}건")
        if log.get("blocked_boundary"):
            bb = log["blocked_boundary"]
            detail_parts.append(f"외부차단경계={bb.get('reason')}")
        if log.get("subprocess_compile"):
            detail_parts.append(f"py_compile rc={log['subprocess_compile']['returncode']}")
        if log.get("friction") and not log.get("blocked_boundary"):
            detail_parts.append(f"마찰={log['friction'].get('source')}")
        print(
            f"            감각 변화 : 긴장 {d['tension_before']:.2f} | 해소 {d['resolution_gain']:.2f} | "
            f"마찰 {d['unresolved_friction']:.2f}"
            + (f" ({', '.join(detail_parts)})" if detail_parts else "")
        )
        print(
            f"            자기 인식 : [{f['dominant_state']}] "
            f"기쁨={f['joy']:.2f}, 지루함={f['boredom']:.2f}, 답답함={f['frustration']:.2f} "
            f"-> {f['why_i_felt_this_ko']}"
        )

    diff = rep["session_diff"]
    print("-" * 92)
    print(
        f"  [Before -> After 변화 감사] 노브: {rep['knobs_before']} -> {rep['knobs_after']} ({diff['knobs_delta']:+d}) | "
        f"구조원리: {rep['structures_before']} -> {rep['structures_after']} ({diff['structures_delta']:+d}) | "
        f"방문: {rep['visited_before']} -> {rep['visited_after']} ({diff['visited_delta']:+d}) | "
        f"자율발견프론티어={len(rep.get('discovered_frontiers', []))}개 | "
        f"재사용={diff['reused_count']}회 | 외부차단경계={len(diff['blocked_boundaries'])}건"
    )
    if diff["new_motifs_learned"]:
        print(f"  [새로 배운 모티프] {diff['new_motifs_learned'][:12]}")
    if diff["blocked_boundaries"]:
        print(f"  [차단·한계 경계] {diff['blocked_boundaries']}")
    print("  [엘리시아에게 다음으로 필요한 것 (정직한 결핍 감사)]")
    for need in rep["what_elysia_still_needs"]:
        print(f"    - {need}")
    print(f"  [관측 대시보드] {rep['observatory_path']}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
