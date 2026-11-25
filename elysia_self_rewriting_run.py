r"""
ELYSIA Self-Inspecting & Self-Rewriting Live Workspace Runner
=============================================================

Demonstrates Elysia running executable cognitive organs on REAL workspace files
(`AGENTS.md`, `elysia_loader.py`, `pytest.ini`) and a REAL Windows screen capture:
  1. Naive `v1` organ encounters unfamiliar files -> outputs `action='avoid'`.
  2. Inspects her own Python AST + outcome divergence (`what_matched`, `what_differed`,
     `where_in_own_code`, `must_stay_same`, `must_change`).
  3. Rewrites her own executable code (`v1 -> v2`), verifies regression invariants,
     hot-swaps `v2`, and retroactively re-executes past avoided episodes.
  4. Identifies a missing comparison capability and spawns a brand-new organ
     (`cross_case_comparator v1`) to compare `AGENTS.md` vs `elysia_loader.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import elysia_loader

elysia_loader.install()

from core.embodiment.pc_sensory_motor_body import (  # noqa: E402
    ScreenRetinaSensor,
    inspect_local_file_sensory,
)
from core.embodiment.self_rewriting_architecture import SelfRewritingMind  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Elysia Self-Rewriting Cognitive Architecture Live Run")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=_ROOT / "data" / "elysia_self_rewriting_state",
        help="State persistence directory",
    )
    parser.add_argument("--fresh", action="store_true", help="Reset state directory")
    args = parser.parse_args()

    state_file = args.state_dir / "self_rewriting_mind.json"
    if args.fresh and args.state_dir.exists():
        shutil.rmtree(args.state_dir, ignore_errors=True)

    mind = SelfRewritingMind.load(state_file)
    agents_file = inspect_local_file_sensory(_ROOT / "AGENTS.md")
    loader_file = inspect_local_file_sensory(_ROOT / "elysia_loader.py")
    pytest_ini = inspect_local_file_sensory(_ROOT / "pytest.ini")
    snap = ScreenRetinaSensor().capture()

    print("=" * 92)
    print("  1. v1 초기 인지 기관: 실제 파일(AGENTS.md, elysia_loader.py)에 부딪혀 회피 발생")
    print("=" * 92)
    out_agents_v1 = mind.run_organ("primary_perceiver", {"kind": "file", "name": "AGENTS.md", "content": agents_file.get("text_body", "")})
    out_loader_v1 = mind.run_organ("primary_perceiver", {"kind": "file", "name": "elysia_loader.py", "content": loader_file.get("text_body", "")})
    print(f"  [AGENTS.md @ v1]      {out_agents_v1}")
    print(f"  [elysia_loader @ v1]  {out_loader_v1}")

    print("\n" + "=" * 92)
    print("  2. 자기 코드 AST 검사 & 결과 대조: 무엇이 같아야 하고 무엇이 달라져야 하는가")
    print("=" * 92)
    human_ref = {
        "stance": "differentiated_curiosity",
        "action": "analyze_and_differentiate",
    }
    contrast = mind.inspect_and_contrast_self(
        "primary_perceiver",
        {"kind": "file", "name": "AGENTS.md", "content": agents_file.get("text_body", "")},
        out_agents_v1,
        human_ref,
    )
    print(f"  [결과 차이] {contrast.what_differed}")
    print(f"  [자기 코드 원인 위치 (AST)] {contrast.where_in_own_code}")
    print(f"  [유지해야 할 것(must_stay_same)] {contrast.must_stay_same}")
    print(f"  [달라져야 할 것(must_change)]   {contrast.must_change}")

    print("\n" + "=" * 92)
    print("  3. 자기 코드 개변 (v1 -> v2) -> 회귀 불변성 검증 -> 과거 회피 기억 역방향 재해석")
    print("=" * 92)
    rw = mind.attempt_self_rewrite(
        contrast,
        {"kind": "file", "name": "AGENTS.md", "content": agents_file.get("text_body", "")},
        human_ref,
    )
    print(f"  [개변 판정] adopted={rw.adopted} | v{rw.from_version}->v{rw.to_version} | 회귀검증={rw.regression_passed}/{rw.regression_total} | 재해석된 과거={rw.past_episodes_reinterpreted}건")
    for ep in mind.episodes:
        print(f"  [{ep.episode_id} ({ep.percept['name']})] 초기(v1)={ep.initial_output['action']} -> 재해석(v2)={ep.reinterpreted_outputs[-1]['new_output']['category']} | {ep.reinterpreted_outputs[-1]['new_output']['differentiation']}")

    print("\n" + "=" * 92)
    print("  4. 개변된 v2로 새 대상(pytest.ini · 실시간 모니터 화면) 즉시 분별")
    print("=" * 92)
    out_agents_v2 = mind.run_organ("primary_perceiver", {"kind": "file", "name": "AGENTS.md", "content": agents_file.get("text_body", "")})
    out_loader_v2 = mind.run_organ("primary_perceiver", {"kind": "file", "name": "elysia_loader.py", "content": loader_file.get("text_body", "")})
    out_ini_v2 = mind.run_organ("primary_perceiver", {"kind": "file", "name": "pytest.ini", "content": pytest_ini.get("text_body", "")})
    out_screen_v2 = mind.run_organ("primary_perceiver", {"kind": "visual_screen", "name": "desktop_monitor", "content": f"resolution={snap.screen_size} contrast={snap.contrast_std} warm={snap.warm_cool_ratio}"})
    print(f"  [pytest.ini @ v2]     {out_ini_v2['category']} | {out_ini_v2['differentiation']}")
    print(f"  [live_screen @ v2]    {out_screen_v2['category']} | {out_screen_v2['differentiation']}")

    print("\n" + "=" * 92)
    print("  5. 부재한 능력 지각 -> 새 비교 인지 기관(cross_case_comparator v1) 자체 생성·실행")
    print("=" * 92)
    comp_contrast = mind.inspect_and_contrast_self(
        "primary_perceiver",
        {},
        out_agents_v2,
        {"comparison_ok": True},
        requires_new_organ_id="cross_case_comparator",
    )
    rw_new = mind.attempt_self_rewrite(
        comp_contrast,
        {
            "case_a": {"label": "AGENTS.md", "content": agents_file.get("text_body", ""), "output": out_agents_v2},
            "case_b": {"label": "elysia_loader.py", "content": loader_file.get("text_body", ""), "output": out_loader_v2},
        },
        {"comparison_ok": True},
        new_organ_id="cross_case_comparator",
    )
    print(f"  [새 기관 생성] {rw_new.organ_id} v{rw_new.to_version} adopted={rw_new.adopted}")
    comp_out = mind.run_organ(
        "cross_case_comparator",
        {
            "case_a": {"label": "AGENTS.md", "content": agents_file.get("text_body", ""), "output": out_agents_v2},
            "case_b": {"label": "elysia_loader.py", "content": loader_file.get("text_body", ""), "output": out_loader_v2},
        },
    )
    print(f"  [비교 결과] {comp_out['why_different']}")
    print(f"    공통점: {comp_out['same_features'][:6]} | A만: {comp_out['only_in_a'][:4]} | B만: {comp_out['only_in_b'][:4]}")
    mind.save(state_file)
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
