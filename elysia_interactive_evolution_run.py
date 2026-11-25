r"""
ELYSIA Cross-Domain Counterexample-Driven Interactive Evolution Live Run
========================================================================

Shows ONE template-free `UniversalInteractiveEvolutionLoop` evolving AST organs
across 3 completely distinct domains via counterexamples & reality friction:
  1. Code/Files (`primary_perceiver`: v1 -> v2 -> v3 via `elysia_loader.py` counterexample)
  2. Visual Pixels (`visual_scene_interpreter`: numeric threshold AST + live Windows monitor)
  3. Relational Dialogue (`dialogue_framing_organ`: sensory-cue conditional AST)
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import elysia_loader

elysia_loader.install()

from core.embodiment.interactive_evolution_loop import UniversalInteractiveEvolutionLoop  # noqa: E402
from core.embodiment.pc_sensory_motor_body import (  # noqa: E402
    ScreenRetinaSensor,
    inspect_local_file_sensory,
)


def _python_reality_verifier(percept: Dict[str, Any], my_out: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Direct non-human reality ground truth: if `ast.parse` parses it, it IS Python code."""
    txt = str(percept.get("content", ""))
    try:
        tree = ast.parse(txt)
        if any(isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef)) for n in tree.body):
            return {
                "stance": "differentiated_curiosity",
                "action": "analyze_and_differentiate",
                "category": "code_structure",
            }
    except SyntaxError:
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Elysia Cross-Domain Interactive Evolution Run")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=_ROOT / "data" / "elysia_interactive_evolution_state",
        help="State persistence directory",
    )
    parser.add_argument("--fresh", action="store_true", help="Reset state first")
    args = parser.parse_args()

    state_file = args.state_dir / "interactive_evolution.json"
    if args.fresh and args.state_dir.exists():
        shutil.rmtree(args.state_dir, ignore_errors=True)

    loop = UniversalInteractiveEvolutionLoop.load(state_file)
    py_files = sorted((_ROOT / "L1_causal_field" / "core" / "embodiment").glob("*.py"))
    boot = loop.bootstrap_from_python_files(py_files)

    print("=" * 92)
    print(f"  0. 코드 구성요소 습득: {boot['files_parsed']}개 파일 -> 원자 {boot['learned_atoms_count']}종")
    print("=" * 92)

    print("\n" + "=" * 92)
    print("  1. 도메인 ① 코드/문서 ('primary_perceiver'): v1 -> v2 -> 반례(elysia_loader.py) -> v3")
    print("=" * 92)
    seed_pairs = [
        ({"content": "# Doc\nhello"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "structured_document"}),
        ({"content": "import os\ndef f():\n    return 1"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "code_structure"}),
        ({"content": "[cfg]\na = 1"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "config_spec"}),
        ({"content": "plain hello"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "novel_text"}),
    ]
    for idx, (p, exp) in enumerate(seed_pairs, start=1):
        rec = loop.interact_and_evolve("primary_perceiver", p, feedback_expected=exp, feedback_source=f"human_seed_{idx}")
        if rec.evolved:
            print(f"  [초기 진화] v{rec.from_version}->v{rec.to_version} | 과거재해석={rec.past_reinterpreted}건")

    agents_txt = inspect_local_file_sensory(_ROOT / "AGENTS.md")["text_body"]
    loader_txt = inspect_local_file_sensory(_ROOT / "elysia_loader.py")["text_body"]
    ini_txt = inspect_local_file_sensory(_ROOT / "pytest.ini")["text_body"]
    life_txt = inspect_local_file_sensory(_ROOT / "elysia_autonomous_life.py")["text_body"]

    print(f"  [v2에서 elysia_loader.py 첫 접촉 전 출력] {loop.mind.run_organ('primary_perceiver', {'content': loader_txt}, record_episode=False)}")
    rec_ce = loop.interact_and_evolve(
        "primary_perceiver",
        {"content": loader_txt},
        feedback_source="reality_ast_parse",
        reality_verifier=_python_reality_verifier,
    )
    print(f"  [반례 진화!] counterexample={rec_ce.counterexample_triggered} | evolved={rec_ce.evolved} | v{rec_ce.from_version}->v{rec_ce.to_version} | 과거재해석={rec_ce.past_reinterpreted}건")
    print(f"  [합성된 v3 AST 코드]:\n{rec_ce.synthesized_source}")
    print("  [처음 본 파일들 v3 실측 검증]")
    print(f"    - AGENTS.md                 -> {loop.mind.run_organ('primary_perceiver', {'content': agents_txt}, record_episode=False).get('category')}")
    print(f"    - pytest.ini                -> {loop.mind.run_organ('primary_perceiver', {'content': ini_txt}, record_episode=False).get('category')}")
    print(f"    - elysia_loader.py          -> {loop.mind.run_organ('primary_perceiver', {'content': loader_txt}, record_episode=False).get('category')}")
    print(f"    - elysia_autonomous_life.py -> {loop.mind.run_organ('primary_perceiver', {'content': life_txt}, record_episode=False).get('category')}")

    print("\n" + "=" * 92)
    print("  2. 도메인 ② 시각·물리 수치 ('visual_scene_interpreter'): 수치 비교 임계치 AST 자율 합성")
    print("=" * 92)
    vis_cases = [
        ({"contrast_std": 0.01, "motion_flux": 0.0}, {"scene_state": "blank_static", "attention": "rest"}),
        ({"contrast_std": 0.02, "motion_flux": 0.0}, {"scene_state": "blank_static", "attention": "rest"}),
        ({"contrast_std": 0.09, "motion_flux": 0.0}, {"scene_state": "textured_still", "attention": "inspect_detail"}),
        ({"contrast_std": 0.14, "motion_flux": 0.01}, {"scene_state": "textured_still", "attention": "inspect_detail"}),
        ({"contrast_std": 0.08, "motion_flux": 0.18}, {"scene_state": "active_motion", "attention": "track_change"}),
    ]
    last_vis = None
    for vp, vexp in vis_cases:
        last_vis = loop.interact_and_evolve("visual_scene_interpreter", vp, feedback_expected=vexp, feedback_source="visual_grounding")
    print(f"  [시각 AST 진화 결과] v{loop.mind.active_organs['visual_scene_interpreter'].version}\n{loop.mind.active_organs['visual_scene_interpreter'].source_code}")
    snap = ScreenRetinaSensor().capture()
    live_vis_out = loop.mind.run_organ(
        "visual_scene_interpreter",
        {"contrast_std": snap.contrast_std, "motion_flux": snap.motion_flux},
        record_episode=False,
    )
    print(f"  [실제 Windows 화면 실측 ({snap.screen_size}, 대비={snap.contrast_std:.3f})] -> {live_vis_out}")

    print("\n" + "=" * 92)
    print("  3. 도메인 ③ 관계적 대화 ('dialogue_framing_organ'): 감각 단서 대화 틀 AST 자율 합성")
    print("=" * 92)
    dlg_cases = [
        ({"referent_cue": "auditory", "intent": "point_and_name"}, {"frame": "이건 {REFERENT}{이/가} 들리는 소리야", "mode": "auditory_frame"}),
        ({"referent_cue": "visual", "intent": "point_and_name"}, {"frame": "저거 봐, {REFERENT}{이야/야}", "mode": "visual_frame"}),
        ({"referent_cue": "visual", "intent": "ask_meaning"}, {"frame": "{REFERENT}{이/가} 무슨 뜻이야?", "mode": "inquiry_frame"}),
    ]
    for dp, dexp in dlg_cases:
        loop.interact_and_evolve("dialogue_framing_organ", dp, feedback_expected=dexp, feedback_source="relational_dialogue")
    print(f"  [대화 틀 AST 진화 결과] v{loop.mind.active_organs['dialogue_framing_organ'].version}\n{loop.mind.active_organs['dialogue_framing_organ'].source_code}")

    loop.save(state_file)
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
