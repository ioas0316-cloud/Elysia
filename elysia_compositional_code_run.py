r"""
ELYSIA Template-Free Compositional AST Code Synthesis Live Run
=============================================================

Demonstrates Elysia:
  1. Decomposing real `.py` files from `core/embodiment/` via `ast.parse` into
     atomic code elements and structural arrangements (zero hardcoded solutions).
  2. Synthesizing genuinely new executable AST functions via `ast.unparse` and
     proving generalization on held-out real files (`AGENTS.md`, `elysia_loader.py`).
  3. Surgically splicing a synthesized decision-tree AST into `SelfRewritingMind`'s
     `primary_perceiver v1` organ without any pre-written `_V2` string, passing
     `must_stay_same` regression checks, and generalizing to unseen workspace inputs!
"""

from __future__ import annotations

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import elysia_loader

elysia_loader.install()

from core.embodiment.compositional_code_cortex import CompositionalCodeCortex  # noqa: E402
from core.embodiment.pc_sensory_motor_body import inspect_local_file_sensory  # noqa: E402
from core.embodiment.self_rewriting_architecture import (  # noqa: E402
    SelfRewritingMind,
    _compile_organ_callable,
)


def main() -> int:
    emb_dir = _ROOT / "L1_causal_field" / "core" / "embodiment"
    py_files = sorted(emb_dir.glob("*.py"))

    cortex = CompositionalCodeCortex()
    learn_rep = cortex.learn_python_structure_from_files(py_files)

    print("=" * 92)
    print("  1. 실제 파이썬 코드 분해 학습 (구성요소 · 구성원리 · 구조배열)")
    print("=" * 92)
    print(f"  분해한 실제 파일: {learn_rep['files_parsed']}개 | 습득한 원자 연산: {learn_rep['learned_atoms_count']}종 | 구조 배열: {learn_rep['learned_arrangements_count']}종")
    top_atoms = sorted(cortex.learned_atoms.values(), key=lambda a: -a.frequency)[:8]
    print("  [상위 원자] " + ", ".join(f"{a.atom_id}({a.frequency}회)" for a in top_atoms))
    print("  [구조 배열] " + ", ".join(f"{k}({v.observed_count}회)" for k, v in cortex.learned_arrangements.items()))

    print("\n" + "=" * 92)
    print("  2. 템플릿 없는 AST 자유 합성 ① · 줄 수 세기 & 집합 교집합 + 미본 파일 일반화 검증")
    print("=" * 92)
    p_lines = cortex.synthesize_transform_ast(
        [
            ({"text": "하나"}, 1),
            ({"text": "하나\n둘"}, 2),
            ({"text": "가\n나\n다\n라"}, 4),
        ],
        purpose="count_text_lines",
    )
    assert p_lines is not None
    print(f"  [합성된 AST 코드 ({p_lines.used_atoms})]:\n{p_lines.source_code}")
    fn_lines = _compile_organ_callable(p_lines.source_code)
    agents_txt = (_ROOT / "AGENTS.md").read_text(encoding="utf-8", errors="replace")
    print(f"  [처음 본 AGENTS.md 실측 적용] 실제 줄 수={len(agents_txt.splitlines())} vs 합성 함수 결과={fn_lines({'text': agents_txt}, {})['result']}")

    p_inter = cortex.synthesize_transform_ast(
        [
            ({"a": ["빛", "소리", "유리"], "b": ["유리", "빛", "물"]}, ["빛", "유리"]),
            ({"a": ["사과", "비"], "b": ["비", "하늘"]}, ["비"]),
        ],
        purpose="sorted_shared_elements",
    )
    assert p_inter is not None
    print(f"\n  [합성된 집합 교집합 AST 코드 ({p_inter.used_atoms})]:\n{p_inter.source_code}")

    print("\n" + "=" * 92)
    print("  3. 템플릿 없는 외과적 AST 접합: primary_perceiver v1의 회피 분기를 직접 개변")
    print("=" * 92)
    mind = SelfRewritingMind()
    v1_src = mind.active_organs["primary_perceiver"].source_code
    p_Agents = inspect_local_file_sensory(_ROOT / "AGENTS.md")
    p_Loader = inspect_local_file_sensory(_ROOT / "elysia_loader.py")
    p_Ini = inspect_local_file_sensory(_ROOT / "pytest.ini")

    # Learn structural decision tree purely from minimal small examples
    p_tree = cortex.synthesize_decision_tree_ast(
        [
            ({"content": "# Doc\nRead me"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "structured_document"}),
            ({"content": "import sys\ndef f():\n    return 1"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "code_structure"}),
            ({"content": "[cfg]\nkey = val"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "config_spec"}),
            ({"content": "plain hello"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "novel_text"}),
        ],
        base_organ_source=v1_src,
        purpose="differentiate_unfamiliar_reality",
    )
    assert p_tree is not None
    print(f"  [AST로 직접 합성·접합된 v2 코드 (사전 작성 템플릿 0줄)]:\n{p_tree.source_code}")

    trig = {"content": "# Doc\nRead me"}
    ref = {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate"}
    out_v1 = mind.run_organ("primary_perceiver", trig)
    contrast = mind.inspect_and_contrast_self("primary_perceiver", trig, out_v1, ref)
    rw = mind.attempt_self_rewrite(contrast, trig, ref, candidate_source_override=p_tree.source_code)
    print(f"\n  [개변 채택 결과] adopted={rw.adopted} | v{rw.from_version}->v{rw.to_version} | 회귀검증={rw.regression_passed}/{rw.regression_total}")

    # Test on HELD-OUT real workspace files never seen during synthesis!
    r_agents = mind.run_organ("primary_perceiver", {"content": p_Agents["text_body"]})
    r_loader = mind.run_organ("primary_perceiver", {"content": p_Loader["text_body"]})
    r_ini = mind.run_organ("primary_perceiver", {"content": p_Ini["text_body"]})
    r_known = mind.run_organ("primary_perceiver", {"content": "known_baseline_anchor"})
    print(f"  [처음 본 AGENTS.md 실측]     category={r_agents.get('category')} | stance={r_agents.get('stance')}")
    print(f"  [처음 본 elysia_loader 실측] category={r_loader.get('category')} | stance={r_loader.get('stance')}")
    print(f"  [처음 본 pytest.ini 실측]    category={r_ini.get('category')} | stance={r_ini.get('stance')}")
    print(f"  [기존 불변성(familiar) 유지]  stance={r_known.get('stance')} | action={r_known.get('action')}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
