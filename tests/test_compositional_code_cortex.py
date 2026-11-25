from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
L1_ROOT = ROOT / "L1_causal_field"
if str(L1_ROOT) not in sys.path:
    sys.path.insert(0, str(L1_ROOT))

from core.embodiment.compositional_code_cortex import CompositionalCodeCortex  # noqa: E402
from core.embodiment.self_rewriting_architecture import (  # noqa: E402
    SelfRewritingMind,
    _compile_organ_callable,
)


def test_learns_real_python_atoms_and_arrangements_from_workspace_source_files() -> None:
    cortex = CompositionalCodeCortex()
    # Before learning any files, no atoms exist and synthesis cannot fake code
    assert len(cortex.learned_atoms) == 0
    assert cortex.synthesize_transform_ast([({"text": "a\nb"}, 2)]) is None

    emb_dir = L1_ROOT / "core" / "embodiment"
    rep = cortex.learn_python_structure_from_files(sorted(emb_dir.glob("*.py")))
    assert rep["files_parsed"] >= 5
    assert "call::len" in cortex.learned_atoms
    assert "method::splitlines" in cortex.learned_atoms
    assert "guard_return" in cortex.learned_arrangements
    assert "if_else_split" in cortex.learned_arrangements


def test_synthesizes_novel_ast_transforms_from_harvested_atoms_and_generalizes_to_unseen_inputs() -> None:
    cortex = CompositionalCodeCortex()
    emb_dir = L1_ROOT / "core" / "embodiment"
    cortex.learn_python_structure_from_files(sorted(emb_dir.glob("*.py")))

    # 1. Line count transform synthesized purely from AST atoms
    p_lines = cortex.synthesize_transform_ast(
        [
            ({"text": "one"}, 1),
            ({"text": "one\ntwo"}, 2),
            ({"text": "a\nb\nc\nd"}, 4),
        ],
        purpose="count_lines",
    )
    assert p_lines is not None
    assert "splitlines" in p_lines.source_code
    assert "len(" in p_lines.source_code
    fn_lines = _compile_organ_callable(p_lines.source_code)
    assert fn_lines({"text": "x\ny\nz\nw\nv"}, {})["result"] == 5

    # 2. Sorted set intersection synthesized purely from AST atoms
    p_inter = cortex.synthesize_transform_ast(
        [
            ({"a": ["x", "y", "z"], "b": ["z", "a", "x"]}, ["x", "z"]),
            ({"a": ["1", "2"], "b": ["2", "3"]}, ["2"]),
        ],
        purpose="shared_sorted",
    )
    assert p_inter is not None
    fn_inter = _compile_organ_callable(p_inter.source_code)
    assert fn_inter({"a": ["p", "q", "r"], "b": ["r", "q", "s"]}, {})["result"] == ["q", "r"]


def test_synthesizes_conditional_ast_and_surgically_mutates_existing_organ_without_templates() -> None:
    cortex = CompositionalCodeCortex()
    emb_dir = L1_ROOT / "core" / "embodiment"
    cortex.learn_python_structure_from_files(sorted(emb_dir.glob("*.py")))

    mind = SelfRewritingMind()
    v1_src = mind.active_organs["primary_perceiver"].source_code

    p_tree = cortex.synthesize_decision_tree_ast(
        [
            ({"content": "# Doc\nhello"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "structured_document"}),
            ({"content": "import os\ndef f():\n    return 1"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "code_structure"}),
            ({"content": "[cfg]\na = 1"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "config_spec"}),
            ({"content": "hello world"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "novel_text"}),
        ],
        base_organ_source=v1_src,
        purpose="surgical_upgrade",
    )
    assert p_tree is not None
    assert "familiar" in p_tree.source_code  # preserved from v1 surgical splice!

    trig = {"content": "# Doc\nhello"}
    ref = {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate"}
    out_v1 = mind.run_organ("primary_perceiver", trig)
    contrast = mind.inspect_and_contrast_self("primary_perceiver", trig, out_v1, ref)
    rw = mind.attempt_self_rewrite(contrast, trig, ref, candidate_source_override=p_tree.source_code)
    assert rw.adopted is True
    assert rw.regression_passed >= 1

    # Generalize to unseen held-out workspace files & verify familiar invariant stays intact
    agents_txt = (ROOT / "AGENTS.md").read_text(encoding="utf-8", errors="replace")
    ini_txt = (ROOT / "pytest.ini").read_text(encoding="utf-8", errors="replace")
    assert mind.run_organ("primary_perceiver", {"content": agents_txt})["category"] == "structured_document"
    assert mind.run_organ("primary_perceiver", {"content": ini_txt})["category"] == "config_spec"
    assert mind.run_organ("primary_perceiver", {"content": "known_baseline_anchor"})["stance"] == "familiar"


def test_compositional_cortex_persistence_roundtrip(tmp_path: Path) -> None:
    cortex = CompositionalCodeCortex()
    cortex.learn_python_structure_from_files([L1_ROOT / "core" / "embodiment" / "compositional_code_cortex.py"])
    cortex.synthesize_transform_ast([({"text": "a\nb"}, 2)], purpose="lines")

    save_p = tmp_path / "cortex.json"
    cortex.save(save_p)
    reloaded = CompositionalCodeCortex.load(save_p)
    assert len(reloaded.learned_atoms) == len(cortex.learned_atoms)
    assert len(reloaded.synthesized_history) == 1
