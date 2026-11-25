from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
L1_ROOT = ROOT / "L1_causal_field"
if str(L1_ROOT) not in sys.path:
    sys.path.insert(0, str(L1_ROOT))

from core.embodiment.interactive_evolution_loop import UniversalInteractiveEvolutionLoop  # noqa: E402


def _python_ast_verifier(percept, my_out):
    try:
        tree = ast.parse(str(percept.get("content", "")))
        if any(isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)) for n in tree.body):
            return {
                "stance": "differentiated_curiosity",
                "action": "analyze_and_differentiate",
                "category": "code_structure",
            }
    except SyntaxError:
        pass
    return None


def test_counterexample_on_elysia_loader_evolves_v2_to_v3_and_generalizes_to_unseen_py_file() -> None:
    loop = UniversalInteractiveEvolutionLoop()
    loop.bootstrap_from_python_files(sorted((L1_ROOT / "core" / "embodiment").glob("*.py")))

    seed_pairs = [
        ({"content": "# Doc\nhello"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "structured_document"}),
        ({"content": "import os\ndef f():\n    return 1"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "code_structure"}),
        ({"content": "[cfg]\na = 1"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "config_spec"}),
        ({"content": "plain hello"}, {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate", "category": "novel_text"}),
    ]
    for p, exp in seed_pairs:
        loop.interact_and_evolve("primary_perceiver", p, feedback_expected=exp, feedback_source="seed")

    v_before = loop.mind.active_organs["primary_perceiver"].version
    assert v_before >= 2

    # Docstring-first Python file acts as genuine counterexample to `startswith('i')`
    loader_txt = (ROOT / "elysia_loader.py").read_text(encoding="utf-8", errors="replace")[:2500]
    rec = loop.interact_and_evolve(
        "primary_perceiver",
        {"content": loader_txt},
        feedback_source="reality_ast_parse",
        reality_verifier=_python_ast_verifier,
    )
    assert rec.counterexample_triggered is True
    assert rec.evolved is True
    assert rec.from_version == v_before and rec.to_version == v_before + 1
    assert rec.past_reinterpreted >= 1

    # Verify v3 generalizes to ALL unseen files including `elysia_autonomous_life.py` & keeps `familiar` invariant
    agents_txt = (ROOT / "AGENTS.md").read_text(encoding="utf-8", errors="replace")[:2500]
    ini_txt = (ROOT / "pytest.ini").read_text(encoding="utf-8", errors="replace")[:2500]
    life_txt = (ROOT / "elysia_autonomous_life.py").read_text(encoding="utf-8", errors="replace")[:2500]

    assert loop.mind.run_organ("primary_perceiver", {"content": agents_txt})["category"] == "structured_document"
    assert loop.mind.run_organ("primary_perceiver", {"content": ini_txt})["category"] == "config_spec"
    assert loop.mind.run_organ("primary_perceiver", {"content": loader_txt})["category"] == "code_structure"
    assert loop.mind.run_organ("primary_perceiver", {"content": life_txt})["category"] == "code_structure"
    assert loop.mind.run_organ("primary_perceiver", {"content": "known_baseline_anchor"})["stance"] == "familiar"


def test_numeric_visual_counterexamples_synthesize_threshold_ast_organ() -> None:
    loop = UniversalInteractiveEvolutionLoop()
    loop.bootstrap_from_python_files(sorted((L1_ROOT / "core" / "embodiment").glob("*.py")))

    cases = [
        ({"contrast_std": 0.01, "motion_flux": 0.0}, {"scene_state": "blank_static", "attention": "rest"}),
        ({"contrast_std": 0.08, "motion_flux": 0.0}, {"scene_state": "textured_still", "attention": "inspect_detail"}),
        ({"contrast_std": 0.07, "motion_flux": 0.20}, {"scene_state": "active_motion", "attention": "track_change"}),
    ]
    for p, exp in cases:
        loop.interact_and_evolve("visual_scene_interpreter", p, feedback_expected=exp, feedback_source="visual")

    vis_org = loop.mind.active_organs["visual_scene_interpreter"]
    assert vis_org.version >= 2
    assert ">" in vis_org.source_code
    assert loop.mind.run_organ("visual_scene_interpreter", {"contrast_std": 0.005, "motion_flux": 0.0})["scene_state"] == "blank_static"
    assert loop.mind.run_organ("visual_scene_interpreter", {"contrast_std": 0.15, "motion_flux": 0.01})["scene_state"] == "textured_still"
    assert loop.mind.run_organ("visual_scene_interpreter", {"contrast_std": 0.12, "motion_flux": 0.35})["scene_state"] == "active_motion"


def test_relational_dialogue_counterexamples_evolve_framing_organ() -> None:
    loop = UniversalInteractiveEvolutionLoop()
    loop.bootstrap_from_python_files(sorted((L1_ROOT / "core" / "embodiment").glob("*.py")))

    cases = [
        ({"referent_cue": "auditory", "intent": "point_and_name"}, {"frame": "이건 {REFERENT}{이/가} 들리는 소리야", "mode": "auditory_frame"}),
        ({"referent_cue": "visual", "intent": "point_and_name"}, {"frame": "저거 봐, {REFERENT}{이야/야}", "mode": "visual_frame"}),
        ({"referent_cue": "visual", "intent": "ask_meaning"}, {"frame": "{REFERENT}{이/가} 무슨 뜻이야?", "mode": "inquiry_frame"}),
    ]
    for p, exp in cases:
        loop.interact_and_evolve("dialogue_framing_organ", p, feedback_expected=exp, feedback_source="dialogue")

    assert loop.mind.run_organ("dialogue_framing_organ", {"referent_cue": "auditory", "intent": "point_and_name"})["mode"] == "auditory_frame"
    assert loop.mind.run_organ("dialogue_framing_organ", {"referent_cue": "visual", "intent": "point_and_name"})["mode"] == "visual_frame"
    assert loop.mind.run_organ("dialogue_framing_organ", {"referent_cue": "visual", "intent": "ask_meaning"})["mode"] == "inquiry_frame"


def test_interactive_evolution_loop_persistence_roundtrip(tmp_path: Path) -> None:
    loop = UniversalInteractiveEvolutionLoop()
    loop.bootstrap_from_python_files([L1_ROOT / "core" / "embodiment" / "compositional_code_cortex.py"])
    loop.interact_and_evolve(
        "visual_scene_interpreter",
        {"contrast_std": 0.10, "motion_flux": 0.0},
        feedback_expected={"scene_state": "textured", "attention": "look"},
    )
    save_p = tmp_path / "evo_loop.json"
    loop.save(save_p)

    reloaded = UniversalInteractiveEvolutionLoop.load(save_p)
    assert len(reloaded.cortex.learned_atoms) == len(loop.cortex.learned_atoms)
    assert "visual_scene_interpreter" in reloaded.mind.active_organs
    assert reloaded.mind.run_organ("visual_scene_interpreter", {"contrast_std": 0.25, "motion_flux": 0.0})["scene_state"] == "textured"
