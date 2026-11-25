from pathlib import Path

import numpy as np

from core.embodiment.external_reality_lines import (
    FileSurfaceLine,
    FilesystemTopologyLine,
    PythonRuntimeLine,
)
from core.topology.prism_brush_world_system import WholeSystemPrismRotor
from core.topology.self_weaving_loom import PromotedRotorLine, SelfWeavingLoom


def test_ancestor_repulsion_prevents_redundant_s4_explosion(tmp_path: Path) -> None:
    loom = SelfWeavingLoom()
    loom.register_line(FileSurfaceLine())
    loom.register_line(PythonRuntimeLine())
    loom.register_line(FilesystemTopologyLine())

    # Encounter 6 real distinct workspace files
    sample_files = [
        ("a.py", "import math\n\ndef f(x):\n    return math.sin(x) + 1.0\n"),
        ("b.py", "def g(y):\n    for i in range(4):\n        y += i\n    return y\n"),
        ("c.py", "class Box:\n    def __init__(self, v):\n        self.v = v\n"),
        ("d.md", "# Notes\n\nSimple plain prose paragraph with no Python operators.\n"),
        ("e.ini", "[section]\nkey = value\nother = 123\n"),
        ("f.py", "def h(a, b):\n    return a * b - 2\n"),
    ]
    for name, content in sample_files:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        loom.encounter(name, {"path": str(p)})

    promoted = [l for l in loom.lines.values() if isinstance(l, PromotedRotorLine)]
    # Subset ancestor repulsion guarantees bounded, non-degenerate rotors with no child subset reuse
    for cross in loom.crosses.values():
        anc_a = loom._base_ancestors(cross.line_a_id)
        anc_b = loom._base_ancestors(cross.line_b_id)
        assert not (anc_a <= anc_b or anc_b <= anc_a)
    assert len(promoted) <= 12


def test_external_invariants_remain_immutable_anchors(tmp_path: Path) -> None:
    broken_file = tmp_path / "broken_notes.txt"
    broken_file.write_text("this is ((( unclosed natural phrasing", encoding="utf-8")

    system = WholeSystemPrismRotor(workspace_root=tmp_path)
    rep = system.experience_world_event("broken_notes.txt", {"path": str(broken_file)})

    by_id = {chk["invariant_id"]: chk for chk in rep.invariant_checks}
    assert by_id["inv::python_grammar"]["immutable_by_system"] is True
    assert by_id["inv::python_grammar"]["accepted"] is False
    assert by_id["inv::os_filesystem"]["accepted"] is True


def test_prism_refraction_and_brush_outward_projection_across_scales(tmp_path: Path) -> None:
    mod_file = tmp_path / "rain_reflection.py"
    mod_file.write_text(
        "def ripple(drop: float) -> float:\n    return drop * 0.5\n",
        encoding="utf-8",
    )
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    rgb[:, :24, 2] = 200

    system = WholeSystemPrismRotor(workspace_root=tmp_path)
    rep = system.experience_world_event(
        "rain_moment",
        {
            "path": str(mod_file),
            "screen_rgb": rgb,
            "human_utterance": "창밖 빗방울과 코드 구조를 같이 엮어서 통합적으로 보자",
        },
    )

    scales = {p.scale_focus for p in rep.prisms}
    assert "micro" in scales
    assert "meso_static" in scales
    assert "macro_language" in scales
    assert rep.brush_projection.outward_expression_ko
    assert len(rep.brush_projection.external_room_for_change) >= 1


def test_convergence_attraction_repulsion_expansion_cycle_forms_boundaries(tmp_path: Path) -> None:
    doc = tmp_path / "philosophy.md"
    doc.write_text("# Essence\n\nCode conventions alone cannot capture human silence.\n", encoding="utf-8")

    system = WholeSystemPrismRotor(workspace_root=tmp_path)
    rep = system.experience_world_event("philosophy.md", {"path": str(doc)})

    assert len(rep.convergence_seed.repulsion_boundaries) >= 1
    # Repulsion boundary explicitly cites python compiler limit/rejection rather than forcing silence into code
    joined = " | ".join(rep.convergence_seed.repulsion_boundaries)
    assert "불변식 거부" in joined or "한계" in joined


def test_principle_habit_observer_detects_repetition_and_points_outside(tmp_path: Path) -> None:
    system = WholeSystemPrismRotor(workspace_root=tmp_path)

    # Repeatedly collide a fixed text artifact + human pushback against immutable invariants
    for idx in range(1, 5):
        f = tmp_path / f"attempt_{idx}.txt"
        f.write_text(f"attempt {idx} ::: broken syntax (((\n", encoding="utf-8")
        rep = system.experience_world_event(
            f"pushback_{idx}",
            {
                "path": str(f),
                "human_utterance": "아니, 네가 가진 원리 안에서만 반복하지 말고 밖에서 변화의 여지를 찾아봐.",
            },
        )

    assert rep.habit_perception.is_repeating_principle is True
    assert rep.habit_perception.repetition_count >= 2
    assert rep.habit_perception.external_invariant_anchor in ("inv::human_utterance", "inv::python_grammar")
    assert "답습" in rep.habit_perception.why_stuck_ko
    assert len(rep.habit_perception.external_room_for_change) >= 1
    assert rep.brush_projection.brush_mode == "outward_question_to_human"
