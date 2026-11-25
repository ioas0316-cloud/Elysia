from pathlib import Path

import numpy as np

from core.embodiment.external_reality_lines import (
    FileSurfaceLine,
    FilesystemTopologyLine,
    LiveScreenLine,
    PythonRuntimeLine,
    external_world_probe_action,
)
from core.topology.self_weaving_loom import (
    CausalLine,
    LineRefraction,
    PromotedRotorLine,
    SelfWeavingLoom,
    _clamp,
)


def _setup_real_loom(root: Path) -> SelfWeavingLoom:
    loom = SelfWeavingLoom()
    loom.register_line(FileSurfaceLine())
    loom.register_line(PythonRuntimeLine())
    loom.register_line(FilesystemTopologyLine())
    loom.register_action_line(
        "action::external_world_probe",
        lambda ref: external_world_probe_action(ref, workspace_root=root, max_scan_files=16),
    )
    return loom


def test_emergent_poles_without_hardcoded_labels(tmp_path: Path) -> None:
    prose = tmp_path / "notes.md"
    prose.write_text(
        "# Ordinary Journal\n\nToday wind blew softly across open hills without brackets.\n"
        "No formal operators exist inside this simple observational sentence.\n",
        encoding="utf-8",
    )
    code = tmp_path / "engine_core.py"
    code.write_text(
        "def resonate(x: float, y: float) -> float:\n"
        "    if x > y:\n"
        "        return (x * y) + 1.0\n"
        "    return (y - x) * 0.5\n",
        encoding="utf-8",
    )

    loom = _setup_real_loom(tmp_path)
    r1 = loom.encounter("notes.md", {"path": str(prose)})
    r2 = loom.encounter("engine_core.py", {"path": str(code)})

    # Zero labels attached; poles emerge solely from real encountered extremes
    assert "category" not in r1 and "expected" not in r2
    py_line = loom.lines["line::python_runtime"]
    assert py_line.neg_extreme_ref == "notes.md"
    assert py_line.pos_extreme_ref == "engine_core.py"
    assert py_line.neg_extreme_val < py_line.pos_extreme_val


def test_autonomous_cross_weaving_on_divergent_external_reality(tmp_path: Path) -> None:
    # Two files that both look symbol-dense on FileSurfaceLine, yet PythonRuntimeLine separates them
    broken_dense = tmp_path / "pseudo_config.txt"
    broken_dense.write_text(
        "    alpha := [10, 20, 30] -> {x: (1 + 2)}\n"
        "    beta := [40, 50] => @@@\n"
        "    gamma := (((unclosed\n",
        encoding="utf-8",
    )
    valid_dense = tmp_path / "valid_module.py"
    valid_dense.write_text(
        "import math\n\n"
        "def compute_ratio(a: float, b: float) -> float:\n"
        "    if b == 0.0:\n"
        "        return 0.0\n"
        "    return math.tanh((a + 1.0) / b)\n",
        encoding="utf-8",
    )

    loom = _setup_real_loom(tmp_path)
    loom.encounter("pseudo_config.txt", {"path": str(broken_dense)})
    res = loom.encounter("valid_module.py", {"path": str(valid_dense)})

    assert len(loom.crosses) >= 1
    assert any("cross::" in cid for cid in res["events"]["new_crosses"] or loom.crosses.keys())
    # Real surface vs python runtime cross must have emerged autonomously
    assert "cross::line::file_surface__line::python_runtime" in loom.crosses


def test_autonomous_cycle_weave_runs_external_probe_and_records_delta(tmp_path: Path) -> None:
    doc_heavy_py = tmp_path / "story_module.py"
    doc_heavy_py.write_text(
        '"""\n' + ("This long docstring looks like ordinary narrative prose.\n" * 10) + '"""\n\nVALUE = 42\n',
        encoding="utf-8",
    )
    caller_py = tmp_path / "consumer.py"
    caller_py.write_text(
        "from story_module import VALUE\n\ndef use_it() -> int:\n    return VALUE + 1\n",
        encoding="utf-8",
    )

    loom = _setup_real_loom(tmp_path)
    loom.encounter("story_module.py", {"path": str(doc_heavy_py)})
    res = loom.encounter("consumer.py", {"path": str(caller_py)})

    assert len(loom.cycles) >= 1
    executed = [c for c in loom.cycles.values() if c.executions >= 1]
    assert len(executed) >= 1
    # Ensure external probe inspected actual files and recorded concrete consequences
    all_consequences = [item for cy in executed for item in cy.external_consequences]
    assert len(all_consequences) >= 1
    assert any(item["produced_new_evidence"] for item in all_consequences)


def test_recursive_promotion_to_higher_scale_rotor_line_and_scale_descent(tmp_path: Path) -> None:
    loom = _setup_real_loom(tmp_path)

    # Encounter 4 real Python modules that build stable cross mass across FileSurface x PythonRuntime
    for idx in range(4):
        p = tmp_path / f"mod_{idx}.py"
        p.write_text(
            "import math\n\n"
            f"def fn_{idx}(x: float) -> float:\n"
            "    if x > 0.0:\n"
            f"        return math.sin(x + {idx}.0)\n"
            "    return 0.0\n",
            encoding="utf-8",
        )
        loom.encounter(f"mod_{idx}.py", {"path": str(p)})

    # Ensure a resonant cross promoted into a Scale-2 PromotedRotorLine
    promoted = [l for l in loom.lines.values() if isinstance(l, PromotedRotorLine)]
    assert len(promoted) >= 1
    rotor = promoted[0]
    assert rotor.scale >= 2
    assert len(rotor.history) >= 1

    # Now introduce a real SyntaxError-dense artifact that causes strong child divergence on that rotor
    weird = tmp_path / "weird_broken.txt"
    weird.write_text("    [1, 2, 3] :: === ((( broken syntax\n    => {a: b}\n", encoding="utf-8")
    res = loom.encounter("weird_broken.txt", {"path": str(weird)})

    obs = res["observation"]
    assert obs.existence["promoted_rotor_count"] >= 1
    assert obs.existence["max_scale"] >= 2
    assert any(rec.operation == "promote_rotor" for rec in loom.genesis_log)
    # Either scale_descend logged directly or child spread clearly widened under weird_broken.txt
    assert len(res["events"]["scale_descents"]) >= 1 or any(
        r.last_child_spread >= 0.45 for r in promoted
    )


def test_full_four_aspect_observability_and_introspective_self_perception(tmp_path: Path) -> None:
    sample_py = tmp_path / "sample.py"
    sample_py.write_text("def hello(name: str) -> str:\n    return f'hi {name}'\n", encoding="utf-8")

    loom = _setup_real_loom(tmp_path)
    # Also attach LiveScreenLine with synthetic RGB frame
    loom.register_line(LiveScreenLine())
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[:, :32, 0] = 220
    rgb[:, 32:, 2] = 180

    res = loom.encounter("sample_with_screen", {"path": str(sample_py), "screen_rgb": rgb})
    obs = res["observation"]

    assert "lines" in obs.existence
    assert "line::introspective_self" in res["refractions"]
    assert "line_velocities" in obs.motion
    assert "scale_distribution" in obs.constitution
    assert len(obs.causal_why) >= 2
    # Verify 4-axis self-boundary attribution exists on genesis records
    latest_rec = obs.causal_why[-1]
    for axis_key in ("internal_origin", "external_friction", "boundary_ignorance", "criteria_revision"):
        assert latest_rec[axis_key]
    rendered = obs.render_ko()
    assert "1. 존재 (Existence)" in rendered
    assert "2. 움직임 (Motion)" in rendered
    assert "3. 구성 (Constitution)" in rendered
    assert "4. 왜 그렇게 되어지고 있는가 (Causal Why" in rendered


class _ConstantPoleLine(CausalLine):
    def __init__(self, line_id: str, value: float) -> None:
        super().__init__(line_id=line_id, scale=1, description=f"constant {value}")
        self._val = _clamp(value, -1.0, 1.0)

    def refract(self, referent_id: str, raw_referent: dict, step: int) -> LineRefraction:
        return self._record_refraction(referent_id, self._val, {"v": self._val}, (0.33, 0.33, 0.34), {}, step)


def test_physical_tearing_of_unproductive_cross_and_causal_field_sync() -> None:
    loom = SelfWeavingLoom()
    loom.register_line(_ConstantPoleLine("line::pole_neg", -0.95))
    loom.register_line(_ConstantPoleLine("line::pole_pos", +0.95))
    # Action probe returns zero new evidence -> unproductive loop must self-tear
    loom.register_action_line("action::noop_probe", lambda _ref: {"produced_new_evidence": False})

    for step_idx in range(5):
        loom.encounter(f"static_ref_{step_idx}", {"payload": step_idx})

    cross_id = "cross::line::pole_neg__line::pole_pos"
    assert cross_id in loom.crosses
    assert loom.crosses[cross_id].is_torn is True
    assert any(b.is_broken for b in loom.causal_field.beams)
    assert any(rec.operation == "tear_weave" for rec in loom.genesis_log)
