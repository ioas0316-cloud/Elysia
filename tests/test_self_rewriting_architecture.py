from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
L1_ROOT = ROOT / "L1_causal_field"
if str(L1_ROOT) not in sys.path:
    sys.path.insert(0, str(L1_ROOT))

from core.embodiment.self_rewriting_architecture import SelfRewritingMind  # noqa: E402


def test_naive_organ_inspects_own_code_contrasts_outcome_and_rewrites_itself() -> None:
    mind = SelfRewritingMind()
    assert mind.active_organs["primary_perceiver"].version == 1

    # 1. Naive v1 organ avoids unfamiliar input
    p = {"kind": "file", "name": "notes.md", "content": "# Notes\nCausal continuity principles"}
    out_v1 = mind.run_organ("primary_perceiver", p)
    assert out_v1["stance"] == "unfamiliar_noise"
    assert out_v1["action"] == "avoid"

    # 2. Inspect own AST + contrast against human cognitive outcome
    ref = {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate"}
    contrast = mind.inspect_and_contrast_self("primary_perceiver", p, out_v1, ref)
    assert contrast.recommended_evolution == "rewrite_organ"
    assert any("unfamiliar_noise" in ln or "avoid" in ln for ln in contrast.where_in_own_code)
    assert len(contrast.must_stay_same) >= 1
    assert len(contrast.must_change) == 2

    # 3. Synthesize, regression-check, and adopt v2 executable code
    rw = mind.attempt_self_rewrite(contrast, p, ref)
    assert rw.adopted is True
    assert rw.from_version == 1 and rw.to_version == 2
    assert rw.regression_passed >= 1
    assert mind.active_organs["primary_perceiver"].version == 2

    # 4. Run newly rewritten v2 organ -> differentiates cleanly AND respects familiar invariant
    out_code = mind.run_organ("primary_perceiver", {"kind": "file", "name": "mod.py", "content": "def hello():\n    return 1"})
    assert out_code["stance"] == "differentiated_curiosity"
    assert out_code["category"] == "code_structure"

    out_known = mind.run_organ("primary_perceiver", {"content": "known_baseline_anchor"})
    assert out_known["stance"] == "familiar"
    assert out_known["action"] == "keep_observing"


def test_bad_or_unsafe_self_rewrite_is_rejected_and_preserves_working_organ() -> None:
    mind = SelfRewritingMind()
    p = {"kind": "file", "name": "doc.md", "content": "# Title"}
    out_v1 = mind.run_organ("primary_perceiver", p)
    ref = {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate"}
    contrast = mind.inspect_and_contrast_self("primary_perceiver", p, out_v1, ref)

    # 1. Regressing candidate that breaks the `must_stay_same` familiar invariant MUST be rejected
    broken_regression_code = """def process(percept, memory):
    return {
        "stance": "differentiated_curiosity",
        "action": "analyze_and_differentiate",
    }
"""
    rw_reg = mind.attempt_self_rewrite(contrast, p, ref, candidate_source_override=broken_regression_code)
    assert rw_reg.adopted is False
    assert "regression_on_prior_verified_case" in rw_reg.rejection_reason
    assert mind.active_organs["primary_perceiver"].version == 1

    # 2. Unsafe candidate using `import os` MUST be blocked by AST validator
    unsafe_code = """import os
def process(percept, memory):
    return {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate"}
"""
    rw_unsafe = mind.attempt_self_rewrite(contrast, p, ref, candidate_source_override=unsafe_code)
    assert rw_unsafe.adopted is False
    assert "ast_validation_failed" in rw_unsafe.rejection_reason
    assert mind.active_organs["primary_perceiver"].version == 1


def test_spawns_brand_new_cognitive_organ_when_new_capability_is_needed() -> None:
    mind = SelfRewritingMind()
    p = {"kind": "file", "name": "a.py", "content": "def alpha():\n    return 1"}
    out_v1 = mind.run_organ("primary_perceiver", p)
    ref = {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate"}
    mind.attempt_self_rewrite(mind.inspect_and_contrast_self("primary_perceiver", p, out_v1, ref), p, ref)

    out_a = mind.run_organ("primary_perceiver", {"kind": "file", "name": "a.py", "content": "def alpha():\n    return 1"})
    out_b = mind.run_organ("primary_perceiver", {"kind": "file", "name": "b.md", "content": "# Guide\nRead carefully"})

    comp_contrast = mind.inspect_and_contrast_self(
        "primary_perceiver",
        {},
        out_a,
        {"comparison_ok": True},
        requires_new_organ_id="cross_case_comparator",
    )
    assert comp_contrast.recommended_evolution == "create_new_organ"

    pair_percept = {
        "case_a": {"label": "a.py", "content": "def alpha():\n    return 1", "output": out_a},
        "case_b": {"label": "b.md", "content": "# Guide\nRead carefully", "output": out_b},
    }
    rw_new = mind.attempt_self_rewrite(comp_contrast, pair_percept, {"comparison_ok": True}, new_organ_id="cross_case_comparator")
    assert rw_new.adopted is True
    assert "cross_case_comparator" in mind.active_organs

    comp_out = mind.run_organ("cross_case_comparator", pair_percept)
    assert comp_out["comparison_ok"] is True
    assert comp_out["categories_same"] is False
    assert comp_out["category_a"] == "code_structure"
    assert comp_out["category_b"] == "structured_document"


def test_rewritten_organ_retroactively_reinterprets_past_avoided_episodes_and_persists(tmp_path: Path) -> None:
    mind = SelfRewritingMind()
    p1 = {"kind": "file", "name": "readme.md", "content": "# Readme\nOverview text"}
    p2 = {"kind": "file", "name": "core.py", "content": "def run_step():\n    return True"}
    mind.run_organ("primary_perceiver", p1)
    out2 = mind.run_organ("primary_perceiver", p2)

    ref = {"stance": "differentiated_curiosity", "action": "analyze_and_differentiate"}
    contrast = mind.inspect_and_contrast_self("primary_perceiver", p2, out2, ref)
    rw = mind.attempt_self_rewrite(contrast, p2, ref)
    assert rw.adopted is True
    assert rw.past_episodes_reinterpreted == 2

    # Original initial_output ("avoid") is preserved while reinterpreted_outputs holds new understanding
    assert mind.episodes[0].initial_output["action"] == "avoid"
    assert mind.episodes[0].reinterpreted_outputs[0]["new_output"]["category"] == "structured_document"
    assert mind.episodes[1].initial_output["action"] == "avoid"
    assert mind.episodes[1].reinterpreted_outputs[0]["new_output"]["category"] == "code_structure"

    save_p = tmp_path / "mind.json"
    mind.save(save_p)
    reloaded = SelfRewritingMind.load(save_p)
    assert reloaded.active_organs["primary_perceiver"].version == 2
    assert len(reloaded.organ_history) == 1
    assert len(reloaded.episodes) == 2
