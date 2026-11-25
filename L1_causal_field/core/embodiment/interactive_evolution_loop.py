r"""
ELYSIA Universal Counterexample-Driven Interactive Evolution Loop
================================================================

Unifies `CompositionalCodeCortex` (template-free AST learning & synthesis) and
`SelfRewritingMind` (self-inspection, regression guards, hot-swapping, retroactive
reinterpretation) so Elysia continuously evolves her own executable AST organs
(`v1 -> v2 -> v3 -> ...`) whenever interaction with reality or a human partner
exposes a counterexample — across **code, visual pixels, relational dialogue,
or any arbitrary cognitive organ**.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.embodiment.compositional_code_cortex import CompositionalCodeCortex
from core.embodiment.self_rewriting_architecture import (
    CognitiveOrgan,
    SelfRewritingMind,
)


@dataclass
class InteractionEvolutionRecord:
    """Audit record of one live interaction and whether a counterexample evolved Elysia's AST."""

    step: int
    organ_id: str
    feedback_source: str
    counterexample_triggered: bool
    evolved: bool
    from_version: int
    to_version: int
    my_initial_output: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]]
    synthesized_source: str = ""
    rejection_reason: str = ""
    past_reinterpreted: int = 0


def _minimal_organ_ast_source(default_output: Dict[str, Any]) -> str:
    ret_dict = ast.Dict(
        keys=[ast.Constant(value=k) for k in default_output.keys()],
        values=[ast.Constant(value=v) for v in default_output.values()],
    )
    fn = ast.FunctionDef(
        name="process",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="percept"), ast.arg(arg="memory")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Return(value=ret_dict)],
        decorator_list=[],
    )
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    return ast.unparse(mod)


class UniversalInteractiveEvolutionLoop:
    """
    Domain-agnostic interactive evolution loop:
    runs current AST organ -> compares against reality/human feedback ->
    merges counterexamples -> re-synthesizes AST from learned Python atoms ->
    verifies regressions -> hot-swaps -> reinterprets past episodes.
    """

    def __init__(self) -> None:
        self.cortex = CompositionalCodeCortex()
        self.mind = SelfRewritingMind()
        self.training_examples: Dict[str, List[Dict[str, Any]]] = {}
        self.history: List[InteractionEvolutionRecord] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cortex": self.cortex.to_dict(),
            "mind": self.mind.to_dict(),
            "training_examples": copy.deepcopy(self.training_examples),
            "history": [asdict(h) for h in self.history],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "UniversalInteractiveEvolutionLoop":
        obj = cls()
        if raw.get("cortex"):
            obj.cortex = CompositionalCodeCortex.from_dict(raw["cortex"])
        if raw.get("mind"):
            obj.mind = SelfRewritingMind.from_dict(raw["mind"])
        obj.training_examples = copy.deepcopy(raw.get("training_examples") or {})
        obj.history = [InteractionEvolutionRecord(**h) for h in (raw.get("history") or [])]
        return obj

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "UniversalInteractiveEvolutionLoop":
        p = Path(path)
        if not p.is_file():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def bootstrap_from_python_files(self, paths: Sequence[Path | str]) -> Dict[str, Any]:
        return self.cortex.learn_python_structure_from_files(paths)

    def ensure_organ(
        self,
        organ_id: str,
        *,
        default_output: Optional[Dict[str, Any]] = None,
    ) -> CognitiveOrgan:
        existing = self.mind.active_organs.get(organ_id)
        if existing is not None:
            return existing
        out_dict = dict(default_output or {"category": "unknown", "action": "observe"})
        src = _minimal_organ_ast_source(out_dict)
        org = CognitiveOrgan(
            organ_id=organ_id,
            version=1,
            parent_version=0,
            purpose=f"Interactive evolving cognitive organ for '{organ_id}'",
            source_code=src,
            reason_for_change="Initial unspecialized AST seed",
        )
        self.mind.active_organs[organ_id] = org
        return org

    def _upsert_training_example(
        self,
        organ_id: str,
        percept: Dict[str, Any],
        expected: Dict[str, Any],
    ) -> None:
        ex_list = self.training_examples.setdefault(organ_id, [])
        p_sig = json.dumps(percept, sort_keys=True, ensure_ascii=False)
        for item in ex_list:
            if json.dumps(item["percept"], sort_keys=True, ensure_ascii=False) == p_sig:
                item["expected"] = copy.deepcopy(expected)
                return
        ex_list.append({"percept": copy.deepcopy(percept), "expected": copy.deepcopy(expected)})

        # Keep SelfRewritingMind verified_cases synchronized if percept was already there
        for vc in self.mind.verified_cases:
            if vc["organ_id"] == organ_id and json.dumps(vc["percept"], sort_keys=True, ensure_ascii=False) == p_sig:
                vc["expected_checks"] = copy.deepcopy(expected)

    def _matches_expected(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        return all(actual.get(k) == v for k, v in expected.items())

    def interact_and_evolve(
        self,
        organ_id: str,
        percept: Dict[str, Any],
        *,
        feedback_expected: Optional[Dict[str, Any]] = None,
        feedback_source: str = "external_interaction",
        reality_verifier: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
        default_output: Optional[Dict[str, Any]] = None,
    ) -> InteractionEvolutionRecord:
        """
        Executes the current AST organ on `percept`, checks against human feedback or
        direct reality verification, and on ANY counterexample re-synthesizes a stronger
        AST across all accumulated experience without templates.
        """
        org = self.ensure_organ(organ_id, default_output=default_output)
        from_ver = org.version
        my_out = self.mind.run_organ(organ_id, percept)

        expected = (
            copy.deepcopy(feedback_expected)
            if feedback_expected is not None
            else (reality_verifier(percept, my_out) if reality_verifier is not None else None)
        )

        if expected is None:
            rec = InteractionEvolutionRecord(
                step=len(self.history) + 1,
                organ_id=organ_id,
                feedback_source=feedback_source,
                counterexample_triggered=False,
                evolved=False,
                from_version=from_ver,
                to_version=from_ver,
                my_initial_output=my_out,
                expected_output=None,
            )
            self.history.append(rec)
            return rec

        self._upsert_training_example(organ_id, percept, expected)
        if self._matches_expected(my_out, expected):
            rec = InteractionEvolutionRecord(
                step=len(self.history) + 1,
                organ_id=organ_id,
                feedback_source=feedback_source,
                counterexample_triggered=False,
                evolved=False,
                from_version=from_ver,
                to_version=from_ver,
                my_initial_output=my_out,
                expected_output=expected,
            )
            self.history.append(rec)
            return rec

        # Counterexample encountered -> synthesize new AST over ALL verified examples!
        ex_pairs = [
            (copy.deepcopy(it["percept"]), copy.deepcopy(it["expected"]))
            for it in self.training_examples.get(organ_id, [])
        ]
        base_src = org.source_code if organ_id == "primary_perceiver" else ""
        prog = self.cortex.synthesize_decision_tree_ast(
            ex_pairs,
            base_organ_source=base_src,
            purpose=f"evolve_{organ_id}_from_{feedback_source}",
        )
        if prog is None:
            rec = InteractionEvolutionRecord(
                step=len(self.history) + 1,
                organ_id=organ_id,
                feedback_source=feedback_source,
                counterexample_triggered=True,
                evolved=False,
                from_version=from_ver,
                to_version=from_ver,
                my_initial_output=my_out,
                expected_output=expected,
                rejection_reason="no_consistent_ast_predicate_found",
            )
            self.history.append(rec)
            return rec

        contrast = self.mind.inspect_and_contrast_self(organ_id, percept, my_out, expected)
        rw = self.mind.attempt_self_rewrite(
            contrast,
            percept,
            expected,
            candidate_source_override=prog.source_code,
        )
        rec = InteractionEvolutionRecord(
            step=len(self.history) + 1,
            organ_id=organ_id,
            feedback_source=feedback_source,
            counterexample_triggered=True,
            evolved=rw.adopted,
            from_version=rw.from_version,
            to_version=rw.to_version if rw.adopted else rw.from_version,
            my_initial_output=my_out,
            expected_output=expected,
            synthesized_source=prog.source_code if rw.adopted else "",
            rejection_reason=rw.rejection_reason,
            past_reinterpreted=rw.past_episodes_reinterpreted,
        )
        self.history.append(rec)
        return rec
