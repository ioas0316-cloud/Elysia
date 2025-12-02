# [Genesis: 2025-12-02] Purified by Elysia
"""
MathCortex

Purpose: Provide a simple, interpretable proof engine for basic arithmetic
equalities with human-readable steps. This is a foundation for richer
mathematical reasoning while keeping safety and clarity.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import List, Optional, Dict, Any

# Optional symbolic support
try:
    import sympy as sp
except Exception:
    sp = None


_SAFE_EXPR = re.compile(r"^[0-9\s\+\-\*\/\(\)\.]+$")


@dataclass
class ProofStep:
    index: int
    action: str
    detail: str
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Proof:
    statement: str
    steps: List[ProofStep]
    valid: bool
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "steps": [s.to_dict() for s in self.steps],
            "valid": self.valid,
            "verdict": self.verdict,
        }


class MathCortex:
    """
    Role: Agent Sophia core for basic math verification with explainable steps.
    """

    def _safe_eval(self, expr: str) -> float:
        expr = expr.strip()
        if not _SAFE_EXPR.match(expr):
            raise ValueError("Unsafe characters in expression")
        # Evaluate in a restricted environment
        return float(eval(expr, {"__builtins__": {}}, {}))

    def _parse_equality(self, statement: str) -> Optional[tuple[str, str]]:
        m = re.match(r"^\s*(.+?)\s*(?:=|==)\s*(.+?)\s*$", statement)
        if not m:
            return None
        return m.group(1), m.group(2)

    def prove_equality(self, lhs: str, rhs: str) -> Proof:
        steps: List[ProofStep] = []
        idx = 1

        steps.append(ProofStep(idx, "parse", f"Left expression parsed: {lhs}")); idx += 1
        steps.append(ProofStep(idx, "parse", f"Right expression parsed: {rhs}")); idx += 1

        try:
            left_val = self._safe_eval(lhs)
            steps.append(ProofStep(idx, "evaluate", f"Evaluate LHS: {lhs}", result=str(left_val)))
            idx += 1
        except Exception as e:
            steps.append(ProofStep(idx, "error", f"Failed to evaluate LHS: {e}"))
            return Proof(f"{lhs} = {rhs}", steps, False, "Failed to evaluate LHS")

        try:
            right_val = self._safe_eval(rhs)
            steps.append(ProofStep(idx, "evaluate", f"Evaluate RHS: {rhs}", result=str(right_val)))
            idx += 1
        except Exception as e:
            steps.append(ProofStep(idx, "error", f"Failed to evaluate RHS: {e}"))
            return Proof(f"{lhs} = {rhs}", steps, False, "Failed to evaluate RHS")

        equal = abs(left_val - right_val) < 1e-9
        steps.append(ProofStep(idx, "compare", f"Compare values {left_val} vs {right_val}", result=str(equal)))
        verdict = "Equality holds" if equal else "Equality does not hold"
        return Proof(f"{lhs} = {rhs}", steps, equal, verdict)

    def verify(self, statement: str) -> Proof:
        parsed = self._parse_equality(statement)
        if not parsed:
            return Proof(statement, [ProofStep(1, "error", "Not an equality statement")], False, "Parse error")
        lhs, rhs = parsed
        return self.prove_equality(lhs, rhs)

    # --- Symbolic reasoning (optional if sympy is available) ---
    def symbolic_verify(self, statement: str) -> Proof:
        steps: List[ProofStep] = []
        if sp is None:
            steps.append(ProofStep(1, "error", "Sympy not available for symbolic verification"))
            return Proof(statement, steps, False, "Sympy unavailable")

        parsed = self._parse_equality(statement)
        if not parsed:
            return Proof(statement, [ProofStep(1, "error", "Not an equality statement")], False, "Parse error")

        lhs_str, rhs_str = parsed
        idx = 1
        steps.append(ProofStep(idx, "parse", f"Parsed equality: {lhs_str} = {rhs_str}")); idx += 1

        try:
            # Parse symbols dynamically
            symbols = sorted(set(re.findall(r"[a-zA-Z]", statement)))
            sym_objs = sp.symbols(" ".join(symbols)) if symbols else ()
            sym_map = {str(s): s for s in (sym_objs if isinstance(sym_objs, (list, tuple)) else [sym_objs]) if s}

            lhs = sp.sympify(lhs_str, locals=sym_map)
            rhs = sp.sympify(rhs_str, locals=sym_map)
            steps.append(ProofStep(idx, "sympify", f"Sympify both sides")); idx += 1

            # Simplify both sides
            lhs_s = sp.simplify(lhs)
            rhs_s = sp.simplify(rhs)
            steps.append(ProofStep(idx, "simplify", f"LHS -> {sp.srepr(lhs_s)}")); idx += 1
            steps.append(ProofStep(idx, "simplify", f"RHS -> {sp.srepr(rhs_s)}")); idx += 1

            # Compare canonical difference
            diff = sp.simplify(lhs_s - rhs_s)
            is_zero = sp.simplify(diff) == 0
            steps.append(ProofStep(idx, "compare", f"Simplify(LHS - RHS) -> {diff}")); idx += 1
            verdict = "Symbolic equality holds" if is_zero else "Symbolic equality does not hold"
            return Proof(f"{lhs_str} = {rhs_str}", steps, bool(is_zero), verdict)
        except Exception as e:
            steps.append(ProofStep(idx, "error", f"Symbolic verification failed: {e}"))
            return Proof(f"{lhs_str} = {rhs_str}", steps, False, "Symbolic error")