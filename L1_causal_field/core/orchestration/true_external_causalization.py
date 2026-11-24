"""
ELYSIA Phase 17: True Template-Free External Causalization & Self-Conviction Rewiring
====================================================================================

Eliminates developer code templates (`f-strings`) and closed domain vocabularies:
Elysia encounters completely unseen external numerical data, arbitrary pure Python
source files, and brand-new academic prose/equations, recognizes real deficits in
her own empty/evolving `CausalMindGraph`, extracts candidate causal mechanisms
directly from external structure, falsifies lies, proves generalization counterfactually
on held-out observations, and **rewires her own cognitive-sensory graph only upon
genuine self-conviction**.

Zero hardcoding guarantees:
- NO `AST_ALIAS_TO_CANONICAL` closed variable table
- NO `f'''class ...'''` organ string templates
- NO domain-specific regression columns (`g`, `drag`, `wien`, etc.)
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np

from core.orchestration.elysia_unified_living_cycle import (
    ElysiaUnifiedLivingCycle,
    LivingEncounter,
)
from core.orchestration.extended_cognitive_organism import ExtendedWorldState
from core.physics.spectral_scale_rotor_manifold import encode_waveform_to_scale_rotors

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField


# ---------------------------------------------------------------------------
# 1. Open Cognitive-Sensory Mind Graph Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ConceptNode:
    """A dynamically sprouted concept/variable node inside Elysia's open mind."""

    name: str
    role: str = "latent"
    origin_source: str = "experience"
    aliases: List[str] = field(default_factory=list)


@dataclass
class ConvictionProof:
    """Empirical before/after counterfactual proof justifying self-rewiring or rejection."""

    target_var: str
    convinced: bool
    rewired: bool
    chosen_operator_id: Optional[str]
    source_kind: str
    source_locator: str
    formula_str: str
    before_rmse: float
    after_rmse: float
    held_out_r2: float
    mdl_complexity: float
    rejected_candidates: List[Dict[str, Any]]
    conviction_statement_ko: str


@dataclass
class CausalEdgeOperator:
    """A live causal law extracted from external data, code AST, or academic prose."""

    operator_id: str
    source_kind: str
    source_locator: str
    input_vars: Tuple[str, ...]
    output_var: str
    formula_str: str
    evaluator: Optional[Callable[[Dict[str, Any]], Any]]
    qualitative_signs: Dict[str, int] = field(default_factory=dict)
    mdl_complexity: float = 1.0
    validation_r2: float = 0.0
    ast_source: str = ""
    module_constants_snapshot: Dict[str, Any] = field(default_factory=dict)
    output_aliases: Tuple[str, ...] = ()
    conviction_proof: Optional[ConvictionProof] = None


@dataclass
class InferenceTrace:
    """Step-by-step causal propagation trace through Elysia's live mind graph."""

    reached: bool
    target_var: str
    target_value: Any
    fired_chain: List[str]
    intermediate_values: Dict[str, Any]
    missing_dependencies: List[str]


@dataclass
class CausalDeficit:
    """Honest diagnosis that Elysia's current mind cannot explain an external variable."""

    target_var: str
    is_disconnected: bool
    baseline_rmse: float
    baseline_r2: float
    needs_learning: bool
    diagnosis_ko: str


@dataclass
class MultiHopLearningReport:
    """End-to-end outcome across chained heterogeneous external learning stages."""

    final_target: str
    reached_before: bool
    reached_after: bool
    stage_proofs: List[ConvictionProof]
    inference_trace: InferenceTrace
    end_to_end_sensitivity: Optional[float]
    summary_ko: str


def _serialize_constant(val: Any) -> Any:
    if isinstance(val, np.ndarray):
        return {"__ndarray__": val.tolist()}
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (list, tuple)):
        return [_serialize_constant(x) for x in val]
    return val


def _deserialize_constant(val: Any) -> Any:
    if isinstance(val, dict) and "__ndarray__" in val:
        return np.asarray(val["__ndarray__"], dtype=np.float64)
    if isinstance(val, list):
        return [_deserialize_constant(x) for x in val]
    return val


class CausalMindGraph:
    """
    Initially empty open causal mind graph supporting dynamic alias registration,
    multi-hop inference, counterfactual cloning, finite-difference sensitivities,
    and template-free JSON serialization + AST recompilation.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, ConceptNode] = {}
        self.operators: Dict[str, CausalEdgeOperator] = {}
        self.alias_map: Dict[str, str] = {}

    def register_alias(self, alias: str, canonical: str) -> None:
        a = alias.strip()
        c = canonical.strip()
        if not a or not c:
            return
        self.alias_map[a] = c
        if c in self.nodes and a not in self.nodes[c].aliases:
            self.nodes[c].aliases.append(a)

    def canonicalize(self, name: str) -> str:
        cur = name.strip()
        seen: set[str] = set()
        while cur in self.alias_map and cur not in seen:
            seen.add(cur)
            cur = self.alias_map[cur]
        return cur

    def register_concept(
        self,
        name: str,
        role: str = "latent",
        origin_source: str = "experience",
        aliases: Sequence[str] = (),
    ) -> ConceptNode:
        canon = self.canonicalize(name)
        if canon not in self.nodes:
            self.nodes[canon] = ConceptNode(
                name=canon,
                role=role,
                origin_source=origin_source,
                aliases=list(aliases),
            )
        else:
            node = self.nodes[canon]
            if node.role == "latent" and role != "latent":
                node.role = role
            for al in aliases:
                if al not in node.aliases:
                    node.aliases.append(al)
        for al in aliases:
            self.register_alias(al, canon)
        return self.nodes[canon]

    def commit_operator(self, op: CausalEdgeOperator) -> None:
        in_canon = tuple(self.canonicalize(v) for v in op.input_vars)
        out_canon = self.canonicalize(op.output_var)
        for v in in_canon:
            self.register_concept(v, role="cause", origin_source=op.source_locator)
        self.register_concept(
            out_canon,
            role="derived",
            origin_source=op.source_locator,
            aliases=op.output_aliases,
        )
        op_canon = CausalEdgeOperator(
            operator_id=op.operator_id,
            source_kind=op.source_kind,
            source_locator=op.source_locator,
            input_vars=in_canon,
            output_var=out_canon,
            formula_str=op.formula_str,
            evaluator=op.evaluator,
            qualitative_signs={self.canonicalize(k): v for k, v in op.qualitative_signs.items()},
            mdl_complexity=op.mdl_complexity,
            validation_r2=op.validation_r2,
            ast_source=op.ast_source,
            module_constants_snapshot=dict(op.module_constants_snapshot),
            output_aliases=op.output_aliases,
            conviction_proof=op.conviction_proof,
        )
        self.operators[op_canon.operator_id] = op_canon

    def clone(self) -> "CausalMindGraph":
        g = CausalMindGraph()
        g.alias_map = dict(self.alias_map)
        g.nodes = {
            k: ConceptNode(n.name, n.role, n.origin_source, list(n.aliases))
            for k, n in self.nodes.items()
        }
        g.operators = dict(self.operators)
        return g

    def infer(
        self,
        known_values: Dict[str, Any],
        target_var: str,
        max_hops: int = 12,
    ) -> InferenceTrace:
        canon_target = self.canonicalize(target_var)
        env: Dict[str, Any] = {}
        for k, v in known_values.items():
            env[self.canonicalize(k)] = v

        fired: List[str] = []
        for _ in range(max(1, max_hops)):
            if canon_target in env:
                break
            progressed = False
            for op in self.operators.values():
                if op.evaluator is None or op.output_var in env:
                    continue
                if all(iv in env for iv in op.input_vars):
                    try:
                        val = op.evaluator(env)
                    except Exception:
                        continue
                    env[op.output_var] = val
                    fired.append(op.operator_id)
                    progressed = True
            if not progressed:
                break

        reached = canon_target in env
        missing: List[str] = []
        if not reached:
            for op in self.operators.values():
                if op.output_var == canon_target:
                    missing.extend([iv for iv in op.input_vars if iv not in env])
            if not missing:
                missing = [canon_target]

        return InferenceTrace(
            reached=reached,
            target_var=canon_target,
            target_value=env.get(canon_target),
            fired_chain=fired,
            intermediate_values=env,
            missing_dependencies=sorted(set(missing)),
        )

    def compute_sensitivity(
        self,
        known_values: Dict[str, float],
        input_var: str,
        target_var: str,
        eps: float = 1e-4,
    ) -> Optional[float]:
        canon_in = self.canonicalize(input_var)
        if canon_in not in known_values and input_var not in known_values:
            return None
        base_val = float(known_values.get(canon_in, known_values.get(input_var, 0.0)))
        plus_env = dict(known_values)
        plus_env[input_var] = base_val + eps
        minus_env = dict(known_values)
        minus_env[input_var] = base_val - eps
        tr_p = self.infer(plus_env, target_var)
        tr_m = self.infer(minus_env, target_var)
        if not (tr_p.reached and tr_m.reached):
            return None
        try:
            vp = float(np.mean(tr_p.target_value))
            vm = float(np.mean(tr_m.target_value))
        except Exception:
            return None
        return float((vp - vm) / (2.0 * eps))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "alias_map": dict(self.alias_map),
            "operators": [
                {
                    "operator_id": op.operator_id,
                    "source_kind": op.source_kind,
                    "source_locator": op.source_locator,
                    "input_vars": list(op.input_vars),
                    "output_var": op.output_var,
                    "formula_str": op.formula_str,
                    "qualitative_signs": dict(op.qualitative_signs),
                    "mdl_complexity": float(op.mdl_complexity),
                    "validation_r2": float(op.validation_r2),
                    "ast_source": op.ast_source,
                    "module_constants_snapshot": {
                        k: _serialize_constant(v)
                        for k, v in op.module_constants_snapshot.items()
                    },
                    "output_aliases": list(op.output_aliases),
                }
                for op in self.operators.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalMindGraph":
        g = cls()
        g.alias_map = {str(k): str(v) for k, v in data.get("alias_map", {}).items()}
        for nd in data.get("nodes", []):
            g.nodes[str(nd["name"])] = ConceptNode(
                name=str(nd["name"]),
                role=str(nd.get("role", "latent")),
                origin_source=str(nd.get("origin_source", "reloaded")),
                aliases=[str(a) for a in nd.get("aliases", [])],
            )
        for od in data.get("operators", []):
            ast_src = str(od.get("ast_source", ""))
            src_kind = str(od["source_kind"])
            const_snap = {
                str(k): _deserialize_constant(v)
                for k, v in od.get("module_constants_snapshot", {}).items()
            }
            evaluator: Optional[Callable[[Dict[str, Any]], Any]] = None
            if ast_src.strip():
                if src_kind == "external_code_ast":
                    fn_node = ast.parse(ast_src).body[0]
                    if not isinstance(fn_node, ast.FunctionDef):
                        raise UnsafeASTError("Expected FunctionDef in saved external_code_ast")
                    SafeOpenASTCompiler.validate_func_ast(fn_node)
                    in_vars = tuple(str(v) for v in od["input_vars"])
                    evaluator = ASTCodeCausalizer._make_func_evaluator(
                        fn_node, in_vars, const_snap
                    )
                else:
                    comp_eval, _, _ = SafeOpenASTCompiler.compile_expression(ast_src)
                    evaluator = comp_eval

            op = CausalEdgeOperator(
                operator_id=str(od["operator_id"]),
                source_kind=src_kind,
                source_locator=str(od["source_locator"]),
                input_vars=tuple(str(v) for v in od["input_vars"]),
                output_var=str(od["output_var"]),
                formula_str=str(od["formula_str"]),
                evaluator=evaluator,
                qualitative_signs={
                    str(k): int(v) for k, v in od.get("qualitative_signs", {}).items()
                },
                mdl_complexity=float(od.get("mdl_complexity", 1.0)),
                validation_r2=float(od.get("validation_r2", 0.0)),
                ast_source=ast_src,
                module_constants_snapshot=const_snap,
                output_aliases=tuple(str(a) for a in od.get("output_aliases", [])),
            )
            g.operators[op.operator_id] = op
        return g


# ---------------------------------------------------------------------------
# 2. Safe Open-Vocabulary Pure AST Interpreter & External Code Causalizer
# ---------------------------------------------------------------------------


class UnsafeASTError(ValueError):
    """Raised when external code or expressions violate safe side-effect-free AST bounds."""


class _ReturnSignal(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


def _safe_sin(x: Any) -> Any:
    return np.sin(x) if isinstance(x, np.ndarray) else math.sin(float(x))


def _safe_cos(x: Any) -> Any:
    return np.cos(x) if isinstance(x, np.ndarray) else math.cos(float(x))


def _safe_tan(x: Any) -> Any:
    return np.tan(x) if isinstance(x, np.ndarray) else math.tan(float(x))


def _safe_exp(x: Any) -> Any:
    return np.exp(x) if isinstance(x, np.ndarray) else math.exp(float(x))


def _safe_expm1(x: Any) -> Any:
    return np.expm1(x) if isinstance(x, np.ndarray) else math.expm1(float(x))


def _safe_log(x: Any) -> Any:
    return np.log(x) if isinstance(x, np.ndarray) else math.log(float(x))


def _safe_log10(x: Any) -> Any:
    return np.log10(x) if isinstance(x, np.ndarray) else math.log10(float(x))


def _safe_sqrt(x: Any) -> Any:
    return np.sqrt(x) if isinstance(x, np.ndarray) else math.sqrt(float(x))


def _safe_tanh(x: Any) -> Any:
    return np.tanh(x) if isinstance(x, np.ndarray) else math.tanh(float(x))


def _safe_abs(x: Any) -> Any:
    return np.abs(x) if isinstance(x, np.ndarray) else abs(x)


def _safe_clip(a: Any, a_min: Any, a_max: Any) -> Any:
    if isinstance(a, np.ndarray):
        return np.clip(a, a_min, a_max)
    return min(a_max, max(a_min, a))


def _safe_np_array(obj: Any, dtype: Any = None) -> np.ndarray:
    return np.array(obj, dtype=dtype) if dtype is not None else np.array(obj)


def _safe_np_asarray(obj: Any, dtype: Any = None) -> np.ndarray:
    return np.asarray(obj, dtype=dtype) if dtype is not None else np.asarray(obj)


_DIRECT_SAFE_FUNCS: Dict[str, Callable[..., Any]] = {
    "sin": _safe_sin,
    "cos": _safe_cos,
    "tan": _safe_tan,
    "exp": _safe_exp,
    "expm1": _safe_expm1,
    "log": _safe_log,
    "log10": _safe_log10,
    "sqrt": _safe_sqrt,
    "tanh": _safe_tanh,
    "abs": _safe_abs,
    "fabs": _safe_abs,
    "hypot": math.hypot,
    "atan2": math.atan2,
    "min": min,
    "max": max,
    "float": float,
    "int": int,
    "len": len,
    "pow": pow,
    "clip": _safe_clip,
    "sum": np.sum,
    "mean": np.mean,
    "std": np.std,
    "array": _safe_np_array,
    "asarray": _safe_np_asarray,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "dot": np.dot,
}

_MODULE_SAFE_ATTRS: Dict[Tuple[str, str], Any] = {
    ("math", "sin"): _safe_sin,
    ("math", "cos"): _safe_cos,
    ("math", "tan"): _safe_tan,
    ("math", "exp"): _safe_exp,
    ("math", "expm1"): _safe_expm1,
    ("math", "log"): _safe_log,
    ("math", "log10"): _safe_log10,
    ("math", "sqrt"): _safe_sqrt,
    ("math", "tanh"): _safe_tanh,
    ("math", "fabs"): _safe_abs,
    ("math", "hypot"): math.hypot,
    ("math", "atan2"): math.atan2,
    ("math", "pi"): math.pi,
    ("math", "e"): math.e,
    ("np", "sin"): _safe_sin,
    ("np", "cos"): _safe_cos,
    ("np", "tan"): _safe_tan,
    ("np", "exp"): _safe_exp,
    ("np", "expm1"): _safe_expm1,
    ("np", "log"): _safe_log,
    ("np", "log10"): _safe_log10,
    ("np", "sqrt"): _safe_sqrt,
    ("np", "tanh"): _safe_tanh,
    ("np", "abs"): _safe_abs,
    ("np", "clip"): _safe_clip,
    ("np", "sum"): np.sum,
    ("np", "mean"): np.mean,
    ("np", "std"): np.std,
    ("np", "array"): _safe_np_array,
    ("np", "asarray"): _safe_np_asarray,
    ("np", "maximum"): np.maximum,
    ("np", "minimum"): np.minimum,
    ("np", "dot"): np.dot,
    ("np", "pi"): math.pi,
    ("np", "e"): math.e,
    ("np", "float64"): np.float64,
    ("np", "float32"): np.float32,
    ("np", "int64"): np.int64,
    ("np", "int32"): np.int32,
    ("numpy", "sin"): _safe_sin,
    ("numpy", "cos"): _safe_cos,
    ("numpy", "tan"): _safe_tan,
    ("numpy", "exp"): _safe_exp,
    ("numpy", "expm1"): _safe_expm1,
    ("numpy", "log"): _safe_log,
    ("numpy", "sqrt"): _safe_sqrt,
    ("numpy", "tanh"): _safe_tanh,
    ("numpy", "abs"): _safe_abs,
    ("numpy", "clip"): _safe_clip,
    ("numpy", "sum"): np.sum,
    ("numpy", "mean"): np.mean,
    ("numpy", "array"): _safe_np_array,
    ("numpy", "asarray"): _safe_np_asarray,
    ("numpy", "float64"): np.float64,
    ("numpy", "float32"): np.float32,
    ("numpy", "pi"): math.pi,
}

_DIRECT_CONSTANTS: Dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "True": True,
    "False": False,
}

_BANNED_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "system",
        "popen",
        "subprocess",
        "shutil",
        "unlink",
        "rmdir",
        "remove",
        "socket",
        "urllib",
        "http",
        "os",
        "sys",
        "builtins",
        "__builtins__",
        "__subclasses__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
    }
)

_ALLOWED_IMPORTS = frozenset({"__future__", "math", "numpy", "dataclasses", "typing", "pathlib", "time", "wave", "abc", "core"})


class SafeOpenASTCompiler:
    """Zero-`eval` / zero-`exec` pure Python/NumPy AST validator & interpreter."""

    @classmethod
    def validate_expr_ast(cls, node: ast.AST) -> None:
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in _BANNED_NAMES:
                raise UnsafeASTError(f"Banned identifier '{n.id}' in expression")
            if isinstance(n, ast.Attribute):
                if n.attr in _BANNED_NAMES or n.attr.startswith("__"):
                    raise UnsafeASTError(f"Banned attribute '{n.attr}'")
                if isinstance(n.value, ast.Name) and n.value.id in ("math", "np", "numpy"):
                    if (n.value.id, n.attr) not in _MODULE_SAFE_ATTRS:
                        raise UnsafeASTError(f"Disallowed module attribute {n.value.id}.{n.attr}")
            if isinstance(
                n,
                (
                    ast.Lambda,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.Await,
                ),
            ):
                raise UnsafeASTError(f"Unsupported dynamic AST node {type(n).__name__}")
            if isinstance(n, ast.Call):
                fn = n.func
                if isinstance(fn, ast.Name):
                    if fn.id not in _DIRECT_SAFE_FUNCS:
                        raise UnsafeASTError(f"Disallowed function call '{fn.id}'")
                elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                    key = (fn.value.id, fn.attr)
                    if key not in _MODULE_SAFE_ATTRS or not callable(_MODULE_SAFE_ATTRS[key]):
                        raise UnsafeASTError(f"Disallowed call '{fn.value.id}.{fn.attr}'")
                else:
                    raise UnsafeASTError(f"Disallowed call target {ast.dump(fn)}")

    @classmethod
    def validate_func_ast(cls, fn_node: ast.FunctionDef) -> None:
        for n in ast.walk(fn_node):
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.While, ast.For, ast.AsyncFor, ast.With, ast.Global, ast.Nonlocal, ast.Raise, ast.Try, ast.ClassDef, ast.FunctionDef)):
                if n is not fn_node:
                    raise UnsafeASTError(f"Disallowed control/definition node {type(n).__name__} in pure organ function '{fn_node.name}'")
            if isinstance(n, ast.Name) and n.id in _BANNED_NAMES:
                raise UnsafeASTError(f"Banned identifier '{n.id}' in function '{fn_node.name}'")
            if isinstance(n, ast.Attribute) and (n.attr in _BANNED_NAMES or n.attr.startswith("__")):
                raise UnsafeASTError(f"Banned attribute '{n.attr}' in function '{fn_node.name}'")
        for stmt in fn_node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
            if isinstance(stmt, ast.Return):
                if stmt.value is not None:
                    cls.validate_expr_ast(stmt.value)
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    raise UnsafeASTError("Only single-Name assignments supported in AST functions")
                cls.validate_expr_ast(stmt.value)
            elif isinstance(stmt, ast.AugAssign):
                if not isinstance(stmt.target, ast.Name):
                    raise UnsafeASTError("Only Name AugAssign supported in AST functions")
                cls.validate_expr_ast(stmt.value)
            elif isinstance(stmt, ast.If):
                cls.validate_expr_ast(stmt.test)
                for sub in (*stmt.body, *stmt.orelse):
                    if isinstance(sub, ast.Return) and sub.value is not None:
                        cls.validate_expr_ast(sub.value)
                    elif isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                        cls.validate_expr_ast(sub.value)
                    else:
                        raise UnsafeASTError(f"Unsupported statement {type(sub).__name__} inside If")
            else:
                raise UnsafeASTError(f"Unsupported statement {type(stmt).__name__} in AST function '{fn_node.name}'")

    @classmethod
    def eval_expr(cls, node: ast.AST, env: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Expression):
            return cls.eval_expr(node.body, env)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in _DIRECT_CONSTANTS:
                return _DIRECT_CONSTANTS[node.id]
            raise KeyError(f"Unbound symbol '{node.id}' in expression evaluation")
        if isinstance(node, ast.List):
            return [cls.eval_expr(e, env) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(cls.eval_expr(e, env) for e in node.elts)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in ("math", "np", "numpy"):
                return _MODULE_SAFE_ATTRS[(node.value.id, node.attr)]
            raise UnsafeASTError(f"Disallowed attribute access {ast.dump(node)}")
        if isinstance(node, ast.UnaryOp):
            val = cls.eval_expr(node.operand, env)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            if isinstance(node.op, ast.Not):
                return not val
            raise UnsafeASTError(f"Unsupported unary op {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            l = cls.eval_expr(node.left, env)
            r = cls.eval_expr(node.right, env)
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.Div):
                return l / r
            if isinstance(node.op, ast.Pow):
                return l ** r
            if isinstance(node.op, ast.FloorDiv):
                return l // r
            if isinstance(node.op, ast.Mod):
                return l % r
            raise UnsafeASTError(f"Unsupported binary op {type(node.op).__name__}")
        if isinstance(node, ast.Compare):
            left = cls.eval_expr(node.left, env)
            for op, comp in zip(node.ops, node.comparators):
                right = cls.eval_expr(comp, env)
                if isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                elif isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                else:
                    raise UnsafeASTError(f"Unsupported compare op {type(op).__name__}")
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                res = True
                for val_node in node.values:
                    res = cls.eval_expr(val_node, env)
                    if not res:
                        return res
                return res
            if isinstance(node.op, ast.Or):
                res = False
                for val_node in node.values:
                    res = cls.eval_expr(val_node, env)
                    if res:
                        return res
                return res
            raise UnsafeASTError(f"Unsupported BoolOp {type(node.op).__name__}")
        if isinstance(node, ast.IfExp):
            return cls.eval_expr(node.body, env) if cls.eval_expr(node.test, env) else cls.eval_expr(node.orelse, env)
        if isinstance(node, ast.Subscript):
            arr = cls.eval_expr(node.value, env)
            slc = cls.eval_expr(node.slice, env)
            return arr[slc]
        if isinstance(node, ast.Call):
            args = [cls.eval_expr(a, env) for a in node.args]
            kwargs = {kw.arg: cls.eval_expr(kw.value, env) for kw in node.keywords if kw.arg is not None}
            if isinstance(node.func, ast.Name):
                fn = _DIRECT_SAFE_FUNCS[node.func.id]
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                fn = _MODULE_SAFE_ATTRS[(node.func.value.id, node.func.attr)]
            else:
                raise UnsafeASTError("Disallowed call")
            return fn(*args, **kwargs)
        raise UnsafeASTError(f"Unsupported AST expression node {type(node).__name__}")

    @classmethod
    def compile_expression(cls, expr_str: str) -> Tuple[Callable[[Dict[str, Any]], Any], Tuple[str, ...], int]:
        tree = ast.parse(expr_str.strip(), mode="eval")
        cls.validate_expr_ast(tree)

        free_vars: List[str] = []
        op_count = 0
        # Identify free variables (skipping function names, module bases `math`/`np`/`numpy`, constants)
        fn_or_mod_names: set[int] = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Call)):
                op_count += 1
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                fn_or_mod_names.add(id(n.func))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                fn_or_mod_names.add(id(n.value))

        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                if id(n) in fn_or_mod_names:
                    continue
                if n.id in _DIRECT_CONSTANTS or n.id in _DIRECT_SAFE_FUNCS:
                    continue
                if n.id not in free_vars:
                    free_vars.append(n.id)

        def _evaluator(env: Dict[str, Any], _t=tree) -> Any:
            return cls.eval_expr(_t, env)

        return _evaluator, tuple(free_vars), max(1, op_count)

    @classmethod
    def _exec_stmts(cls, stmts: Sequence[ast.stmt], local_env: Dict[str, Any]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Expr):
                continue
            if isinstance(stmt, ast.Return):
                val = cls.eval_expr(stmt.value, local_env) if stmt.value is not None else None
                raise _ReturnSignal(val)
            if isinstance(stmt, ast.Assign):
                tgt = stmt.targets[0]
                assert isinstance(tgt, ast.Name)
                local_env[tgt.id] = cls.eval_expr(stmt.value, local_env)
            elif isinstance(stmt, ast.AugAssign):
                assert isinstance(stmt.target, ast.Name)
                cur = local_env[stmt.target.id]
                rhs = cls.eval_expr(stmt.value, local_env)
                if isinstance(stmt.op, ast.Add):
                    local_env[stmt.target.id] = cur + rhs
                elif isinstance(stmt.op, ast.Sub):
                    local_env[stmt.target.id] = cur - rhs
                elif isinstance(stmt.op, ast.Mult):
                    local_env[stmt.target.id] = cur * rhs
                elif isinstance(stmt.op, ast.Div):
                    local_env[stmt.target.id] = cur / rhs
            elif isinstance(stmt, ast.If):
                branch = stmt.body if cls.eval_expr(stmt.test, local_env) else stmt.orelse
                cls._exec_stmts(branch, local_env)

    @classmethod
    def exec_function_ast(
        cls,
        fn_node: ast.FunctionDef,
        input_env: Dict[str, Any],
        module_constants: Dict[str, Any],
    ) -> Any:
        local_env: Dict[str, Any] = dict(module_constants)
        for arg in fn_node.args.args:
            if arg.arg not in input_env:
                raise KeyError(f"Missing argument '{arg.arg}' for AST function '{fn_node.name}'")
            local_env[arg.arg] = input_env[arg.arg]
        try:
            cls._exec_stmts(fn_node.body, local_env)
        except _ReturnSignal as sig:
            return sig.value
        return None


_GENERIC_RETURN_NAMES = frozenset({"res", "out", "val", "ret", "result", "x", "y", "ans"})


class ASTCodeCausalizer:
    """Extracts safe, executable causal operators directly from external Python source files."""

    @staticmethod
    def _make_func_evaluator(
        fn_node: ast.FunctionDef,
        input_vars: Tuple[str, ...],
        module_constants: Dict[str, Any],
    ) -> Callable[[Dict[str, Any]], Any]:
        consts = dict(module_constants)

        def _eval(env: Dict[str, Any], _fn=fn_node, _iv=input_vars, _mc=consts) -> Any:
            arg_env = {v: env[v] for v in _iv}
            return SafeOpenASTCompiler.exec_function_ast(_fn, arg_env, _mc)

        return _eval

    @staticmethod
    def validate_file_safety(py_path: Path) -> Tuple[bool, str]:
        src = Path(py_path).read_text(encoding="utf-8", errors="replace")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(src, filename=str(py_path))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    top = a.name.split(".")[0]
                    if top in _BANNED_NAMES or top not in _ALLOWED_IMPORTS:
                        return False, f"Disallowed import '{a.name}'"
            elif isinstance(n, ast.ImportFrom):
                top = (n.module or "").split(".")[0]
                if top in _BANNED_NAMES or top not in _ALLOWED_IMPORTS:
                    return False, f"Disallowed from-import '{n.module}'"
            elif isinstance(n, ast.Name) and n.id in _BANNED_NAMES:
                return False, f"Banned symbol '{n.id}'"
            elif isinstance(n, ast.Attribute) and n.attr in _BANNED_NAMES:
                return False, f"Banned attribute '{n.attr}'"
        return True, "SAFE"

    @classmethod
    def extract_operators_from_file(
        cls,
        py_path: Path,
        strict: bool = False,
    ) -> List[CausalEdgeOperator]:
        fpath = Path(py_path)
        if strict:
            ok, reason = cls.validate_file_safety(fpath)
            if not ok:
                raise UnsafeASTError(f"{fpath.name} rejected by safety gate: {reason}")

        src = fpath.read_text(encoding="utf-8", errors="replace")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(src, filename=str(fpath))

        module_consts: Dict[str, Any] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                cname = node.targets[0].id
                try:
                    SafeOpenASTCompiler.validate_expr_ast(node.value)
                    val = SafeOpenASTCompiler.eval_expr(node.value, module_consts)
                    if isinstance(val, (int, float, np.number, np.ndarray, tuple, list)):
                        module_consts[cname] = val
                except Exception:
                    pass

        ops: List[CausalEdgeOperator] = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            if not node.args.args:
                continue
            try:
                SafeOpenASTCompiler.validate_func_ast(node)
            except UnsafeASTError:
                if strict:
                    raise
                continue

            ret_stmt = next((s for s in reversed(node.body) if isinstance(s, ast.Return)), None)
            if ret_stmt is None or ret_stmt.value is None:
                continue

            in_vars = tuple(a.arg for a in node.args.args)
            if isinstance(ret_stmt.value, ast.Name) and ret_stmt.value.id not in in_vars and ret_stmt.value.id not in _GENERIC_RETURN_NAMES:
                out_var = ret_stmt.value.id
            else:
                out_var = node.name
            aliases = tuple(dict.fromkeys([node.name] + ([ret_stmt.value.id] if isinstance(ret_stmt.value, ast.Name) else [])))

            # Keep only constants referenced inside this function for compact serialization
            used_names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            snap = {k: v for k, v in module_consts.items() if k in used_names}

            evaluator = cls._make_func_evaluator(node, in_vars, snap)
            fn_src = ast.unparse(node)
            formula_desc = f"{node.name}({', '.join(in_vars)}) -> {ast.unparse(ret_stmt.value)}"
            op_id = f"code::{fpath.stem}.{node.name}_{hashlib.sha256(fn_src.encode('utf-8')).hexdigest()[:6]}"
            ops.append(
                CausalEdgeOperator(
                    operator_id=op_id,
                    source_kind="external_code_ast",
                    source_locator=str(fpath),
                    input_vars=in_vars,
                    output_var=out_var,
                    formula_str=formula_desc,
                    evaluator=evaluator,
                    mdl_complexity=float(max(1, len(node.body))),
                    validation_r2=1.0,
                    ast_source=fn_src,
                    module_constants_snapshot=snap,
                    output_aliases=aliases,
                )
            )
        return ops


# ---------------------------------------------------------------------------
# 3. Structure-Agnostic Open-Basis Symbolic Data Causalizer (STLSQ + MDL)
# ---------------------------------------------------------------------------


class SymbolicDataCausalizer:
    """
    Extracts exact, interpretable sparse mathematical equations from ANY unknown
    numerical arrays without knowing domain variables or equations beforehand.
    Uses 70% discovery / 30% held-out validation + MDL complexity pruning.
    """

    @staticmethod
    def _build_universal_basis(
        data: Dict[str, np.ndarray],
        input_vars: Sequence[str],
    ) -> Tuple[List[str], List[np.ndarray], List[Tuple[str, ...]]]:
        n = len(next(iter(data.values())))
        names: List[str] = ["1.0"]
        cols: List[np.ndarray] = [np.ones(n, dtype=np.float64)]
        used_sets: List[Tuple[str, ...]] = [()]

        vecs = {v: np.asarray(data[v], dtype=np.float64).ravel() for v in input_vars}

        def _try_add(expr: str, vec: np.ndarray, deps: Tuple[str, ...]) -> None:
            if not (len(vec) == n and np.all(np.isfinite(vec))):
                return
            if float(np.linalg.norm(vec)) <= 1e-12:
                return
            names.append(expr)
            cols.append(vec)
            used_sets.append(deps)

        for v in input_vars:
            x = vecs[v]
            _try_add(v, x, (v,))
            _try_add(f"({v} ** 2)", x ** 2, (v,))
            _try_add(f"sin({v})", np.sin(x), (v,))
            _try_add(f"cos({v})", np.cos(x), (v,))
            if float(np.min(np.abs(x))) > 1e-3:
                _try_add(f"(1.0 / {v})", 1.0 / x, (v,))
                _try_add(f"(1.0 / ({v} ** 2))", 1.0 / (x ** 2), (v,))
            if float(np.min(x)) > 1e-3:
                _try_add(f"log({v})", np.log(x), (v,))
            if float(np.min(x)) >= 0.0:
                _try_add(f"sqrt({v})", np.sqrt(x), (v,))

        for i in range(len(input_vars)):
            for j in range(i + 1, len(input_vars)):
                u, v = input_vars[i], input_vars[j]
                _try_add(f"({u} * {v})", vecs[u] * vecs[v], (u, v))

        return names, cols, used_sets

    @classmethod
    def discover(
        cls,
        data: Dict[str, np.ndarray],
        target_var: str,
        input_vars: Optional[Sequence[str]] = None,
        source_locator: str = "external_observations",
    ) -> Optional[CausalEdgeOperator]:
        if target_var not in data:
            return None
        y_all = np.asarray(data[target_var], dtype=np.float64).ravel()
        n = len(y_all)
        if n < 10:
            return None

        in_candidates = (
            list(input_vars)
            if input_vars is not None
            else [k for k in data.keys() if k != target_var]
        )
        if not in_candidates:
            return None

        idx = np.arange(n)
        tr_mask = (idx % 10) < 7
        va_mask = ~tr_mask
        if int(np.sum(va_mask)) < 3 or int(np.sum(tr_mask)) < 5:
            return None

        b_names, b_cols, b_deps = cls._build_universal_basis(data, in_candidates)
        Phi = np.column_stack(b_cols)
        Phi_tr, y_tr = Phi[tr_mask], y_all[tr_mask]
        Phi_va, y_va = Phi[va_mask], y_all[va_mask]

        col_norms = np.linalg.norm(Phi_tr, axis=0)
        col_norms = np.where(col_norms > 1e-12, col_norms, 1.0)
        Phi_tr_norm = Phi_tr / col_norms[None, :]
        n_val = len(y_va)
        var_va = max(1e-12, float(np.var(y_va)))

        best_mdl = float("inf")
        best_support: Optional[np.ndarray] = None
        best_coefs: Optional[np.ndarray] = None
        best_val_r2 = -float("inf")

        thresholds = (0.002, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35)
        for thresh in thresholds:
            active = np.ones(Phi.shape[1], dtype=bool)
            for _ in range(8):
                act_idx = np.where(active)[0]
                if len(act_idx) == 0:
                    break
                w_sub = np.linalg.lstsq(Phi_tr_norm[:, act_idx], y_tr, rcond=None)[0]
                max_w = float(np.max(np.abs(w_sub))) if len(w_sub) else 0.0
                if max_w <= 1e-12:
                    active[:] = False
                    break
                keep_local = np.abs(w_sub) >= max(1e-9, thresh * max_w)
                if np.all(keep_local):
                    break
                active[act_idx[~keep_local]] = False

            act_idx = np.where(active)[0]
            if len(act_idx) == 0:
                continue

            # Refit raw unnormalized coefficients on train split
            raw_coef = np.linalg.lstsq(Phi_tr[:, act_idx], y_tr, rcond=None)[0]
            # Strip coefficients negligible relative to target scale
            sig_mask = (np.abs(raw_coef) * col_norms[act_idx]) > 1e-6 * max(1.0, float(np.linalg.norm(y_tr)))
            if not np.any(sig_mask):
                continue
            act_idx = act_idx[sig_mask]
            raw_coef = np.linalg.lstsq(Phi_tr[:, act_idx], y_tr, rcond=None)[0]

            y_pred_va = Phi_va[:, act_idx] @ raw_coef
            val_mse = float(np.mean((y_va - y_pred_va) ** 2))
            val_r2 = float(1.0 - val_mse / var_va)
            k_terms = len(act_idx)
            mdl = n_val * math.log(max(val_mse, 1e-15)) + 2.5 * k_terms * math.log(max(2, n_val))

            if mdl < best_mdl:
                best_mdl = mdl
                best_support = act_idx
                best_coefs = raw_coef
                best_val_r2 = val_r2

        if best_support is None or best_coefs is None:
            return None

        # Format clean algebraic expression and compile via SafeOpenASTCompiler
        terms: List[str] = []
        used_vars: List[str] = []
        for j, coef in zip(best_support, best_coefs):
            expr_basis = b_names[int(j)]
            for dv in b_deps[int(j)]:
                if dv not in used_vars:
                    used_vars.append(dv)
            c_val = float(coef)
            if expr_basis == "1.0":
                terms.append(f"{c_val:.8g}")
            else:
                terms.append(f"({c_val:.8g} * {expr_basis})")

        full_expr = " + ".join(terms).replace("+ (-", "- (")
        try:
            evaluator, compiled_vars, _ = SafeOpenASTCompiler.compile_expression(full_expr)
        except Exception:
            return None

        final_vars = tuple(dict.fromkeys([*used_vars, *compiled_vars]))
        op_id = f"sym::{target_var}_{hashlib.sha256(full_expr.encode('utf-8')).hexdigest()[:6]}"
        return CausalEdgeOperator(
            operator_id=op_id,
            source_kind="symbolic_data_discovery",
            source_locator=source_locator,
            input_vars=final_vars,
            output_var=target_var,
            formula_str=f"{target_var} = {full_expr}",
            evaluator=evaluator,
            mdl_complexity=float(len(best_support)),
            validation_r2=best_val_r2,
            ast_source=full_expr,
        )


# ---------------------------------------------------------------------------
# 4. Open-Vocabulary Academic Text & Equation Causalizer
# ---------------------------------------------------------------------------


_BILINGUAL_DEF_RE = re.compile(r"((?:[가-힣][가-힣0-9_]{1,12}\s+)?[가-힣][가-힣0-9_]{1,14})\s*\(([a-zA-Z_][a-zA-Z0-9_]*)\)")
_INLINE_TAG_RE = re.compile(r"\(([a-zA-Z_][a-zA-Z0-9_]*)\)")
_INLINE_STRIP_RE = re.compile(r"([가-힣][가-힣0-9_]*(?:\s+[가-힣][가-힣0-9_]*)?)\(([a-zA-Z_][a-zA-Z0-9_]*)\)")
_GRAMMAR_PREV_ENDINGS = ("은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "로", "면", "수록", "고", "며", "서", "다")


def _strip_kr_suffix(word: str) -> str:
    for suf in ("에서는", "으로는", "에서", "으로", "에게", "보다", "만큼", "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만"):
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[: -len(suf)]
    return word


class OpenAcademicCausalizer:
    """
    Parses academic definitions, inline equations, and directional qualitative
    propositions in ANY unseen domain without a pre-populated variable dictionary.
    """

    @staticmethod
    def _mine_aliases(text: str, base_aliases: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        aliases = dict(base_aliases or {})
        for kr_raw, asc in _BILINGUAL_DEF_RE.findall(text):
            tokens = [t for t in kr_raw.strip().split() if t]
            if not tokens:
                continue
            head = _strip_kr_suffix(tokens[-1])
            if head:
                aliases[head] = asc
            if len(tokens) == 2 and not tokens[0].endswith(_GRAMMAR_PREV_ENDINGS):
                compound = f"{tokens[0]} {head}"
                aliases[compound] = asc
                aliases[compound.replace(" ", "")] = asc
        return aliases

    @staticmethod
    def _normalize_expr_with_aliases(expr: str, aliases: Dict[str, str]) -> str:
        s = _INLINE_STRIP_RE.sub(r" \2 ", expr.strip())
        for al in sorted(aliases.keys(), key=len, reverse=True):
            canon = aliases[al]
            if any("\uac00" <= c <= "\ud7a3" for c in al):
                s = s.replace(al, canon)
            else:
                s = re.sub(rf"\b{re.escape(al)}\b", canon, s)
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def extract_from_text(
        cls,
        text: str,
        source_name: str = "academic_text",
        existing_aliases: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        aliases = cls._mine_aliases(text, base_aliases=existing_aliases)
        operators: List[CausalEdgeOperator] = []

        # 1. Extract backticked & standalone equations
        candidates: List[str] = re.findall(r"`([^`]+)`", text)
        for ln in text.splitlines():
            ls = ln.strip()
            if "=" in ls and "`" not in ls:
                candidates.append(ls)

        seen_eq: set[str] = set()
        for eq in candidates:
            if "=" not in eq or eq in seen_eq:
                continue
            seen_eq.add(eq)
            parts = eq.replace("==", "=").split("=")
            if len(parts) != 2:
                continue
            lhs_raw = cls._normalize_expr_with_aliases(parts[0], aliases).strip()
            rhs_raw = cls._normalize_expr_with_aliases(parts[1], aliases).strip()
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", lhs_raw):
                continue
            try:
                evaluator, in_vars, comp = SafeOpenASTCompiler.compile_expression(rhs_raw)
            except Exception:
                continue
            op_id = f"acad_eq::{lhs_raw}_{hashlib.sha256(rhs_raw.encode('utf-8')).hexdigest()[:6]}"
            operators.append(
                CausalEdgeOperator(
                    operator_id=op_id,
                    source_kind="academic_equation",
                    source_locator=source_name,
                    input_vars=in_vars,
                    output_var=lhs_raw,
                    formula_str=f"{lhs_raw} = {rhs_raw}",
                    evaluator=evaluator,
                    mdl_complexity=float(comp),
                    validation_r2=1.0,
                    ast_source=rhs_raw,
                )
            )

        # 2. Extract directional qualitative propositions over dynamically mined concepts
        clean_prose = re.sub(r"`[^`]+`", " ", text)
        clean_prose = _INLINE_TAG_RE.sub("", clean_prose)
        known_terms = sorted(
            set(aliases.keys()) | set(aliases.values()),
            key=len,
            reverse=True,
        )

        sentences = [s.strip() for s in re.split(r"(?<!\d)[.!?;\n](?!\d)", clean_prose) if s.strip()]
        for sent in sentences:
            if not known_terms:
                continue
            hits: List[Tuple[int, str, int]] = []
            occupied: List[Tuple[int, int]] = []
            for term in known_terms:
                start = 0
                while True:
                    idx = sent.find(term, start)
                    if idx < 0:
                        break
                    end = idx + len(term)
                    if term.isascii():
                        ok_l = idx == 0 or not (sent[idx - 1].isalnum() or sent[idx - 1] == "_")
                        ok_r = end == len(sent) or not (sent[end].isalnum() or sent[end] == "_")
                        if not (ok_l and ok_r):
                            start = end
                            continue
                    if not any(not (end <= s0 or idx >= e0) for s0, e0 in occupied):
                        canon = aliases.get(term, term)
                        hits.append((idx, canon, end))
                        occupied.append((idx, end))
                    start = end
            hits.sort(key=lambda item: item[0])

            # Preserve distinct concept order
            distinct: List[Tuple[int, str, int]] = []
            seen_c: set[str] = set()
            for h in hits:
                if h[1] not in seen_c:
                    seen_c.add(h[1])
                    distinct.append(h)

            if len(distinct) < 2:
                continue

            sign: Optional[int] = None
            cause_var: Optional[str] = None
            effect_var: Optional[str] = None

            pos_cues = ("커질수록", "높아질수록", "증가할수록", "증가하면", "커지면", "increases", "raises")
            inc_eff = ("증가", "상승", "커진", "높아진", "비례", "rises", "grows", "increases")
            dec_eff = ("감소", "낮아진", "줄어든", "반비례", "떨어", "decreases", "drops", "reduces")

            cue_idx = max((sent.find(c) for c in pos_cues), default=-1)
            if cue_idx >= 0:
                before = [h for h in distinct if h[2] <= cue_idx + 1]
                after = [h for h in distinct if h[0] >= cue_idx]
                if before and after:
                    cause_var = before[-1][1]
                    effect_var = after[0][1]
                    tail = sent[after[0][2] :]
                    if any(d in tail for d in dec_eff):
                        sign = -1
                    elif any(i in tail for i in inc_eff):
                        sign = +1

            elif "반비례" in sent or "비례" in sent:
                inv_i = sent.find("반비례")
                prop_i = sent.find("비례")
                c_pos = inv_i if inv_i >= 0 else prop_i
                rel_hits = [h for h in distinct if h[0] < c_pos]
                if len(rel_hits) >= 2:
                    effect_var = rel_hits[0][1]
                    cause_var = rel_hits[-1][1]
                    sign = -1 if inv_i >= 0 else +1

            if cause_var and effect_var and cause_var != effect_var and sign is not None:
                q_id = f"acad_qual::{cause_var}_to_{effect_var}_{hashlib.sha256(sent.encode('utf-8')).hexdigest()[:6]}"
                operators.append(
                    CausalEdgeOperator(
                        operator_id=q_id,
                        source_kind="academic_qualitative",
                        source_locator=source_name,
                        input_vars=(cause_var,),
                        output_var=effect_var,
                        formula_str=f"sign(d({effect_var})/d({cause_var})) == {'+' if sign > 0 else '-'}1",
                        evaluator=None,
                        qualitative_signs={cause_var: sign},
                        mdl_complexity=1.0,
                        validation_r2=0.0,
                        ast_source="",
                    )
                )

        return {"aliases": aliases, "operators": operators}

    @staticmethod
    def verify_qualitative_claim(
        op: CausalEdgeOperator,
        data: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        y_name = op.output_var
        if y_name not in data:
            return {"verified": False, "falsified": False, "details": {}, "explanation_ko": f"검증 대상 '{y_name}' 관측 없음"}

        y = np.asarray(data[y_name], dtype=np.float64).ravel()
        cov_keys = [k for k in data.keys() if k != y_name]
        details: Dict[str, Any] = {}
        all_ok = True
        any_falsified = False

        for x_name, claimed_sign in op.qualitative_signs.items():
            if x_name not in data:
                all_ok = False
                continue
            x = np.asarray(data[x_name], dtype=np.float64).ravel()
            others = [np.asarray(data[k], dtype=np.float64).ravel() for k in cov_keys if k != x_name]
            if others:
                Z = np.column_stack([np.ones(len(y), dtype=np.float64), *others])
                x_res = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
                y_res = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
            else:
                x_res = x - float(np.mean(x))
                y_res = y - float(np.mean(y))

            denom = float(np.linalg.norm(x_res) * np.linalg.norm(y_res))
            pcorr = float(np.dot(x_res, y_res) / denom) if denom > 1e-12 else 0.0
            obs_sign = 1 if pcorr > 0.05 else (-1 if pcorr < -0.05 else 0)
            ok = obs_sign == claimed_sign
            if not ok:
                all_ok = False
                if obs_sign != 0:
                    any_falsified = True
            details[x_name] = {
                "claimed_sign": claimed_sign,
                "observed_partial_corr": pcorr,
                "observed_sign": obs_sign,
                "matched": ok,
            }

        expl = (
            f"학문 정성 명제 '{op.formula_str}'를 실측 편상관으로 확인해 "
            + ("참으로 입증했습니다." if all_ok else "실제 인과 방향과 모순됨을 포착해 거짓으로 기각했습니다.")
        )
        return {
            "verified": all_ok and bool(details),
            "falsified": any_falsified,
            "details": details,
            "explanation_ko": expl,
        }


# ---------------------------------------------------------------------------
# 5. Deficit-Driven Self-Conviction Learner & CausalField Projection
# ---------------------------------------------------------------------------


class ExternalCausalLearner:
    """
    Maintains Elysia's live `CausalMindGraph`, diagnoses real deficits before acting,
    discovers & evaluates external equations/AST code/data counterfactually on
    held-out splits, and commits permanent structural rewiring ONLY upon conviction.
    """

    def __init__(
        self,
        mind: Optional[CausalMindGraph] = None,
        living_cycle: Optional[ElysiaUnifiedLivingCycle] = None,
    ) -> None:
        self.mind: CausalMindGraph = mind if mind is not None else CausalMindGraph()
        self.living_cycle = living_cycle
        self.conviction_history: List[ConvictionProof] = []
        self.qualitative_verdicts: List[Dict[str, Any]] = []

    def _canonical_obs(
        self,
        observations: Dict[str, np.ndarray],
        extra_aliases: Optional[Dict[str, str]] = None,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        aliases = {**self.mind.alias_map, **(extra_aliases or {})}
        for k, arr in observations.items():
            c = aliases.get(k, self.mind.canonicalize(k))
            out[c] = np.asarray(arr, dtype=np.float64).ravel()
        return out

    def _evaluate_graph_on_indices(
        self,
        graph: CausalMindGraph,
        canon_obs: Dict[str, np.ndarray],
        target_var: str,
        indices: np.ndarray,
    ) -> Tuple[bool, float, float]:
        y_true = canon_obs[target_var][indices]
        preds: List[float] = []
        for idx in indices:
            row_in = {k: float(arr[idx]) for k, arr in canon_obs.items() if k != target_var}
            tr = graph.infer(row_in, target_var)
            if not tr.reached or tr.target_value is None:
                return False, float("inf"), -float("inf")
            try:
                val = float(tr.target_value)
            except Exception:
                return False, float("inf"), -float("inf")
            if not math.isfinite(val):
                return False, float("inf"), -float("inf")
            preds.append(val)

        y_pred = np.asarray(preds, dtype=np.float64)
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(math.sqrt(mse))
        var_y = max(1e-12, float(np.var(y_true)))
        r2 = float(1.0 - mse / var_y)
        return True, rmse, r2

    def diagnose_deficit(
        self,
        target_var: str,
        observations: Dict[str, np.ndarray],
    ) -> CausalDeficit:
        canon_obs = self._canonical_obs(observations)
        canon_t = self.mind.canonicalize(target_var)
        n = len(canon_obs[canon_t])
        all_idx = np.arange(n)

        reached, rmse, r2 = self._evaluate_graph_on_indices(
            self.mind, canon_obs, canon_t, all_idx
        )
        is_disconn = not reached
        needs = is_disconn or r2 < 0.90
        if is_disconn:
            diag_ko = f"현재 내 인지망 안에 '{canon_t}'에 도달하는 인과 경로가 전혀 없어 예측할 수 없습니다."
        elif needs:
            diag_ko = f"'{canon_t}' 예측 오차(RMSE={rmse:.4g}, R²={r2:.3f})가 높아 외부 원리 학습이 필요합니다."
        else:
            diag_ko = f"'{canon_t}'는 이미 내 인지망 안에서 R²={r2:.4f}로 설명되고 있습니다."

        return CausalDeficit(
            target_var=canon_t,
            is_disconnected=is_disconn,
            baseline_rmse=rmse,
            baseline_r2=r2,
            needs_learning=needs,
            diagnosis_ko=diag_ko,
        )

    def learn_from_external_sources(
        self,
        target_var: str,
        observations: Dict[str, np.ndarray],
        code_files: Sequence[Path | str] = (),
        academic_texts: Sequence[Tuple[str, str]] = (),
        allow_data_discovery: bool = True,
        min_r2_for_conviction: float = 0.92,
    ) -> ConvictionProof:
        # Step 1: Parse academic texts first so aliases map cleanly
        candidate_aliases: Dict[str, str] = {}
        acad_eq_candidates: List[CausalEdgeOperator] = []
        acad_qual_claims: List[CausalEdgeOperator] = []

        for src_name, txt in academic_texts:
            parsed = OpenAcademicCausalizer.extract_from_text(
                txt, source_name=src_name, existing_aliases={**self.mind.alias_map, **candidate_aliases}
            )
            candidate_aliases.update(parsed["aliases"])
            for op in parsed["operators"]:
                if op.source_kind == "academic_qualitative":
                    acad_qual_claims.append(op)
                else:
                    acad_eq_candidates.append(op)

        canon_target = candidate_aliases.get(target_var, self.mind.canonicalize(target_var))
        canon_obs = self._canonical_obs(observations, extra_aliases=candidate_aliases)
        n = len(canon_obs[canon_target])
        idx_all = np.arange(n)
        tr_idx = idx_all[(idx_all % 10) < 7]
        va_idx = idx_all[(idx_all % 10) >= 7]

        # Diagnose baseline on validation split
        reached_before, before_rmse, before_r2 = self._evaluate_graph_on_indices(
            self.mind, canon_obs, canon_target, va_idx
        )

        # Verify qualitative academic claims empirically (records both verified & falsified)
        rejected_list: List[Dict[str, Any]] = []
        for qop in acad_qual_claims:
            qres = OpenAcademicCausalizer.verify_qualitative_claim(qop, canon_obs)
            self.qualitative_verdicts.append(qres)
            if qres["falsified"]:
                rejected_list.append(
                    {
                        "operator_id": qop.operator_id,
                        "source_kind": qop.source_kind,
                        "formula": qop.formula_str,
                        "reason": "Empirical partial correlation contradicts qualitative claim",
                    }
                )

        # Step 2: Gather executable candidates matching target_var
        candidates: List[CausalEdgeOperator] = []
        for op in acad_eq_candidates:
            if op.output_var == canon_target:
                candidates.append(op)

        for cf in code_files:
            try:
                extracted = ASTCodeCausalizer.extract_operators_from_file(Path(cf), strict=False)
            except Exception as exc:
                rejected_list.append(
                    {"operator_id": str(cf), "source_kind": "external_code_ast", "formula": "", "reason": str(exc)}
                )
                continue
            for op in extracted:
                if op.output_var == canon_target or canon_target in op.output_aliases:
                    candidates.append(
                        CausalEdgeOperator(
                            operator_id=op.operator_id,
                            source_kind=op.source_kind,
                            source_locator=op.source_locator,
                            input_vars=op.input_vars,
                            output_var=canon_target,
                            formula_str=op.formula_str,
                            evaluator=op.evaluator,
                            mdl_complexity=op.mdl_complexity,
                            validation_r2=op.validation_r2,
                            ast_source=op.ast_source,
                            module_constants_snapshot=op.module_constants_snapshot,
                            output_aliases=op.output_aliases,
                        )
                    )

        if allow_data_discovery and len(tr_idx) >= 5:
            # Feed train-split data + already-inferred mind intermediates on train split!
            train_data = {k: arr[tr_idx] for k, arr in canon_obs.items()}
            # If intermediate concepts already learned in mind help bridge train_data -> target, enrich train_data
            extra_tr: Dict[str, List[float]] = {}
            for i in range(len(tr_idx)):
                row_in = {k: float(arr[i]) for k, arr in train_data.items() if k != canon_target}
                tr_inf = self.mind.infer(row_in, canon_target)
                for kv, vv in tr_inf.intermediate_values.items():
                    if kv not in train_data and isinstance(vv, (int, float, np.number)) and math.isfinite(float(vv)):
                        extra_tr.setdefault(kv, []).append(float(vv))
            for kv, lst in extra_tr.items():
                if len(lst) == len(tr_idx):
                    train_data[kv] = np.asarray(lst, dtype=np.float64)

            sym_op = SymbolicDataCausalizer.discover(
                train_data,
                canon_target,
                source_locator="empirical_observations",
            )
            if sym_op is not None:
                candidates.append(sym_op)

        # Step 3: Counterfactual Sandbox Test on Held-Out Validation Split
        evaluated: List[Tuple[CausalEdgeOperator, float, float]] = []
        for cand in candidates:
            sandbox = self.mind.clone()
            for al, canon in candidate_aliases.items():
                sandbox.register_alias(al, canon)
            sandbox.commit_operator(cand)
            ok, va_rmse, va_r2 = self._evaluate_graph_on_indices(
                sandbox, canon_obs, canon_target, va_idx
            )
            if ok and va_r2 >= min_r2_for_conviction:
                evaluated.append((cand, va_rmse, va_r2))
            else:
                rejected_list.append(
                    {
                        "operator_id": cand.operator_id,
                        "source_kind": cand.source_kind,
                        "formula": cand.formula_str,
                        "held_out_r2": float(va_r2),
                        "reason": (
                            "Execution/dependency unresolved on held-out observations"
                            if not ok
                            else f"Held-out R2={va_r2:.4f} below conviction bar {min_r2_for_conviction}"
                        ),
                    }
                )

        if not evaluated:
            stmt = (
                f"목표 '{canon_target}'에 대해 외부 후보 {len(candidates)}건과 정성 명제 {len(acad_qual_claims)}건을 "
                f"홀드아웃 실측에서 반사실적으로 검증했으나 납득할 근거(R²>={min_r2_for_conviction:.2f})에 "
                "도달한 원리가 없어 인지 구조 개변을 거부했습니다."
            )
            proof = ConvictionProof(
                target_var=canon_target,
                convinced=False,
                rewired=False,
                chosen_operator_id=None,
                source_kind="none",
                source_locator="",
                formula_str="",
                before_rmse=before_rmse,
                after_rmse=before_rmse,
                held_out_r2=before_r2 if reached_before else -float("inf"),
                mdl_complexity=0.0,
                rejected_candidates=rejected_list,
                conviction_statement_ko=stmt,
            )
            self.conviction_history.append(proof)
            return proof

        # Choose highest generalization, breaking near-ties cleanly by lowest MDL complexity
        best_r2_top = max(r2 for _, _, r2 in evaluated)
        near_top = [item for item in evaluated if item[2] >= best_r2_top - 1e-3]
        winner, after_rmse, after_r2 = min(
            near_top, key=lambda item: (item[0].mdl_complexity, item[1])
        )

        stmt = (
            f"내 인지망의 공백 '{canon_target}'(기존 도달={reached_before}, RMSE={before_rmse:.4g})을 "
            f"외부 원리 [{winner.source_kind}] `{winner.formula_str}` 로 연결해 홀드아웃 검증을 수행한 결과 "
            f"오차가 {after_rmse:.4g}(R²={after_r2:.5f}, MDL={int(winner.mdl_complexity)})로 붕괴해 "
            "스스로 납득하고 내 인지·감각 그래프를 영구 개변했습니다."
        )
        proof = ConvictionProof(
            target_var=canon_target,
            convinced=True,
            rewired=True,
            chosen_operator_id=winner.operator_id,
            source_kind=winner.source_kind,
            source_locator=winner.source_locator,
            formula_str=winner.formula_str,
            before_rmse=before_rmse,
            after_rmse=after_rmse,
            held_out_r2=after_r2,
            mdl_complexity=winner.mdl_complexity,
            rejected_candidates=rejected_list,
            conviction_statement_ko=stmt,
        )
        winner.validation_r2 = after_r2
        winner.conviction_proof = proof

        # Commit genuine permanent self-modification!
        for al, canon in candidate_aliases.items():
            self.mind.register_alias(al, canon)
        self.mind.commit_operator(winner)
        self.conviction_history.append(proof)

        # Anchor autobiographical witness if LivingCycle is attached
        if self.living_cycle is not None:
            w = ExtendedWorldState(source_pos_x=self.living_cycle.organism.body_pos_x + 10.0, source_temp_k=3200.0)
            self.living_cycle.breathe_and_understand(
                LivingEncounter(
                    encounter_name=f"external_convict_{canon_target}",
                    world=w,
                    active_organs=("eye", "ear", "hand"),
                )
            )
            if self.living_cycle.memory_runtime is not None:
                t = np.linspace(0.0, 1.0, 64, endpoint=False)
                sig = np.sin(2.0 * math.pi * (1.0 + len(self.mind.operators)) * t) + 0.3 * np.cos(
                    2.0 * math.pi * after_r2 * 5.0 * t
                )
                man = encode_waveform_to_scale_rotors(sig, domain=(0.0, 1.0), max_macro_modes=6, max_levels=2, max_knots_per_level=4)
                self.living_cycle.memory_runtime.ingest_to_ice(f"mind_rewire_{canon_target}", man)

        return proof

    def bridge_multi_hop_goal(
        self,
        stage_specs: Sequence[Dict[str, Any]],
        end_to_end_inputs: Dict[str, float],
        final_target: str,
        sensitivity_input: Optional[str] = None,
    ) -> MultiHopLearningReport:
        before_trace = self.mind.infer(end_to_end_inputs, final_target)
        proofs: List[ConvictionProof] = []
        for sp in stage_specs:
            pr = self.learn_from_external_sources(
                target_var=sp["target_var"],
                observations=sp["observations"],
                code_files=sp.get("code_files", ()),
                academic_texts=sp.get("academic_texts", ()),
                allow_data_discovery=bool(sp.get("allow_data_discovery", True)),
                min_r2_for_conviction=float(sp.get("min_r2_for_conviction", 0.92)),
            )
            proofs.append(pr)

        after_trace = self.mind.infer(end_to_end_inputs, final_target)
        sens = (
            self.mind.compute_sensitivity(end_to_end_inputs, sensitivity_input, final_target)
            if sensitivity_input is not None
            else None
        )
        summary = (
            f"이질적 외부 원천({len(stage_specs)}단계: 코드·데이터·학문)을 인과화해 "
            f"'{final_target}' 도달 상태({before_trace.reached} -> {after_trace.reached}, "
            f"경로={len(after_trace.fired_chain)}홉)로 확장했습니다."
        )
        return MultiHopLearningReport(
            final_target=self.mind.canonicalize(final_target),
            reached_before=before_trace.reached,
            reached_after=after_trace.reached,
            stage_proofs=proofs,
            inference_trace=after_trace,
            end_to_end_sensitivity=sens,
            summary_ko=summary,
        )

    def sync_mind_to_causal_field(
        self,
        causal_field: Optional["CausalField"] = None,
        prefix: str = "mind",
    ) -> Dict[str, Any]:
        """Projects every learned concept node & causal operator into `CausalField` (4D tensors)."""
        from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

        field = causal_field if causal_field is not None else CausalField(dimensions=3)
        vids: Dict[str, str] = {}
        nodes = list(self.mind.nodes.values())
        ops = list(self.mind.operators.values())

        for idx, node in enumerate(nodes):
            ang = (2.0 * math.pi * idx) / max(1, len(nodes))
            vid = f"{prefix}_node_{node.name}"
            vids[f"node:{node.name}"] = vid
            field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={"concept": node.name, "role": node.role, "origin": node.origin_source},
                    tensor=np.array(
                        [1.0, float(len(node.aliases)), float(idx) / 10.0, 1.0],
                        dtype=np.float32,
                    ),
                    mass=1.4 if node.role == "derived" else 1.0,
                    position=np.array(
                        [2.0 * math.cos(ang), 2.0 * math.sin(ang), 0.5], dtype=np.float32
                    )[: field.dimensions],
                    curiosity_charge=0.2 if node.role == "derived" else 0.0,
                )
            )

        links = 0
        for idx, op in enumerate(ops):
            ang = (2.0 * math.pi * idx) / max(1, len(ops))
            vid = f"{prefix}_op_{idx}"
            vids[f"op:{op.operator_id}"] = vid
            field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={"operator_id": op.operator_id, "kind": op.source_kind, "formula": op.formula_str},
                    tensor=np.array(
                        [
                            float(op.validation_r2),
                            float(op.mdl_complexity) / 10.0,
                            float(len(op.input_vars)) / 5.0,
                            1.0,
                        ],
                        dtype=np.float32,
                    ),
                    mass=1.8,
                    position=np.array(
                        [1.0 * math.cos(ang), 1.0 * math.sin(ang), 1.6], dtype=np.float32
                    )[: field.dimensions],
                    curiosity_charge=max(0.0, 1.0 - float(op.validation_r2)),
                )
            )
            strength = float(max(0.08, min(1.0, op.validation_r2)))
            for iv in op.input_vars:
                nk = f"node:{iv}"
                if nk in vids:
                    field.link_voxels(vids[nk], vid, strength=strength)
                    links += 1
            ok = f"node:{op.output_var}"
            if ok in vids:
                field.link_voxels(vid, vids[ok], strength=strength)
                links += 1

        field.register_engram(
            EngramAttractor(
                id=f"{prefix}_engram_open_mind",
                name="OpenExternalCausalMind",
                position=np.array([0.0, 0.0, 1.6][: field.dimensions], dtype=np.float32),
                intensity=2.4,
                tier="macro",
                symbolic_intent=f"Concepts={len(nodes)}, Operators={len(ops)}",
            )
        )
        return {
            "causal_field": field,
            "voxels": vids,
            "concept_count": len(nodes),
            "operator_count": len(ops),
            "beams_linked": links,
        }

    def save_mind(self, path: Path | str) -> Path:
        fpath = Path(path)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        payload = self.mind.to_dict()
        fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return fpath

    def load_mind(self, path: Path | str) -> Dict[str, Any]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.mind = CausalMindGraph.from_dict(data)
        return {
            "concepts": len(self.mind.nodes),
            "operators": len(self.mind.operators),
            "aliases": len(self.mind.alias_map),
        }
