r"""
ELYSIA Template-Free Compositional Code Understanding & AST Synthesis Cortex
==========================================================================

Teaches Elysia **what Python code is made of** across the 4 structural axes
(Atoms · Principle · Arrangement · Environment) directly from real `.py` files,
and enables genuine template-free synthesis of new `ast.AST` executables:

1. **Decompose (`learn_python_structure_from_files`):** Walks real `.py` ASTs to
   harvest atomic transforms/operators (`CodeAtom`) and control-flow topologies
   (`StructuralArrangement`).
2. **Compose Transforms (`synthesize_transform_ast`):** Searches bounded,
   type-guided AST compositions strictly from learned atoms (MDL smallest AST)
   and emits executable Python via `ast.unparse` without canned templates.
3. **Compose Conditional Trees & Surgical AST Mutation (`synthesize_decision_tree_ast`):**
   Induces real `ast.If` decision trees from predicates in the examples and
   surgically splices them into existing `CognitiveOrgan` source code while
   keeping `must_stay_same` invariant guards intact.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.embodiment.self_rewriting_architecture import (
    _compile_organ_callable,
    validate_organ_source,
)


_BUILTIN_UNARY_SPECS: Dict[str, Tuple[str, str, Callable[[Any], Any]]] = {
    "len": ("seq_or_str", "int", lambda x: len(x)),
    "sum": ("seq_num", "num", lambda x: sum(x)),
    "sorted": ("seq", "list", lambda x: sorted(x)),
    "set": ("seq", "set", lambda x: set(x)),
    "min": ("seq_num", "num", lambda x: min(x)),
    "max": ("seq_num", "num", lambda x: max(x)),
    "abs": ("num", "num", lambda x: abs(x)),
}

_METHOD_UNARY_SPECS: Dict[str, Tuple[str, str, Callable[[Any], Any]]] = {
    "splitlines": ("str", "list", lambda s: s.splitlines()),
    "strip": ("str", "str", lambda s: s.strip()),
    "lower": ("str", "str", lambda s: s.lower()),
    "split": ("str", "list", lambda s: s.split()),
}

_BINOP_NODE_MAP: Dict[type, Tuple[str, str]] = {
    ast.Add: ("+", "num"),
    ast.Sub: ("-", "num"),
    ast.Mult: ("*", "num"),
    ast.Div: ("/", "num"),
    ast.FloorDiv: ("//", "num"),
    ast.BitAnd: ("&", "set"),
    ast.BitOr: ("|", "set"),
}

_CMP_NODE_MAP: Dict[type, str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Gt: ">",
    ast.Lt: "<",
    ast.In: "in",
    ast.NotIn: "not_in",
}


@dataclass
class CodeAtom:
    """Single atomic operation discovered by parsing real Python source ASTs."""

    atom_id: str
    kind: str  # "unary_transform" | "method_transform" | "binary_op" | "predicate"
    op_name: str
    input_arity: int
    input_type: str
    output_type: str
    frequency: int = 0
    source_files: List[str] = field(default_factory=list)


@dataclass
class StructuralArrangement:
    """Control/data topology observed inside real Python ASTs."""

    arrangement_id: str
    topology: str  # "guard_return" | "if_else_split" | "pipeline" | "dict_assembly"
    observed_count: int = 0
    source_files: List[str] = field(default_factory=list)


@dataclass
class SynthesizedProgram:
    """Result of template-free AST composition emitted via `ast.unparse`."""

    program_id: str
    purpose: str
    source_code: str
    used_atoms: List[str]
    arrangement_used: str
    examples_passed: int
    examples_total: int
    ast_node_count: int


def _safe_type_tag(val: Any) -> str:
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, (int, float)):
        return "num"
    if isinstance(val, str):
        return "str"
    if isinstance(val, set):
        return "set"
    if isinstance(val, (list, tuple)):
        return "list"
    return "other"


def _sub_expr_key(key_name: str) -> ast.Subscript:
    return ast.Subscript(
        value=ast.Name(id="percept", ctx=ast.Load()),
        slice=ast.Constant(value=key_name),
        ctx=ast.Load(),
    )


def _eval_binary(op_sym: str, a: Any, b: Any) -> Any:
    if op_sym == "+":
        return a + b
    if op_sym == "-":
        return a - b
    if op_sym == "*":
        return a * b
    if op_sym == "/":
        return a / b
    if op_sym == "//":
        return a // b
    if op_sym == "&":
        return a & b
    if op_sym == "|":
        return a | b
    raise ValueError(op_sym)


def _make_binop_ast(op_sym: str, left: ast.expr, right: ast.expr) -> ast.BinOp:
    rev = {v[0]: k for k, v in _BINOP_NODE_MAP.items()}
    return ast.BinOp(left=left, op=rev[op_sym](), right=right)


def _dict_to_ast(d: Dict[str, Any]) -> ast.Dict:
    keys: List[Optional[ast.expr]] = [ast.Constant(value=k) for k in d.keys()]
    vals: List[ast.expr] = []
    for v in d.values():
        if isinstance(v, list):
            vals.append(ast.List(elts=[ast.Constant(value=x) for x in v], ctx=ast.Load()))
        else:
            vals.append(ast.Constant(value=v))
    return ast.Dict(keys=keys, values=vals)


class CompositionalCodeCortex:
    """
    Learns atomic code constituents & arrangements directly from real `.py` ASTs,
    and composes genuinely new executable Python ASTs without template strings.
    """

    def __init__(self) -> None:
        self.learned_atoms: Dict[str, CodeAtom] = {}
        self.learned_arrangements: Dict[str, StructuralArrangement] = {}
        self.synthesized_history: List[SynthesizedProgram] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learned_atoms": {k: asdict(v) for k, v in self.learned_atoms.items()},
            "learned_arrangements": {k: asdict(v) for k, v in self.learned_arrangements.items()},
            "synthesized_history": [asdict(s) for s in self.synthesized_history],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CompositionalCodeCortex":
        obj = cls()
        for k, v in (raw.get("learned_atoms") or {}).items():
            obj.learned_atoms[str(k)] = CodeAtom(**v)
        for k, v in (raw.get("learned_arrangements") or {}).items():
            obj.learned_arrangements[str(k)] = StructuralArrangement(**v)
        for s in raw.get("synthesized_history") or []:
            obj.synthesized_history.append(SynthesizedProgram(**s))
        return obj

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "CompositionalCodeCortex":
        p = Path(path)
        if not p.is_file():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def _register_atom(
        self,
        atom_id: str,
        kind: str,
        op_name: str,
        arity: int,
        in_type: str,
        out_type: str,
        filename: str,
    ) -> None:
        atom = self.learned_atoms.get(atom_id)
        if atom is None:
            atom = CodeAtom(
                atom_id=atom_id,
                kind=kind,
                op_name=op_name,
                input_arity=arity,
                input_type=in_type,
                output_type=out_type,
                frequency=0,
                source_files=[],
            )
            self.learned_atoms[atom_id] = atom
        atom.frequency += 1
        if filename not in atom.source_files and len(atom.source_files) < 12:
            atom.source_files.append(filename)

    def _register_arrangement(self, topology: str, filename: str) -> None:
        arr = self.learned_arrangements.get(topology)
        if arr is None:
            arr = StructuralArrangement(
                arrangement_id=f"arr::{topology}",
                topology=topology,
                observed_count=0,
                source_files=[],
            )
            self.learned_arrangements[topology] = arr
        arr.observed_count += 1
        if filename not in arr.source_files and len(arr.source_files) < 12:
            arr.source_files.append(filename)

    def learn_python_structure_from_files(self, paths: Sequence[Path | str]) -> Dict[str, Any]:
        """Decomposes real `.py` files into atomic operations and structural arrangements."""
        files_parsed = 0
        for p_item in paths:
            p = Path(p_item)
            if not p.is_file():
                continue
            try:
                src = p.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(src)
            except SyntaxError:
                continue
            files_parsed += 1
            fname = p.name

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in _BUILTIN_UNARY_SPECS:
                        it, ot, _ = _BUILTIN_UNARY_SPECS[node.func.id]
                        self._register_atom(f"call::{node.func.id}", "unary_transform", node.func.id, 1, it, ot, fname)
                    elif isinstance(node.func, ast.Attribute):
                        m_name = node.func.attr
                        if m_name in _METHOD_UNARY_SPECS:
                            it, ot, _ = _METHOD_UNARY_SPECS[m_name]
                            self._register_atom(f"method::{m_name}", "method_transform", m_name, 1, it, ot, fname)
                        elif m_name in ("startswith", "endswith"):
                            self._register_atom(f"pred::{m_name}", "predicate", m_name, 2, "str", "bool", fname)
                    if any(isinstance(a, ast.Call) for a in node.args) or (
                        isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call)
                    ):
                        self._register_arrangement("pipeline", fname)

                elif isinstance(node, ast.BinOp):
                    spec = _BINOP_NODE_MAP.get(type(node.op))
                    if spec is not None:
                        sym, kind_t = spec
                        self._register_atom(f"binop::{sym}", "binary_op", sym, 2, kind_t, kind_t, fname)

                elif isinstance(node, ast.Compare):
                    for op_n in node.ops:
                        sym = _CMP_NODE_MAP.get(type(op_n))
                        if sym is not None:
                            self._register_atom(f"pred::{sym}", "predicate", sym, 2, "any", "bool", fname)

                elif isinstance(node, ast.If):
                    has_ret = any(isinstance(st, ast.Return) for st in node.body)
                    if has_ret and not node.orelse:
                        self._register_arrangement("guard_return", fname)
                    elif node.orelse:
                        self._register_arrangement("if_else_split", fname)

                elif isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                    self._register_arrangement("dict_assembly", fname)

        return {
            "files_parsed": files_parsed,
            "learned_atoms_count": len(self.learned_atoms),
            "learned_arrangements_count": len(self.learned_arrangements),
        }

    def synthesize_transform_ast(
        self,
        examples: Sequence[Tuple[Dict[str, Any], Any]],
        *,
        purpose: str = "transform",
        max_depth: int = 3,
    ) -> Optional[SynthesizedProgram]:
        """
        Synthesizes an executable Python transform purely from `self.learned_atoms`
        via type-guided bounded AST search + MDL selection + `ast.unparse`.
        """
        if not examples or not self.learned_atoms:
            return None

        target_vals = [ex[1] for ex in examples]
        input_keys = sorted({k for ex in examples for k in ex[0].keys()})
        if not input_keys:
            return None

        # Candidate entry tuple: (ast_expr, values_list, type_tag, ast_size, atoms_used)
        pool: List[Tuple[ast.expr, List[Any], str, int, Tuple[str, ...]]] = []
        seen_sigs: set[str] = set()

        def _try_add(node: ast.expr, vals: List[Any], size: int, atoms: Tuple[str, ...]) -> None:
            sig = "|".join(repr(v) for v in vals)
            if sig in seen_sigs:
                return
            seen_sigs.add(sig)
            t_tag = _safe_type_tag(vals[0])
            pool.append((node, vals, t_tag, size, atoms))

        # Depth 0: input fields + small structural integer constants (0, 1, 2)
        for k in input_keys:
            vals = [ex[0].get(k) for ex in examples]
            _try_add(_sub_expr_key(k), vals, 1, ())
        for c_int in (1, 2):
            _try_add(ast.Constant(value=c_int), [c_int] * len(examples), 1, ())

        avail_unary = [a for a in self.learned_atoms.values() if a.kind == "unary_transform"]
        avail_method = [a for a in self.learned_atoms.values() if a.kind == "method_transform"]
        avail_binops = [a for a in self.learned_atoms.values() if a.kind == "binary_op"]

        # Iteratively expand AST expressions using ONLY learned atoms
        for _depth in range(max_depth):
            snapshot = list(pool)

            # Unary builtin calls & string method calls
            for expr_n, vals, t_tag, sz, atoms in snapshot:
                for ua in avail_unary:
                    in_t, _out_t, fn = _BUILTIN_UNARY_SPECS[ua.op_name]
                    if in_t == "seq_or_str" and t_tag not in ("str", "list", "set"):
                        continue
                    if in_t in ("seq", "seq_num") and t_tag not in ("list", "set"):
                        continue
                    if in_t == "num" and t_tag != "num":
                        continue
                    try:
                        new_vals = [fn(v) for v in vals]
                    except Exception:
                        continue
                    new_ast = ast.Call(func=ast.Name(id=ua.op_name, ctx=ast.Load()), args=[copy.deepcopy(expr_n)], keywords=[])
                    _try_add(new_ast, new_vals, sz + 1, (*atoms, ua.atom_id))

                if t_tag == "str":
                    for ma in avail_method:
                        _it, _ot, fn = _METHOD_UNARY_SPECS[ma.op_name]
                        try:
                            new_vals = [fn(v) for v in vals]
                        except Exception:
                            continue
                        new_ast = ast.Call(
                            func=ast.Attribute(value=copy.deepcopy(expr_n), attr=ma.op_name, ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        )
                        _try_add(new_ast, new_vals, sz + 1, (*atoms, ma.atom_id))

            # Binary operators between compatible types
            for i, (e1, v1, t1, s1, a1) in enumerate(snapshot):
                for j, (e2, v2, t2, s2, a2) in enumerate(snapshot):
                    if s1 + s2 > 5:
                        continue
                    for ba in avail_binops:
                        sym = ba.op_name
                        if sym in ("&", "|") and not (t1 == "set" and t2 == "set"):
                            continue
                        if sym == "-" and not ((t1 == "num" and t2 == "num") or (t1 == "set" and t2 == "set")):
                            continue
                        if sym in ("+", "*", "/", "//") and not (t1 == "num" and t2 == "num"):
                            if not (sym == "+" and t1 == t2 == "str"):
                                continue
                        if i == j and sym in ("-", "/", "//", "&", "|"):
                            continue
                        try:
                            new_vals = [_eval_binary(sym, x, y) for x, y in zip(v1, v2)]
                        except Exception:
                            continue
                        new_ast = _make_binop_ast(sym, copy.deepcopy(e1), copy.deepcopy(e2))
                        _try_add(new_ast, new_vals, s1 + s2 + 1, (*a1, *a2, ba.atom_id))

            # Check if smallest match already found
            matches = [item for item in pool if item[1] == target_vals]
            if matches:
                break

        matches = [item for item in pool if item[1] == target_vals]
        if not matches:
            return None

        # MDL: pick expression with minimal AST size
        matches.sort(key=lambda item: (item[3], len(item[4])))
        best_expr, _vals, _t, best_sz, best_atoms = matches[0]

        fn_def = ast.FunctionDef(
            name="process",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="percept"), ast.arg(arg="memory")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[
                ast.Return(
                    value=ast.Dict(
                        keys=[ast.Constant(value="result")],
                        values=[best_expr],
                    )
                )
            ],
            decorator_list=[],
        )
        mod = ast.Module(body=[fn_def], type_ignores=[])
        ast.fix_missing_locations(mod)
        src_code = ast.unparse(mod)

        ok_ast, _ = validate_organ_source(src_code)
        if not ok_ast:
            return None
        fn = _compile_organ_callable(src_code)
        passed = 0
        for p_in, exp_val in examples:
            out = fn(copy.deepcopy(p_in), {})
            if out.get("result") == exp_val:
                passed += 1
        if passed != len(examples):
            return None

        prog = SynthesizedProgram(
            program_id=f"prog_{len(self.synthesized_history) + 1:04d}",
            purpose=purpose,
            source_code=src_code,
            used_atoms=list(dict.fromkeys(best_atoms)),
            arrangement_used="pipeline" if len(best_atoms) >= 2 else "dict_assembly",
            examples_passed=passed,
            examples_total=len(examples),
            ast_node_count=best_sz,
        )
        self.synthesized_history.append(prog)
        return prog

    def _discover_candidate_predicates(
        self,
        examples: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    ) -> List[Tuple[str, Callable[[Dict[str, Any]], bool], Callable[[], ast.expr], Tuple[str, ...]]]:
        """Discovers candidate boolean AST predicates dynamically from the examples & learned atoms."""
        preds: List[Tuple[str, Callable[[Dict[str, Any]], bool], Callable[[], ast.expr], Tuple[str, ...]]] = []
        has_startswith = "pred::startswith" in self.learned_atoms
        has_in = "pred::in" in self.learned_atoms
        has_gt = "pred::>" in self.learned_atoms
        has_eq = "pred::==" in self.learned_atoms
        if not (has_startswith or has_in or has_gt or has_eq):
            return preds

        all_keys = sorted({k for ex in examples for k in ex[0].keys()})
        common_keys = [k for k in all_keys if all(k in ex[0] for ex in examples)]
        seen_names: set[str] = set()

        # 1. String structural predicates (`startswith`, `in`)
        str_keys = [k for k in common_keys if all(isinstance(ex[0].get(k), str) for ex in examples)]
        for k in str_keys:
            texts = [str(ex[0].get(k, "")) for ex in examples]
            prefixes: set[str] = set()
            tokens: set[str] = set()
            for t in texts:
                st = t.strip()
                if st:
                    prefixes.add(st[:1])
                    first_w = st.split()[0]
                    if 1 <= len(first_w) <= 6:
                        prefixes.add(first_w)
                for ch in ("#", "[", "=", ":"):
                    if ch in t:
                        tokens.add(ch)
                for w in t.replace("\n", " ").split(" "):
                    if 2 <= len(w) <= 10 and w.isascii():
                        tokens.add(w)
                        if w.isalpha() and len(w) <= 8 and (f"{w} " in t):
                            tokens.add(f"{w} ")

            if has_startswith:
                for pref in sorted(prefixes):
                    bools = [str(ex[0].get(k, "")).strip().startswith(pref) for ex in examples]
                    if len(set(bools)) < 2:
                        continue
                    pname = f"{k}.strip().startswith({pref!r})"
                    if pname in seen_names:
                        continue
                    seen_names.add(pname)
                    preds.append(
                        (
                            pname,
                            (lambda p, key=k, pr=pref: str(p.get(key, "")).strip().startswith(pr)),
                            (
                                lambda key=k, pr=pref: ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Call(
                                            func=ast.Attribute(value=_sub_expr_key(key), attr="strip", ctx=ast.Load()),
                                            args=[],
                                            keywords=[],
                                        ),
                                        attr="startswith",
                                        ctx=ast.Load(),
                                    ),
                                    args=[ast.Constant(value=pr)],
                                    keywords=[],
                                )
                            ),
                            ("method::strip", "pred::startswith"),
                        )
                    )

            if has_in:
                for tok in sorted(tokens):
                    bools = [tok in str(ex[0].get(k, "")) for ex in examples]
                    if len(set(bools)) < 2:
                        continue
                    pname = f"{tok!r} in {k}"
                    if pname in seen_names:
                        continue
                    seen_names.add(pname)
                    preds.append(
                        (
                            pname,
                            (lambda p, key=k, tk=tok: tk in str(p.get(key, ""))),
                            (
                                lambda key=k, tk=tok: ast.Compare(
                                    left=ast.Constant(value=tk),
                                    ops=[ast.In()],
                                    comparators=[_sub_expr_key(key)],
                                )
                            ),
                            ("pred::in",),
                        )
                    )

            # Categorical equality for short string tags
            if has_eq and all(len(t) <= 24 and "\n" not in t for t in texts):
                for val in sorted(set(texts)):
                    bools = [str(ex[0].get(k, "")) == val for ex in examples]
                    if len(set(bools)) < 2:
                        continue
                    pname = f"{k} == {val!r}"
                    if pname in seen_names:
                        continue
                    seen_names.add(pname)
                    preds.append(
                        (
                            pname,
                            (lambda p, key=k, target_v=val: str(p.get(key, "")) == target_v),
                            (
                                lambda key=k, target_v=val: ast.Compare(
                                    left=_sub_expr_key(key),
                                    ops=[ast.Eq()],
                                    comparators=[ast.Constant(value=target_v)],
                                )
                            ),
                            ("pred::==",),
                        )
                    )

        # 2. Numeric midpoint threshold predicates (`percept[k] > mid`)
        if has_gt:
            num_keys = [
                k for k in common_keys
                if all(isinstance(ex[0].get(k), (int, float)) and not isinstance(ex[0].get(k), bool) for ex in examples)
            ]
            for k in num_keys:
                uniq_vals = sorted({float(ex[0][k]) for ex in examples})
                for idx in range(len(uniq_vals) - 1):
                    mid = round((uniq_vals[idx] + uniq_vals[idx + 1]) / 2.0, 5)
                    bools = [float(ex[0][k]) > mid for ex in examples]
                    if len(set(bools)) < 2:
                        continue
                    pname = f"{k} > {mid}"
                    if pname in seen_names:
                        continue
                    seen_names.add(pname)
                    preds.append(
                        (
                            pname,
                            (lambda p, key=k, thr=mid: float(p.get(key, 0.0)) > thr),
                            (
                                lambda key=k, thr=mid: ast.Compare(
                                    left=_sub_expr_key(key),
                                    ops=[ast.Gt()],
                                    comparators=[ast.Constant(value=thr)],
                                )
                            ),
                            ("pred::>",),
                        )
                    )

        # 3. Boolean equality predicates
        if has_eq:
            bool_keys = [k for k in common_keys if all(isinstance(ex[0].get(k), bool) for ex in examples)]
            for k in bool_keys:
                for bval in (True, False):
                    bools = [ex[0].get(k) == bval for ex in examples]
                    if len(set(bools)) < 2:
                        continue
                    pname = f"{k} == {bval}"
                    if pname in seen_names:
                        continue
                    seen_names.add(pname)
                    preds.append(
                        (
                            pname,
                            (lambda p, key=k, bv=bval: p.get(key) == bv),
                            (
                                lambda key=k, bv=bval: ast.Compare(
                                    left=_sub_expr_key(key),
                                    ops=[ast.Eq()],
                                    comparators=[ast.Constant(value=bv)],
                                )
                            ),
                            ("pred::==",),
                        )
                    )
        return preds

    def synthesize_decision_tree_ast(
        self,
        examples: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
        *,
        base_organ_source: str = "",
        purpose: str = "decision_tree",
    ) -> Optional[SynthesizedProgram]:
        """
        Synthesizes an `ast.If` decision tree strictly from discovered predicates and
        learned atoms, optionally splicing it directly into an existing `CognitiveOrgan` AST.
        """
        if not examples or not self.learned_atoms:
            return None

        preds = self._discover_candidate_predicates(examples)
        used_atoms: List[str] = []

        def _all_equal_outputs(subset: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]) -> bool:
            first = subset[0][1]
            return all(item[1] == first for item in subset[1:])

        def _build_stmts(subset: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], depth: int) -> List[ast.stmt]:
            if _all_equal_outputs(subset) or depth >= 5:
                return [ast.Return(value=_dict_to_ast(subset[0][1]))]

            best_pred = None
            best_score = 1e9
            best_pos: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            best_neg: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

            for p_name, eval_fn, ast_fn, p_atoms in preds:
                pos = [ex for ex in subset if eval_fn(ex[0])]
                neg = [ex for ex in subset if not eval_fn(ex[0])]
                if not pos or not neg:
                    continue
                # Prefer pure splits; structurally anchored predicates (startswith / multi-char words) generalize better than lone punctuation
                n_out_pos = len({json.dumps(x[1], sort_keys=True, ensure_ascii=False) for x in pos})
                n_out_neg = len({json.dumps(x[1], sort_keys=True, ensure_ascii=False) for x in neg})
                spec_bias = -0.05 if "startswith" in p_name else (0.04 if "':' in " in p_name or "'=' in " in p_name else 0.0)
                score = float(n_out_pos + n_out_neg) + spec_bias + 0.002 * len(p_name)
                if score < best_score:
                    best_score = score
                    best_pred = (eval_fn, ast_fn, p_atoms)
                    best_pos = pos
                    best_neg = neg

            if best_pred is None:
                return [ast.Return(value=_dict_to_ast(subset[0][1]))]

            _eval_fn, ast_fn, p_atoms = best_pred
            for a in p_atoms:
                if a not in used_atoms:
                    used_atoms.append(a)
            return [
                ast.If(
                    test=ast_fn(),
                    body=_build_stmts(best_pos, depth + 1),
                    orelse=_build_stmts(best_neg, depth + 1),
                )
            ]

        tree_stmts = _build_stmts(examples, 0)

        if base_organ_source.strip():
            base_mod = ast.parse(base_organ_source)
            proc_fn = next(
                (n for n in base_mod.body if isinstance(n, ast.FunctionDef) and n.name == "process"),
                None,
            )
            if proc_fn is None:
                return None
            # Surgical AST splice (idempotent across v1->v2->v3): keep leading Assigns & `familiar` guard only
            kept_prefix: List[ast.stmt] = []
            for stmt in proc_fn.body:
                if isinstance(stmt, ast.Assign):
                    kept_prefix.append(stmt)
                elif isinstance(stmt, ast.If) and "familiar" in ast.unparse(stmt):
                    kept_prefix.append(stmt)
                else:
                    break
            proc_fn.body = [*kept_prefix, *tree_stmts]
            mod = base_mod
        else:
            fn_def = ast.FunctionDef(
                name="process",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="percept"), ast.arg(arg="memory")],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=tree_stmts,
                decorator_list=[],
            )
            mod = ast.Module(body=[fn_def], type_ignores=[])

        ast.fix_missing_locations(mod)
        src_code = ast.unparse(mod)
        ok_ast, _ = validate_organ_source(src_code)
        if not ok_ast:
            return None

        fn = _compile_organ_callable(src_code)
        passed = 0
        for p_in, exp_dict in examples:
            out = dict(fn(copy.deepcopy(p_in), {"known_items": []}) or {})
            if out == exp_dict:
                passed += 1
        if passed != len(examples):
            return None

        prog = SynthesizedProgram(
            program_id=f"prog_{len(self.synthesized_history) + 1:04d}",
            purpose=purpose,
            source_code=src_code,
            used_atoms=used_atoms,
            arrangement_used="if_else_split",
            examples_passed=passed,
            examples_total=len(examples),
            ast_node_count=sum(1 for _ in ast.walk(mod)),
        )
        self.synthesized_history.append(prog)
        return prog
