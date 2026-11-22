"""
hypothesis_space_organ.py

Hypothesis-Space Organ: seeing "what I can / cannot distinguish" and exploring programs to fill the gap
=======================================================================================================
Missing so far: ELYSIA could only score candidate programs that a human wrote down. Here the candidate
space itself becomes something she perceives and explores:

1. Operator library is DISCOVERED, not listed:
   - numpy ufunc registry (mechanically enumerated, typed by actually calling each operator on probe data)
   - primitives mined from ELYSIA's own organ source via AST (e.g. rfft, gradient), typed by probing
   - a minimal "body plan" (part/whole/order): tail, init, last, first, split_k, slope, spread
2. Layered grammar (hierarchy of structure):
   select/partition -> transform -> partition -> pointwise -> pool/reduce -> aggregate -> relation
3. Blind-spot map: from the columns reachable WITHOUT a structural layer, measure which class pairs are
   still indistinguishable. Those pairs become EpistemicVoids and drive the search.
4. Greedy residual cover: pick the smallest program that separates the still-blind pairs, validated on a
   separate perturbation condition; selected programs grow the library.
5. Relation stage: finds pairs of quantities that agree exactly on a class (an invariant), without any
   hand-written checksum candidates.
"""

from __future__ import annotations

import ast
import hashlib
import math
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.sensory.epistemic_knowledge_spine import EpistemicKnowledgeSpine, SynthesizedOrganRecord

SENSORY_DIR = Path(r"c:\Elysia\L1_causal_field\core\sensory")
ORGAN_SOURCES = ("mechano_acoustic_organ.py", "ocular_scale_organ.py")


# ----------------------------------------------------------------------------
# Data helpers (regular 2D ndarray rows, or ragged list of 1D rows)
# ----------------------------------------------------------------------------
def _rows(fn: Callable[[np.ndarray], Any], data: Any) -> Any:
    if isinstance(data, np.ndarray):
        return fn(data)
    outs = [fn(np.asarray(r)) for r in data]
    try:
        return np.stack(outs)
    except ValueError:
        return outs


def _all_finite(data: Any) -> bool:
    if isinstance(data, np.ndarray):
        return bool(np.all(np.isfinite(data)))
    if isinstance(data, list):
        return all(_all_finite(d) for d in data)
    return True


def _binary(fn: Callable, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(fn(a, b), dtype=np.float64)
    except TypeError:
        return np.asarray(fn(a.astype(np.int64), b.astype(np.int64)), dtype=np.float64)


@dataclass
class Op:
    name: str
    in_kind: str
    out_kind: str
    fn: Callable[[Any], Any]
    origin: str


@dataclass(frozen=True)
class Program:
    names: Tuple[str, ...] = ()
    reducer: str = ""
    agg: Optional[str] = None
    post: Optional[str] = None
    rel: Optional[Tuple[str, "Program", "Program"]] = None

    @property
    def size(self) -> int:
        if self.rel is not None:
            return 1 + self.rel[1].size + self.rel[2].size
        return len(self.names) + 1 + (1 if self.agg else 0) + (1 if self.post else 0)

    def expr(self) -> str:
        if self.rel is not None:
            return f"{self.rel[0]}({self.rel[1].expr()}, {self.rel[2].expr()})"
        e = "x"
        for n in self.names:
            e = f"{n}({e})"
        e = f"{self.reducer}[{e}]"
        if self.agg:
            e = f"{self.agg}({e})"
        if self.post:
            e = f"{self.post}({e})"
        return e

    def uses_partition(self) -> bool:
        if self.rel is not None:
            return self.rel[1].uses_partition() or self.rel[2].uses_partition()
        return any(n.startswith("split") for n in self.names)


# ----------------------------------------------------------------------------
# 1. Operator library discovery
# ----------------------------------------------------------------------------
@dataclass
class OperatorLibrary:
    ops: Dict[str, Op] = field(default_factory=dict)
    transforms: List[str] = field(default_factory=list)
    pointwise: List[str] = field(default_factory=list)
    reducers: List[str] = field(default_factory=list)
    relation_ops: Dict[str, Callable] = field(default_factory=dict)
    organ_names_found: List[str] = field(default_factory=list)
    learned_programs: List[Program] = field(default_factory=list)

    def size(self) -> int:
        return len(self.ops) + len(self.relation_ops) + len(self.learned_programs)


def _reduce_fn(u: np.ufunc) -> Callable[[np.ndarray], Any]:
    def fn(a: np.ndarray, u: np.ufunc = u) -> Any:
        try:
            return u.reduce(a, axis=-1)
        except TypeError:
            return u.reduce(np.asarray(a).astype(np.int64), axis=-1)

    return fn


def discover_operator_library(organ_paths: Sequence[Path]) -> OperatorLibrary:
    lib = OperatorLibrary()
    probe = np.random.default_rng(0).uniform(0.5, 3.0, (4, 32))
    sorted_row = np.linspace(0.5, 3.0, 32)

    organ_names: set = set()
    for p in organ_paths:
        if not Path(p).exists():
            continue
        tree = ast.parse(Path(p).read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                organ_names.add(node.func.attr)

    seen_ufunc: set = set()
    kept_rows: List[np.ndarray] = []
    with np.errstate(all="ignore"):
        for name in sorted(dir(np)):
            obj = getattr(np, name, None)
            if not isinstance(obj, np.ufunc) or obj.__name__ in seen_ufunc:
                continue
            seen_ufunc.add(obj.__name__)
            if obj.nin == 1 and obj.nout == 1:
                try:
                    out = obj(probe)
                except Exception:
                    continue
                if not isinstance(out, np.ndarray) or out.shape != probe.shape or out.dtype.kind not in "fiu":
                    continue
                if not np.all(np.isfinite(out)) or len(np.unique(np.round(out, 9))) < 0.9 * out.size:
                    continue
                row = np.asarray(obj(sorted_row), dtype=np.float64)
                if any(abs(np.corrcoef(row, kr)[0, 1]) > 0.99999 for kr in kept_rows):
                    continue  # linear duplicate of an already discovered pointwise operator
                kept_rows.append(row)
                lib.ops[obj.__name__] = Op(obj.__name__, "A", "A", lambda a, u=obj: u(a), "numpy_ufunc_registry")
                lib.pointwise.append(obj.__name__)
            elif obj.nin == 2 and obj.nout == 1:
                rname = f"{obj.__name__}.reduce"
                fn = _reduce_fn(obj)
                try:
                    out = np.asarray(fn(probe))
                except Exception:
                    out = None
                if out is not None and out.shape == (4,) and out.dtype.kind in "fiu" and np.all(np.isfinite(out)):
                    lib.ops[rname] = Op(rname, "A", "S", fn, "numpy_ufunc_registry")
                    lib.reducers.append(rname)
                # qualifies as a relation operator if g(a, a) is constant and g(a, c) is injective in a
                a = np.random.default_rng(1).uniform(1.0, 255.0, 24)
                ai = np.round(a)
                for dom, av in (("f", a), ("i", ai)):
                    try:
                        same = _binary(obj, av, av)
                        probe_c = _binary(obj, av, np.full_like(av, av[0]))
                    except Exception:
                        continue
                    if np.all(np.isfinite(same)) and np.ptp(same) < 1e-9 and len(np.unique(np.round(probe_c, 9))) >= 0.95 * av.size:
                        lib.relation_ops[obj.__name__] = (lambda x, y, u=obj: _binary(u, x, y))
                        break

        # primitives that ELYSIA's own organ source actually uses (typed empirically)
        for nm in sorted(organ_names):
            fn0 = getattr(np, nm, None)
            if fn0 is None:
                fn0 = getattr(np.fft, nm, None)
            if fn0 is None or isinstance(fn0, np.ufunc) or not callable(fn0) or nm in lib.ops:
                continue

            def wrapper(a: np.ndarray, f: Callable = fn0) -> Any:
                o = f(a, axis=-1)
                return np.abs(o) if np.iscomplexobj(o) else o

            try:
                out = np.asarray(wrapper(probe))
            except Exception:
                continue
            if out.dtype.kind not in "fiu" or not np.all(np.isfinite(out)) or out.shape[:1] != (4,):
                continue
            if out.ndim == 2:
                lib.ops[nm] = Op(nm, "A", "A", wrapper, "own_organ_source")
                lib.transforms.append(nm)
                lib.organ_names_found.append(nm)
            elif out.ndim == 1:
                lib.ops[nm] = Op(nm, "A", "S", wrapper, "own_organ_source")
                lib.reducers.append(nm)
                lib.organ_names_found.append(nm)

    # minimal body plan: part / whole / order
    for k in (1, 2, 3, 4):
        lib.ops[f"tail{k}"] = Op(f"tail{k}", "A", "A", lambda a, k=k: a[..., k:], "body_plan")
    lib.ops["init"] = Op("init", "A", "A", lambda a: a[..., :-1], "body_plan")
    lib.ops["last"] = Op("last", "A", "S", lambda a: a[..., -1], "body_plan")
    lib.ops["first"] = Op("first", "A", "S", lambda a: a[..., 0], "body_plan")
    lib.reducers.extend(["last", "first"])

    def _split(k: int) -> Callable[[Any], Any]:
        def fn(d: Any, k: int = k) -> Any:
            if not isinstance(d, np.ndarray):
                raise ValueError("split needs regular rows")
            return np.array_split(d, k, axis=-1)

        return fn

    for k in (2, 4):
        lib.ops[f"split{k}"] = Op(f"split{k}", "A", "F", _split(k), "body_plan")
    return lib


# ----------------------------------------------------------------------------
# 2. Layered program space
# ----------------------------------------------------------------------------
@dataclass
class Col:
    program: Program
    data: np.ndarray


_SELECTS: List[Tuple[str, ...]] = [
    (), ("tail1",), ("tail2",), ("tail3",), ("tail4",), ("init",),
    ("tail1", "init"), ("tail2", "init"), ("tail3", "init"), ("tail4", "init"),
    ("split2",), ("split4",),
]


class ProgramSpace:
    def __init__(self, lib: OperatorLibrary, modulus: Optional[int] = None) -> None:
        self.lib = lib
        self.modulus = modulus

    def _apply(self, name: str, data: Any, kind: str) -> Tuple[Any, str]:
        op = self.lib.ops[name]
        if op.out_kind == "F":
            if kind != "A" or not isinstance(data, np.ndarray):
                raise ValueError("partition unavailable")
            return op.fn(data), "F"
        if kind == "F":
            return [_rows(op.fn, p) for p in data], "F"
        return _rows(op.fn, data), "A"

    def _reduce(self, name: str, data: Any, kind: str) -> np.ndarray:
        op = self.lib.ops[name]
        if kind == "A":
            return np.asarray(_rows(op.fn, data), dtype=np.float64)
        return np.stack([np.asarray(_rows(op.fn, p), dtype=np.float64) for p in data], axis=1)

    def _chain(self, names: Sequence[str], data: Any, kind: str) -> Tuple[Any, str]:
        for n in names:
            data, kind = self._apply(n, data, kind)
        return data, kind

    def evaluate(self, prog: Program, x: Any) -> np.ndarray:
        if prog.rel is not None:
            g, p1, p2 = prog.rel
            return _binary(self.lib.relation_ops[g], self.evaluate(p1, x), self.evaluate(p2, x))
        data, kind = self._chain(prog.names, x, "A")
        red = self._reduce(prog.reducer, data, kind)
        if red.ndim == 2:
            red = self._aggregate(red, prog.agg or "col0")
        if prog.post == "mod_M":
            red = np.mod(red, self.modulus)
        return red

    @staticmethod
    def _aggregate(v: np.ndarray, agg: str) -> np.ndarray:
        if agg.startswith("col"):
            return v[:, int(agg[3:])]
        if agg == "slope":
            return v[:, -1] - v[:, 0]
        return np.std(v, axis=1)

    def enumerate(self, x: Any) -> List[Col]:
        cols: List[Col] = []
        seen: set = set()

        def add(prog: Program, arr: np.ndarray) -> None:
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim != 1 or not np.all(np.isfinite(arr)) or np.ptp(arr) < 1e-9:
                return
            key = hashlib.blake2b(np.round(arr, 6).tobytes(), digest_size=10).digest()
            if key in seen:
                return
            seen.add(key)
            cols.append(Col(prog, arr))

        def emit(names: Tuple[str, ...], data: Any, kind: str) -> None:
            for rname in self.lib.reducers:
                try:
                    red = self._reduce(rname, data, kind)
                except Exception:
                    continue
                if red.ndim == 1:
                    add(Program(names, rname), red)
                    if self.modulus:
                        add(Program(names, rname, post="mod_M"), np.mod(red, self.modulus))
                else:
                    for j in range(red.shape[1]):
                        add(Program(names, rname, agg=f"col{j}"), red[:, j])
                    add(Program(names, rname, agg="slope"), self._aggregate(red, "slope"))
                    add(Program(names, rname, agg="spread"), self._aggregate(red, "spread"))

        with np.errstate(all="ignore"):
            for sel in _SELECTS:
                try:
                    d1, k1 = self._chain(sel, x, "A")
                except Exception:
                    continue
                for tr in [None] + self.lib.transforms:
                    try:
                        d2, k2 = (d1, k1) if tr is None else self._apply(tr, d1, k1)
                    except Exception:
                        continue
                    if not _all_finite(d2):
                        continue
                    parts: List[Optional[str]] = [None]
                    if k2 == "A":
                        parts += ["split2", "split4"]
                    for pt in parts:
                        try:
                            d3, k3 = (d2, k2) if pt is None else self._apply(pt, d2, k2)
                        except Exception:
                            continue
                        base_names = sel + ((tr,) if tr else ()) + ((pt,) if pt else ())
                        for pw in [None] + self.lib.pointwise:
                            try:
                                d4, k4 = (d3, k3) if pw is None else self._apply(pw, d3, k3)
                            except Exception:
                                continue
                            if not _all_finite(d4):
                                continue
                            emit(base_names + ((pw,) if pw else ()), d4, k4)
        return cols

    def relation_stage(self, cols: List[Col], y: np.ndarray, max_new: int = 3000, group_cap: int = 12) -> List[Col]:
        """Find pairs of quantities that agree (under g) exactly on one class -> an invariant of that class."""
        new: List[Col] = []
        seen_keys: set = set()
        for gname, g in self.lib.relation_ops.items():
            for c in sorted(set(int(v) for v in y)):
                idx = np.where(y == c)[0]
                if len(idx) < 4:
                    continue
                groups: Dict[bytes, List[int]] = {}
                for ci, col in enumerate(cols):
                    v = col.data[idx]
                    try:
                        key = np.round(g(v, np.full_like(v, v[0])), 6)
                    except Exception:
                        continue
                    if not np.all(np.isfinite(key)):
                        continue
                    groups.setdefault(key.tobytes(), []).append(ci)
                for members in groups.values():
                    members = members[:group_cap]
                    for i, j in combinations(range(len(members)), 2):
                        a, b = cols[members[i]], cols[members[j]]
                        try:
                            r = g(a.data, b.data)
                        except Exception:
                            continue
                        if not np.all(np.isfinite(r)):
                            continue
                        dk = hashlib.blake2b(np.round(r, 6).tobytes(), digest_size=10).digest()
                        if dk in seen_keys:
                            continue
                        seen_keys.add(dk)
                        new.append(Col(Program(rel=(gname, a.program, b.program)), np.asarray(r, dtype=np.float64)))
                        if len(new) >= max_new:
                            return new
        return new


# ----------------------------------------------------------------------------
# 3. Pair rules, blind-spot map, residual cover
# ----------------------------------------------------------------------------
Rule = Tuple[str, float, str]  # ('thr', t, 'le'|'gt') or ('eq', m, 'a'|'b')


def _fit_pair_rule(a: np.ndarray, b: np.ndarray) -> Tuple[float, Optional[Rule]]:
    best_acc, best_rule = 0.5, None
    both = np.concatenate([a, b])
    uv = np.unique(both)
    if len(uv) > 1:
        ts = (uv[:-1] + uv[1:]) / 2.0
        if len(ts) > 48:
            ts = np.quantile(both, np.linspace(0.02, 0.98, 48))
        a_s, b_s = np.sort(a), np.sort(b)
        fa = np.searchsorted(a_s, ts, side="right") / len(a)
        fb = np.searchsorted(b_s, ts, side="right") / len(b)
        acc_le = 0.5 * (fa + (1.0 - fb))
        i = int(np.argmax(np.maximum(acc_le, 1.0 - acc_le)))
        if acc_le[i] >= 1.0 - acc_le[i]:
            best_acc, best_rule = float(acc_le[i]), ("thr", float(ts[i]), "le")
        else:
            best_acc, best_rule = float(1.0 - acc_le[i]), ("thr", float(ts[i]), "gt")
    for x_eq, other, tag in ((a, b, "a"), (b, a, "b")):
        vals, counts = np.unique(np.round(x_eq, 6), return_counts=True)
        m = float(vals[int(np.argmax(counts))])
        acc = 0.5 * (np.mean(np.isclose(x_eq, m, atol=1e-6)) + np.mean(~np.isclose(other, m, atol=1e-6)))
        if acc > best_acc + 1e-9:
            best_acc, best_rule = float(acc), ("eq", m, tag)
    return best_acc, best_rule


def _apply_rule(rule: Rule, v: np.ndarray) -> np.ndarray:
    kind, p, tag = rule
    if kind == "thr":
        return v <= p if tag == "le" else v > p
    eq = np.isclose(v, p, atol=1e-6)
    return eq if tag == "a" else ~eq


@dataclass
class SelectedFeature:
    program: Program
    pairs: List[Tuple[int, int]]
    rules: Dict[Tuple[int, int], Rule]
    train_acc: Dict[Tuple[int, int], float]
    val_acc: Dict[Tuple[int, int], float]


@dataclass
class DiscoveryReport:
    domain: str
    n_columns: int
    n_relation_columns: int
    flat_solved_pairs: List[Tuple[int, int]]
    blind_pairs: List[Tuple[int, int]]
    selected: List[SelectedFeature]
    unsolved_pairs: List[Tuple[int, int]]
    test_accuracy: float
    library_size_before: int
    library_size_after: int
    seconds: float


def _vote_predict(space: ProgramSpace, selected: Sequence[SelectedFeature], x: Any, classes: Sequence[int]) -> np.ndarray:
    n = len(x)
    votes = {c: np.zeros(n) for c in classes}
    for feat in selected:
        v = space.evaluate(feat.program, x)
        for (a, b), rule in feat.rules.items():
            pred_a = _apply_rule(rule, v)
            votes[a] += pred_a.astype(float)
            votes[b] += (~pred_a).astype(float)
    mat = np.stack([votes[c] for c in classes], axis=1)
    return np.asarray(classes)[np.argmax(mat, axis=1)]


def _stack_items(items: Any) -> Any:
    return items


def discover_by_blind_spots(
    domain: str,
    x_train: Any,
    y_train: Sequence[int],
    x_val: Any,
    y_val: Sequence[int],
    x_test: Any,
    y_test: Sequence[int],
    spine: EpistemicKnowledgeSpine,
    lib: OperatorLibrary,
    modulus: Optional[int] = None,
    use_relations: bool = False,
    acc_bar: float = 0.95,
) -> DiscoveryReport:
    t0 = time.time()
    size_before = lib.size()
    ytr = np.asarray(y_train, dtype=np.int64)
    yva = np.asarray(y_val, dtype=np.int64)
    yte = np.asarray(y_test, dtype=np.int64)

    if isinstance(x_train, np.ndarray):
        x_all: Any = np.concatenate([x_train, x_val], axis=0)
    else:
        x_all = list(x_train) + list(x_val)
    y_all = np.concatenate([ytr, yva])
    tr_idx = np.arange(len(ytr))
    va_idx = np.arange(len(ytr), len(y_all))

    space = ProgramSpace(lib, modulus=modulus)
    cols = space.enumerate(x_all)
    n_base = len(cols)
    rel_cols: List[Col] = []
    if use_relations:
        rel_cols = space.relation_stage(cols, y_all)
        cols = cols + rel_cols

    classes = sorted(set(int(v) for v in y_all))
    pairs = list(combinations(classes, 2))

    # per-column solved pairs (train rule must hold on the separate validation condition)
    solved: List[Dict[Tuple[int, int], Tuple[float, float, Rule]]] = []
    for col in cols:
        d: Dict[Tuple[int, int], Tuple[float, float, Rule]] = {}
        v = col.data
        for (a, b) in pairs:
            ta = v[tr_idx][ytr == a]
            tb = v[tr_idx][ytr == b]
            if len(ta) < 2 or len(tb) < 2:
                continue
            acc_t, rule = _fit_pair_rule(ta, tb)
            if rule is None or acc_t < acc_bar:
                continue
            va_a = v[va_idx][yva == a]
            va_b = v[va_idx][yva == b]
            pa, pb = _apply_rule(rule, va_a), _apply_rule(rule, va_b)
            acc_v = 0.5 * (float(np.mean(pa)) + float(np.mean(~pb)))
            if acc_v >= acc_bar:
                d[(a, b)] = (acc_t, acc_v, rule)
        solved.append(d)

    flat_solved: set = set()
    for col, d in zip(cols[:n_base], solved[:n_base]):
        if not col.program.uses_partition():
            flat_solved.update(d.keys())
    blind = [p for p in pairs if p not in flat_solved]
    for (a, b) in blind:
        spine.register_void(
            kind="VOID_BLIND_PAIR",
            target=f"{domain}:{a}|{b}",
            priority=0.9,
            origin_medium=domain,
            hypothesis=f"Observation logic without structural layers cannot tell class {a} from {b}.",
            suggested_medium="program_space_exploration",
        )

    unsolved = set(pairs)
    selected: List[SelectedFeature] = []
    while unsolved:
        best_i, best_key = -1, None
        for i, d in enumerate(solved):
            gain = [p for p in d if p in unsolved]
            if not gain:
                continue
            key = (-len(gain), cols[i].program.size, -float(np.mean([min(d[p][0], d[p][1]) for p in gain])), i)
            if best_key is None or key < best_key:
                best_key, best_i = key, i
        if best_i < 0:
            break
        d = solved[best_i]
        gain = [p for p in d if p in unsolved]
        selected.append(
            SelectedFeature(
                program=cols[best_i].program,
                pairs=gain,
                rules={p: d[p][2] for p in gain},
                train_acc={p: d[p][0] for p in gain},
                val_acc={p: d[p][1] for p in gain},
            )
        )
        unsolved -= set(gain)

    for feat in selected:
        lib.learned_programs.append(feat.program)
    for (a, b) in blind:
        if (a, b) not in unsolved:
            owner = next(f for f in selected if (a, b) in f.pairs)
            spine.resolve_voids_by_target("VOID_BLIND_PAIR", f"{domain}:{a}|{b}", owner.program.expr())

    pred = _vote_predict(space, selected, x_test, classes) if selected else np.full(len(yte), classes[0])
    acc = float(np.mean(pred == yte))

    spine.register_synthesized_organ(
        SynthesizedOrganRecord(
            organ_id=f"discovered_{domain}",
            domain="program_space",
            void_resolved=",".join(f"{domain}:{a}|{b}" for a, b in blind),
            source_code="\n".join(f"# pairs {f.pairs}: {f.program.expr()}" for f in selected),
            entry_functions=[],
            ast_node_count=sum(f.program.size for f in selected),
            validation_score=acc,
            falsified_candidates=[],
            discovered_principles={
                "flat_solved_pairs": [list(p) for p in sorted(flat_solved)],
                "blind_pairs": [list(p) for p in blind],
                "programs": [f.program.expr() for f in selected],
                "test_accuracy": acc,
            },
        )
    )
    return DiscoveryReport(
        domain=domain,
        n_columns=len(cols),
        n_relation_columns=len(rel_cols),
        flat_solved_pairs=sorted(flat_solved),
        blind_pairs=blind,
        selected=selected,
        unsolved_pairs=sorted(unsolved),
        test_accuracy=acc,
        library_size_before=size_before,
        library_size_after=lib.size(),
        seconds=time.time() - t0,
    )
