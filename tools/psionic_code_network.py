"""
Psionic Code Network PoC
------------------------

Hyper-Quaternion이 중심이 되는 Khala-style 코드 네트워크 런타임의 최소 구현.
주요 아이디어:
- 함수/모듈을 PsionicEntity(HyperQubit)로 보고, docstring 태그로 위상(스케일 w)과 축 방향(x/y/z)을 설정한다.
- AST로 호출 그래프를 추출해 공명 링크로 해석한다.
- delta_synchronization_factor로 집단 w를 동기화해 위상 공명을 흉내 낸다.

사용법:
    python tools/psionic_code_network.py path/to/file.py
    # 파일을 주지 않으면 내장 샘플 코드로 동작
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Add project root for relative imports when run as a script
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Project_Elysia.core.hyper_qubit import HyperQubit


# --- Tag parsing helpers -----------------------------------------------------

SCALE_W = {
    "point": 0.2,
    "line": 1.0,
    "plane": 2.0,
    "hyper": 3.0,
}

INTENT_AXIS = {
    "internal": (1.0, 0.0, 0.0),  # x
    "external": (0.0, 1.0, 0.0),  # y
    "law": (0.0, 0.0, 1.0),       # z
}


def parse_tags(doc: Optional[str]) -> Tuple[float, Tuple[float, float, float]]:
    """
    docstring에서 scale:intent 태그를 읽어 orientation 정보를 만든다.
    예시:
        \"\"\"do something
        scale: plane
        intent: internal, law
        \"\"\"
    """
    if not doc:
        return SCALE_W["line"], (0.0, 0.0, 1.0)  # 기본: line + law 축

    w = SCALE_W["line"]
    axes: List[Tuple[float, float, float]] = []

    for raw_line in doc.splitlines():
        line = raw_line.strip().lower()
        if line.startswith("scale:"):
            key = line.split(":", 1)[1].strip()
            w = SCALE_W.get(key, w)
        elif line.startswith("intent:"):
            intents = [p.strip() for p in line.split(":", 1)[1].split(",")]
            for intent in intents:
                axis = INTENT_AXIS.get(intent)
                if axis:
                    axes.append(axis)

    if not axes:
        axes = [INTENT_AXIS["law"]]

    # 평균으로 방향을 만든다.
    x = sum(a[0] for a in axes) / len(axes)
    y = sum(a[1] for a in axes) / len(axes)
    z = sum(a[2] for a in axes) / len(axes)
    return w, (x, y, z)


def amplitude_from_w(w: float) -> Tuple[float, float, float, float]:
    """
    스케일 w에 따라 기본 진폭(α,β,γ,δ)을 배치한다.
    """
    if w < 0.5:
        return 0.9, 0.05, 0.03, 0.02
    if w < 1.5:
        return 0.2, 0.7, 0.08, 0.02
    if w < 2.5:
        return 0.1, 0.2, 0.6, 0.1
    return 0.05, 0.1, 0.15, 0.7


# --- Graph model -------------------------------------------------------------

@dataclass
class PsionicNode:
    name: str
    qubit: HyperQubit
    doc_tags: Dict[str, str] = field(default_factory=dict)
    calls: Set[str] = field(default_factory=set)
    called_by: Set[str] = field(default_factory=set)


class CallGraphVisitor(ast.NodeVisitor):
    def __init__(self):
        self.current_fn: Optional[str] = None
        self.calls: Dict[str, Set[str]] = {}
        self.docstrings: Dict[str, str] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev = self.current_fn
        self.current_fn = node.name
        self.docstrings[node.name] = ast.get_docstring(node) or ""
        self.generic_visit(node)
        self.current_fn = prev

    def visit_Call(self, node: ast.Call):
        if self.current_fn:
            callee = self._callee_name(node.func)
            if callee:
                self.calls.setdefault(self.current_fn, set()).add(callee)
        self.generic_visit(node)

    @staticmethod
    def _callee_name(func: ast.AST) -> Optional[str]:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None


def load_tag_overrides(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """
    JSON 형식으로 함수명 → {scale,intent} 매핑을 읽는다.
    예시:
    {
      "core_loop": {"scale": "line", "intent": "law, external"},
      "transform": {"scale": "plane", "intent": "internal, law"}
    }
    """
    if not path:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: {kk.lower(): vv for kk, vv in v.items()} for k, v in data.items()}


def parse_with_override(fn_name: str, doc: Optional[str], overrides: Dict[str, Dict[str, str]]):
    """
    태그 우선순위: override JSON > docstring > 기본값
    """
    if fn_name in overrides:
        o = overrides[fn_name]
        w = SCALE_W.get(o.get("scale", "").strip(), SCALE_W["line"])
        intents = [p.strip() for p in o.get("intent", "").split(",") if p.strip()]
        axes = [INTENT_AXIS[i] for i in intents if i in INTENT_AXIS]
        if not axes:
            axes = [INTENT_AXIS["law"]]
        x = sum(a[0] for a in axes) / len(axes)
        y = sum(a[1] for a in axes) / len(axes)
        z = sum(a[2] for a in axes) / len(axes)
        return w, (x, y, z), True
    w, (x, y, z) = parse_tags(doc)
    return w, (x, y, z), False


def build_psionic_graph(source: str, overrides: Dict[str, Dict[str, str]]) -> Dict[str, PsionicNode]:
    tree = ast.parse(source)
    visitor = CallGraphVisitor()
    visitor.visit(tree)

    nodes: Dict[str, PsionicNode] = {}

    # 함수 노드 구성
    for fn_name, doc in visitor.docstrings.items():
        w, (x, y, z), overridden = parse_with_override(fn_name, doc, overrides)
        alpha, beta, gamma, delta = amplitude_from_w(w)

        hq = HyperQubit(fn_name, value=f"{fn_name}()")
        hq.state.alpha = alpha
        hq.state.beta = beta
        hq.state.gamma = gamma
        hq.state.delta = delta
        hq.state.w = w
        hq.state.x = x
        hq.state.y = y
        hq.state.z = z
        hq.state.normalize()

        nodes[fn_name] = PsionicNode(
            name=fn_name,
            qubit=hq,
            doc_tags={
                "scale": str(w),
                "intent_vector": f"{(x, y, z)}",
                "source": "override" if overridden else "docstring",
            },
        )

    # 호출 관계 설정
    for caller, callees in visitor.calls.items():
        for callee in callees:
            if caller in nodes:
                nodes[caller].calls.add(callee)
            if callee in nodes:
                nodes[callee].called_by.add(caller)

    return nodes


# --- Synchronization ---------------------------------------------------------

def synchronize(nodes: Dict[str, PsionicNode], delta_factor: float = 0.2) -> None:
    """
    집단 w를 동기화해 Δ=1에 가까운 상태를 흉내 낸다.
    """
    if not nodes:
        return
    avg_w = sum(n.qubit.state.w for n in nodes.values()) / len(nodes)
    for node in nodes.values():
        old_w = node.qubit.state.w
        node.qubit.state.w = (1 - delta_factor) * old_w + delta_factor * avg_w
        alpha, beta, gamma, delta = amplitude_from_w(node.qubit.state.w)
        node.qubit.state.alpha = alpha
        node.qubit.state.beta = beta
        node.qubit.state.gamma = gamma
        node.qubit.state.delta = delta
        node.qubit.state.normalize()


# --- Reporting ---------------------------------------------------------------

def summarize(nodes: Dict[str, PsionicNode]) -> str:
    lines: List[str] = []
    lines.append(f"노드 수: {len(nodes)}")
    avg_w = sum(n.qubit.state.w for n in nodes.values()) / len(nodes) if nodes else 0.0
    lines.append(f"평균 w: {avg_w:.2f}")

    lines.append("\n[노드 상태]")
    for node in nodes.values():
        probs = node.qubit.state.probabilities()
        lines.append(
            f"- {node.name}: w={node.qubit.state.w:.2f}, "
            f"P/L/S/G=({probs['Point']:.2f},{probs['Line']:.2f},"
            f"{probs['Space']:.2f},{probs['God']:.2f}), "
            f"calls={sorted(node.calls)}, source={node.doc_tags.get('source')}"
        )

    lines.append("\n[공명 링크]")
    for node in nodes.values():
        for callee in sorted(node.calls):
            lines.append(f"{node.name} -> {callee}")

    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------

# --- DOT helpers -------------------------------------------------------------


def _w_to_color(w: float) -> str:
    """
    Simple blue->purple->gold gradient for DOT rendering.
    """
    w = max(0.0, min(3.0, w))
    t = w / 3.0
    if t < 0.5:
        a, b = (80, 120, 255), (200, 120, 220)
        tt = t / 0.5
    else:
        a, b = (200, 120, 220), (235, 200, 80)
        tt = (t - 0.5) / 0.5
    c = tuple(int(a[i] + (b[i] - a[i]) * tt) for i in range(3))
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def write_dot(nodes: Dict[str, PsionicNode], path: Path, title: str) -> None:
    """
    Emit a DOT file for quick Graphviz rendering.
    """
    lines: List[str] = ["digraph Psionic {", f'label="{title}"; labelloc=t; fontsize=20;']
    for node in nodes.values():
        probs = node.qubit.state.probabilities()
        label = (
            f"{node.name}\\nw={node.qubit.state.w:.2f}"
            f"\\nP/L/S/G={probs['Point']:.2f}/{probs['Line']:.2f}/"
            f"{probs['Space']:.2f}/{probs['God']:.2f}"
        )
        color = _w_to_color(node.qubit.state.w)
        lines.append(
            f'"{node.name}" [label="{label}", style=filled, fillcolor="{color}", fontcolor="black"];'
        )
    for node in nodes.values():
        for callee in node.calls:
            if callee in nodes:
                lines.append(f'"{node.name}" -> "{callee}";')
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")

SAMPLE_CODE = """
def core_loop():
    \"\"\"scale: line
    intent: law, external
    \"\"\"
    fetch_data()
    transform()
    write_out()


def fetch_data():
    \"\"\"scale: point
    intent: external
    \"\"\"
    return 42


def transform():
    \"\"\"scale: plane
    intent: internal, law
    \"\"\"
    return "ok"


def write_out():
    \"\"\"scale: hyper
    intent: external
    \"\"\"
    return True
"""


def main():
    parser = argparse.ArgumentParser(description="Psionic Code Network PoC (Hyper-Quaternion 기반)")
    parser.add_argument("paths", nargs="*", help="분석할 Python 파일 경로")
    parser.add_argument("--delta", type=float, default=0.2, help="동기화 강도 (0~1)")
    parser.add_argument(
        "--delta-sweep",
        help="쉼표로 구분된 Δ 값 리스트(예: 0,0.5,1.0). 지정 시 각 Δ에 대한 요약을 출력.",
    )
    parser.add_argument(
        "--tag-file",
        type=Path,
        help="JSON 파일 경로. 함수명→{scale,intent} 매핑으로 docstring 대신 태그를 덮어씀.",
    )
    parser.add_argument(
        "--dot-out",
        type=Path,
        help="DOT 파일 출력 경로(확장자 .dot 권장). Δ 스윕 시 delta 값이 접미사로 붙음.",
    )
    args = parser.parse_args()

    if not args.paths:
        source = SAMPLE_CODE
        label = "[내장 샘플]"
    else:
        path_list = [Path(p) for p in args.paths]
        source = "\n\n".join(p.read_text(encoding="utf-8") for p in path_list)
        label = ", ".join(str(p) for p in path_list)

    overrides = load_tag_overrides(args.tag_file)

    nodes = build_psionic_graph(source, overrides)

    def run_and_print(delta_values: Iterable[float]):
        sweep_list = list(delta_values)
        for d in sweep_list:
            # deepcopy 없이 반복 적용하면 누적되므로 복사
            import copy

            nodes_copy = copy.deepcopy(nodes)
            print(f"\n=== Psionic Graph for {label} | Δ={d} 전 ===")
            print(summarize(nodes_copy))
            synchronize(nodes_copy, delta_factor=d)
            print(f"\n=== Δ={d} 동기화 후 ===")
            print(summarize(nodes_copy))
            if args.dot_out:
                suffix = f"_delta{d}".replace(".", "_")
                out_path = args.dot_out
                if len(sweep_list) > 1:
                    out_path = out_path.with_name(out_path.stem + suffix + out_path.suffix)
                write_dot(nodes_copy, out_path, f"{label} Δ={d}")

    if args.delta_sweep:
        sweep = [float(x) for x in args.delta_sweep.split(",") if x.strip() != ""]
        run_and_print(sweep)
    else:
        run_and_print([args.delta])


if __name__ == "__main__":
    main()
