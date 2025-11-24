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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


def build_psionic_graph(source: str) -> Dict[str, PsionicNode]:
    tree = ast.parse(source)
    visitor = CallGraphVisitor()
    visitor.visit(tree)

    nodes: Dict[str, PsionicNode] = {}

    # 함수 노드 구성
    for fn_name, doc in visitor.docstrings.items():
        w, (x, y, z) = parse_tags(doc)
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
            doc_tags={"scale": str(w), "intent_vector": f"{(x, y, z)}"},
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
            f"calls={sorted(node.calls)}"
        )

    lines.append("\n[공명 링크]")
    for node in nodes.values():
        for callee in sorted(node.calls):
            lines.append(f"{node.name} -> {callee}")

    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------

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
    args = parser.parse_args()

    if not args.paths:
        source = SAMPLE_CODE
        label = "[내장 샘플]"
    else:
        path_list = [Path(p) for p in args.paths]
        source = "\n\n".join(p.read_text(encoding="utf-8") for p in path_list)
        label = ", ".join(str(p) for p in path_list)

    nodes = build_psionic_graph(source)
    print(f"=== Psionic Graph for {label} ===")
    print(summarize(nodes))

    # 동기화 적용
    synchronize(nodes, delta_factor=args.delta)
    print("\n=== Δ 동기화 후 ===")
    print(summarize(nodes))


if __name__ == "__main__":
    main()
