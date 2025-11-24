"""
Psionic Trace Hook
------------------

간이 런타임 훅: 함수 호출을 추적해 Psionic Code Network 노드의 링크/호출 카운트를 실시간으로 기록.
* sys.settrace를 사용하므로 성능에 민감한 환경에서는 주의.
* 태그 입력은 psionic_code_network.py의 tag-file JSON을 재사용할 수 있도록 의도.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from Project_Elysia.core.hyper_qubit import HyperQubit
from tools.psionic_code_network import (
    CallGraphVisitor,
    PsionicNode,
    SCALE_W,
    INTENT_AXIS,
    parse_tags,
    load_tag_overrides,
    parse_with_override,
    amplitude_from_w,
)
import ast


class PsionicTrace:
    def __init__(self, tag_overrides: Optional[Dict[str, Dict[str, str]]] = None):
        self.tag_overrides = tag_overrides or {}
        self.nodes: Dict[str, PsionicNode] = {}
        self.calls: Set[Tuple[str, str]] = set()

    def _get_or_create(self, fn_name: str, doc: Optional[str]) -> PsionicNode:
        if fn_name in self.nodes:
            return self.nodes[fn_name]
        w, (x, y, z), overridden = parse_with_override(fn_name, doc, self.tag_overrides)
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
        node = PsionicNode(
            name=fn_name,
            qubit=hq,
            doc_tags={
                "scale": str(w),
                "intent_vector": f"{(x, y, z)}",
                "source": "override" if overridden else "runtime",
            },
        )
        self.nodes[fn_name] = node
        return node

    def tracer(self, frame, event, arg):
        if event != "call":
            return self.tracer
        co = frame.f_code
        fn_name = co.co_name
        if fn_name.startswith("<"):
            return self.tracer
        doc = frame.f_globals.get("__doc__", "") or ""
        caller = frame.f_back.f_code.co_name if frame.f_back else None
        callee_node = self._get_or_create(fn_name, doc)
        if caller and not caller.startswith("<"):
            caller_doc = frame.f_back.f_globals.get("__doc__", "") or ""
            caller_node = self._get_or_create(caller, caller_doc)
            caller_node.calls.add(fn_name)
            self.calls.add((caller, fn_name))
        return self.tracer

    def start(self):
        sys.settrace(self.tracer)

    def stop(self):
        sys.settrace(None)


def run_with_trace(script: Path, tag_file: Optional[Path] = None) -> PsionicTrace:
    overrides = load_tag_overrides(tag_file) if tag_file else {}
    tracer = PsionicTrace(overrides)
    tracer.start()
    try:
        # Execute the script in its own global namespace
        ns = {"__file__": str(script), "__name__": "__main__"}
        exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), ns, ns)
    finally:
        tracer.stop()
    return tracer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a Python script with Psionic trace hook.")
    parser.add_argument("script", type=Path, help="실행할 Python 스크립트")
    parser.add_argument("--tag-file", type=Path, help="태그 override JSON (옵션)")
    args = parser.parse_args()

    trace = run_with_trace(args.script, args.tag_file)
    print(f"콜렉션 완료: 노드 {len(trace.nodes)}, 링크 {len(trace.calls)}")
    for n in trace.nodes.values():
        print(f"- {n.name}: calls={sorted(n.calls)} source={n.doc_tags.get('source')}")
