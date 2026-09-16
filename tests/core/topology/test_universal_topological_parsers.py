"""
Unit Tests: Universal Invariant Topological Parsers & Cross-Medium Isomorphism
==============================================================================
Python AST, SymPy, C++, Binary, CAD 파서의 위상 추출 및 동형성 대조 단위 테스트.
"""

import pytest
import sympy as sp

from core.topology.causal_stem_branch_engine import (
    CausalStemBranchTopologyEngine,
    CausalGraph,
    NodeType
)
from core.topology.topological_ast_parser import TopologicalASTParser
from core.topology.sympy_topological_parser import SymPyTopologicalParser
from core.topology.cpp_topological_parser import CppTopologicalParser
from core.topology.binary_topological_parser import BinaryTopologicalParser
from core.topology.cad_kinematic_parser import CADKinematicParser
from core.topology.base_topological_parser import (
    INVARIANT_OP_ADD,
    INVARIANT_OP_MULTIPLY,
    INVARIANT_RIGID_BODY,
    INVARIANT_BYTE_VAL
)


def test_topological_ast_parser():
    parser = TopologicalASTParser()
    code = "x = 5\ny = x + 10\nreturn y"
    graph = parser.parse_code(code)

    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    # Add 연산 노드 존재 확인
    add_nodes = [n for n in graph.nodes.values() if n.invariant_signature == INVARIANT_OP_ADD]
    assert len(add_nodes) == 1
    # 컨텍스트 격리 확인
    node_id = add_nodes[0].node_id
    assert any("py_syntax" in c for c in graph.context_constraints[node_id])


def test_sympy_topological_parser():
    parser = SymPyTopologicalParser()
    x, y = sp.symbols('x y')
    expr = sp.Add(x, sp.Mul(y, 3))
    graph = parser.parse_expr(expr)

    assert len(graph.nodes) > 0
    add_nodes = [n for n in graph.nodes.values() if n.invariant_signature == INVARIANT_OP_ADD]
    mul_nodes = [n for n in graph.nodes.values() if n.invariant_signature == INVARIANT_OP_MULTIPLY]

    assert len(add_nodes) == 1
    assert len(mul_nodes) == 1


def test_cpp_topological_parser():
    parser = CppTopologicalParser()
    cpp_code = """
    int a = 10;
    int b = a * 2;
    return b;
    """
    graph = parser.parse_code(cpp_code)
    assert len(graph.nodes) == 3
    assert graph.nodes["N_CPP_2"].invariant_signature == INVARIANT_OP_MULTIPLY
    assert "cpp_type:binary_op" in graph.context_constraints["N_CPP_2"]


def test_binary_topological_parser():
    parser = BinaryTopologicalParser(unit_bytes=1)
    raw = bytes([0xDE, 0xAD, 0xBE, 0xEF])
    graph = parser.parse_bytes(raw, base_offset=0x8000)

    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3
    assert graph.nodes["N_BYTE_1"].invariant_signature == INVARIANT_BYTE_VAL
    assert "mem_offset:0x8000" in graph.context_constraints["N_BYTE_1"]


def test_cad_kinematic_parser():
    parser = CADKinematicParser()
    data = {
        "assembly_name": "robotic_joint",
        "parts": [
            {"id": "base_link", "mass": 5.0},
            {"id": "arm_link", "mass": 2.0}
        ],
        "mates": [
            {
                "parent_part_id": "base_link",
                "child_part_id": "arm_link",
                "type": "revolute",
                "range_limit": [-90, 90],
                "tolerance": "0.05mm"
            }
        ]
    }
    graph = parser.parse_assembly(data)
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].precondition == "DOF_TRANSFER:1_ROTATION"
    assert "limit_range:[-90,90]" in graph.context_constraints["N_PART_2"]


def test_cross_medium_stem_extraction():
    engine = CausalStemBranchTopologyEngine()
    py_parser = TopologicalASTParser()
    cpp_parser = CppTopologicalParser()

    py_code = "val = 10\nres = val * 2\nreturn res"
    cpp_tokens = [
        {"op_signature": "OP_STATE_BINDING:Constant", "syntax": "int val = 10;", "raw_type": "int"},
        {"op_signature": "INVARIANT_OP_MULTIPLY", "syntax": "int res = val * 2;", "raw_type": "binary_op"},
        {"op_signature": "OP_TERMINATE_EMIT:Name", "syntax": "return res;", "raw_type": "return_stmt"}
    ]

    graph_py = py_parser.parse_code(py_code)
    graph_cpp = cpp_parser.parse_code(cpp_tokens)

    analyzed_cpp, iso_map = engine.extract_stem_and_branch(graph_py, graph_cpp)

    assert len(iso_map) == 3
    assert analyzed_cpp.nodes["N_CPP_2"].node_type == NodeType.STEM
