"""
Elysia Universal Topological Parser Interface
==============================================
모든 이종 매질(Python AST, SymPy 수식, C/C++ 소스코드, 바이너리 메모리, CAD 조립체 등)을
표상 언어 고유의 구문 트리를 도려내고 단일 표준 인과 그래프 G = (V, E, C)로 변환하는 추상 기반 클래스.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Optional
from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType, TrajectoryContext
)

# =====================================================================
# Universal Invariant Signatures (범용 불변 인과 서명 표준 규격)
# =====================================================================

# 1. 산술 및 논리 연산 서명 (Arithmetic & Logic Invariants)
INVARIANT_OP_ADD = "INVARIANT_OP_ADD"
INVARIANT_OP_SUBTRACT = "INVARIANT_OP_SUBTRACT"
INVARIANT_OP_MULTIPLY = "INVARIANT_OP_MULTIPLY"
INVARIANT_OP_DIVIDE = "INVARIANT_OP_DIVIDE"
INVARIANT_OP_EQUALS = "INVARIANT_OP_EQUALS"
INVARIANT_OP_GREATER = "INVARIANT_OP_GREATER"
INVARIANT_OP_LESS = "INVARIANT_OP_LESS"
INVARIANT_OP_BRANCH = "INVARIANT_OP_BRANCH"

# 2. 상태 바인딩 및 전이 서명 (State & Binding Invariants)
INVARIANT_STATE_BINDING = "INVARIANT_STATE_BINDING"
INVARIANT_CONST_VALUE = "INVARIANT_CONST_VALUE"
INVARIANT_VAR_BINDING = "INVARIANT_VAR_BINDING"
INVARIANT_TERMINATE_EMIT = "INVARIANT_TERMINATE_EMIT"

# 3. 비트 및 메모리 물리 서명 (Byte & Memory Invariants)
INVARIANT_MEM_BLOCK = "INVARIANT_MEM_BLOCK"
INVARIANT_MEM_OFFSET = "INVARIANT_MEM_OFFSET"
INVARIANT_BYTE_VAL = "INVARIANT_BYTE_VAL"
INVARIANT_POINTER_DEREF = "INVARIANT_POINTER_DEREF"

# 4. 기구학적 구속 및 자유도 서명 (Kinematics & Assembly Invariants)
INVARIANT_RIGID_BODY = "INVARIANT_RIGID_BODY"
INVARIANT_JOINT_REVOLUTE = "INVARIANT_JOINT_REVOLUTE"     # 1-DOF 회전
INVARIANT_JOINT_PRISMATIC = "INVARIANT_JOINT_PRISMATIC"   # 1-DOF 슬라이더
INVARIANT_SURFACE_MATE = "INVARIANT_SURFACE_MATE"         # 평면 밀착 구속
INVARIANT_GEAR_COUPLING = "INVARIANT_GEAR_COUPLING"       # 기어 연동비 구속


class BaseTopologicalParser(ABC):
    """
    이종 매질을 인과 위상 그래프 G = (V, E, C)로 해부하는 추상 기반 파서.
    모든 하위 파서는 이 인터페이스를 구현하여 각 매질의 AST/CST/바이너리를 동일 규격으로 사상한다.
    """

    def __init__(self, medium_type: str = "abstract_medium"):
        self.medium_type = medium_type
        self._node_idx = 0

    def _generate_node_id(self, prefix: str = "N") -> str:
        self._node_idx += 1
        return f"{prefix}_{self._node_idx}"

    def reset_counter(self):
        self._node_idx = 0

    @abstractmethod
    def parse(self, source: Any) -> CausalGraph:
        """
        주어진 소스(코드 문자열, 수식 객체, 바이트 버퍼, CAD 모델)를
        인과 궤적 그래프 G = (V, E, C)로 변환하여 반환한다.
        """
        pass
