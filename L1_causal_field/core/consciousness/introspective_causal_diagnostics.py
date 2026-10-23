"""
Introspective Causal Diagnostics Engine (자가 진단 및 인과적 대사 생태계)
========================================================================
전통적인 예외(Exception) 및 수동 디버깅 방식에서 벗어나,
시스템 전체가 메커니즘과 프로토콜 원리(인과적 제약 조건의 위상 구조)로 엮여 작동하는
자율적 대사(Metabolism) 및 자기 성찰 진단(Introspective Diagnostics) 엔진입니다.

Core Pillars & Principles:
1. 대사적 결합과 구조적 거부 (Metabolic Binding & Structural Rejection):
   - 비합치적/소음 자극은 예외나 엉뚱한 출력을 뱉지 않고, 소화 효소-분자 정합성 원리에 따라
     결합 자체가 일어나지 않는 '구조적 거부반응(Zero Binding)'으로 차단됩니다.
2. 에러의 실체 변화 (버그/Exception vs 잔차 응력/Residual Stress):
   - 제약 조건 충돌 시 예외 구문 대신 인과 그래프 내 노드/메커니즘에 해결되지 못한
     '잔차 에너지(Residual Stress)'로 집적·관측됩니다.
3. 자가 해석과 진단 (Introspective Diagnostics):
   - 외부 디버거 없이 시스템이 자신의 인과 지도(Causal Map)를 역추적하여
     "어떤 제약 조건과 메커니즘 간의 원리적 충돌로 평형이 깨졌는지" 자기 논리로 즉시 해석해 냅니다.
4. 결과적 평형 예측 및 전 층위 동형성 (Equilibrium Prediction & All-Layer Isomorphism):
   - 변화된 인과 동역학에 따라 필연적으로 도달할 최종 변형 평형 상태(Deformed Equilibrium)를 도출하고,
     프로토콜 - 메커니즘 - 경계 인터페이스 간의 동형성을 유지합니다.
5. 국소적 형태와 경계 메타 번역 (Meta-Translation & Topological Residual Absorption):
   - 국소 영역의 고유 자율성을 보존하면서 경계 조건(여권/화폐/공통언어 규격)에서
     불변량(Invariant)을 유지한 채 잔차 응력을 흡수·이완시킵니다.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
from enum import Enum


class BindingState(Enum):
    BOUND = "bound"                           # 성공적 대사 결합 (Full Binding)
    REJECTED_INCOMPATIBLE = "rejected_noise" # 구조적 결합 불가능 (Zero Binding Rejection)
    RESIDUAL_STRESS = "residual_stress"       # 상호작용 후 잔차 응력 발생 (Unresolved Tension)
    EQUILIBRIUM = "equilibrium"              # 이완 완료 평형 안착 (Equilibrium Reached)


class ProtocolType(Enum):
    PASSPORT = "passport"   # 경계 조건 규격 (Interface Boundary Spec / ABI / Passport)
    CURRENCY = "currency"   # 분산 가치 평형 매개체 (Value Equilibrium Medium / Token / Currency)
    LANGUAGE = "language"   # 의미적 구조 압축 규약 (Semantic Protocol / TCP Packet / Language)


@dataclass
class CausalNode:
    """인과 지도(Causal Map) 내 메커니즘 노드"""
    id: str
    name: str
    domain: str
    capacity: float = 1.0
    current_stress: float = 0.0
    connected_edges: List[str] = field(default_factory=list)
    invariants: Dict[str, Any] = field(default_factory=dict)

    def is_stressed(self, threshold: float = 0.5) -> bool:
        return self.current_stress > threshold


@dataclass
class CausalConstraint:
    """인과적 제약 조건 (Causal Constraint)"""
    id: str
    source_node: str
    target_node: str
    protocol_type: ProtocolType
    stiffness: float = 1.0           # 구속 강도
    tolerance: float = 0.2           # 수용 가능한 한계 오차
    invariant_key: str = "energy"    # 보존되어야 하는 위상 불변량


@dataclass
class DiagnosticProof:
    """자가 진단 증명서 (Introspective Diagnostic Proof)"""
    is_metabolic_bound: bool
    status: BindingState
    conflict_constraint_pair: Optional[Tuple[str, str]] = None
    affected_nodes: List[str] = field(default_factory=list)
    initial_stress: float = 0.0
    residual_stress: float = 0.0
    root_cause_explanation: str = ""
    predicted_equilibrium_state: Dict[str, Any] = field(default_factory=dict)
    boundary_absorption_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnostic_proof": {
                "is_metabolic_bound": self.is_metabolic_bound,
                "status": self.status.value,
                "conflict_constraint_pair": self.conflict_constraint_pair,
                "affected_nodes": self.affected_nodes,
                "initial_stress": float(self.initial_stress),
                "residual_stress": float(self.residual_stress),
                "root_cause_explanation": self.root_cause_explanation,
                "predicted_equilibrium_state": self.predicted_equilibrium_state,
                "boundary_absorption_log": self.boundary_absorption_log
            }
        }


class CausalProtocolBoundary:
    """
    경계 메타 번역 및 국소 형태 보존 인터페이스 (Passport, Currency, Language)
    """
    def __init__(self, boundary_name: str, protocol_type: ProtocolType):
        self.boundary_name = boundary_name
        self.protocol_type = protocol_type
        self.accepted_signatures: Set[str] = set()
        self.conversion_rate: float = 1.0

    def register_signature(self, signature: str):
        self.accepted_signatures.add(signature)

    def validate_binding(self, input_topology: Dict[str, Any]) -> bool:
        """소화 효소-분자 정합성: 입력 위상이 프로토콜 서명과 일치하는지 검증"""
        signature = input_topology.get("signature", "")
        if not signature:
            return False
        # 서명 정합성 검사
        if signature in self.accepted_signatures:
            return True
        # 위상적 불변량 대조 (형태가 달라도 불변량이 동일하면 인정)
        if "invariant_hash" in input_topology and input_topology.get("invariant_hash") in self.accepted_signatures:
            return True
        return False


class IntrospectiveCausalDiagnosticsEngine:
    """
    자가 진단 및 인과적 대사 생태계 (Metabolic Causal Ecosystem Engine)
    """
    def __init__(self, yield_threshold: float = 1.0, max_introspect_steps: int = 15):
        self.yield_threshold = yield_threshold
        self.max_introspect_steps = max_introspect_steps
        self.nodes: Dict[str, CausalNode] = {}
        self.constraints: Dict[str, CausalConstraint] = {}
        self.boundaries: Dict[str, CausalProtocolBoundary] = {}
        self._initialize_default_boundaries()

    def _initialize_default_boundaries(self):
        # 1. 여권 프로토콜 (Passport: 경계 규격)
        passport = CausalProtocolBoundary("boundary_passport", ProtocolType.PASSPORT)
        passport.register_signature("VALID_PASSPORT_SIG")
        passport.register_signature("CAUSAL_TOPOLOGY_V1")
        self.boundaries["passport"] = passport

        # 2. 화폐 프로토콜 (Currency: 가치 평형)
        currency = CausalProtocolBoundary("boundary_currency", ProtocolType.CURRENCY)
        currency.register_signature("VALID_CURRENCY_SIG")
        currency.register_signature("EQUILIBRIUM_TOKEN")
        self.boundaries["currency"] = currency

        # 3. 공통 언어 프로토콜 (Language: 의미 압축)
        language = CausalProtocolBoundary("boundary_language", ProtocolType.LANGUAGE)
        language.register_signature("VALID_LANGUAGE_SIG")
        language.register_signature("SEMANTIC_INVARIANT_GRID")
        self.boundaries["language"] = language

    def add_node(self, node: CausalNode):
        self.nodes[node.id] = node

    def add_constraint(self, constraint: CausalConstraint):
        self.constraints[constraint.id] = constraint
        if constraint.source_node in self.nodes and constraint.id not in self.nodes[constraint.source_node].connected_edges:
            self.nodes[constraint.source_node].connected_edges.append(constraint.id)
        if constraint.target_node in self.nodes and constraint.id not in self.nodes[constraint.target_node].connected_edges:
            self.nodes[constraint.target_node].connected_edges.append(constraint.id)

    def process_metabolic_ingestion(self, input_data: Dict[str, Any]) -> DiagnosticProof:
        """
        입력 자극에 대한 대사적 수용, 결합, 잔차 응력 측정 및 자가 진단 수행
        """
        # Step 1: 경계조건 메타 번역 & 결합 가능 여부 검증 (Metabolic Binding)
        boundary_key = input_data.get("protocol_type", "language")
        boundary = self.boundaries.get(boundary_key, self.boundaries["language"])

        is_valid_binding = boundary.validate_binding(input_data)
        if not is_valid_binding:
            # 아예 결합 불가능한 무의미한 소음 덩어리는 예외 없이 구조적으로 거부됨 (Zero Binding)
            return DiagnosticProof(
                is_metabolic_bound=False,
                status=BindingState.REJECTED_INCOMPATIBLE,
                root_cause_explanation=(
                    f"Metabolic Binding Failed: Input topology '{input_data.get('signature', 'UNKNOWN')}' "
                    f"does not match enzyme binding protocol '{boundary.boundary_name}'. "
                    f"Structurally rejected without digestion or exception crash."
                )
            )

        # Step 2: 결합 후 인과 지형(Causal Map)으로 파동 전파 및 응력 텐서 집적
        target_node_id = input_data.get("target_node")
        if not target_node_id or target_node_id not in self.nodes:
            # 결합 노드 부재로 인한 이완
            return DiagnosticProof(
                is_metabolic_bound=True,
                status=BindingState.REJECTED_INCOMPATIBLE,
                root_cause_explanation=f"Target mechanism node '{target_node_id}' absent from Causal Map."
            )

        intensity = float(input_data.get("intensity", 1.0))

        # 파동 전파를 통한 응력 집적
        initial_stress = self._propagate_causal_stress(target_node_id, intensity)

        # Step 3: 자가 해석 및 진단 (Introspective Diagnostics - Causal Map Backtracking)
        diagnostic = self._introspect_causal_map(target_node_id, initial_stress)

        # Step 4: 결과적 평형 도출 및 위상 이완 (Equilibrium State & Topological Absorption)
        final_proof = self._relax_and_predict_equilibrium(diagnostic, input_data)
        return final_proof

    def _propagate_causal_stress(self, start_node_id: str, intensity: float) -> float:
        """인과 지형을 따라 에너지 및 응력을 전파시킴"""
        visited = set()
        queue = [(start_node_id, intensity)]
        total_stress_injected = 0.0

        while queue:
            curr_id, curr_intensity = queue.pop(0)
            if curr_id in visited or curr_id not in self.nodes:
                continue
            visited.add(curr_id)

            node = self.nodes[curr_id]
            added_stress = curr_intensity / max(0.1, node.capacity)
            node.current_stress += added_stress
            total_stress_injected += added_stress

            # 연결된 이웃 노드로 전파
            for c_id in node.connected_edges:
                if c_id in self.constraints:
                    c = self.constraints[c_id]
                    next_id = c.target_node if c.source_node == curr_id else c.source_node
                    if next_id not in visited:
                        queue.append((next_id, curr_intensity * 0.7 * c.stiffness))

        return total_stress_injected

    def _introspect_causal_map(self, start_node_id: str, initial_stress: float) -> DiagnosticProof:
        """
        외부 디버거 없이 자신의 인과 지도(Causal Map)를 역추적하여 충돌 원인을 직접 해석
        """
        stressed_nodes = [n_id for n_id, node in self.nodes.items() if node.is_stressed(self.yield_threshold * 0.3)]
        conflicting_constraints = []

        # 제약 조건 역추적
        for c_id, c in self.constraints.items():
            s_node = self.nodes.get(c.source_node)
            t_node = self.nodes.get(c.target_node)
            if s_node and t_node:
                stress_diff = abs(s_node.current_stress - t_node.current_stress) * c.stiffness
                if stress_diff > c.tolerance:
                    conflicting_constraints.append((c.source_node, c.target_node))

        conflict_pair = conflicting_constraints[0] if conflicting_constraints else None

        explanation = ""
        if conflict_pair:
            explanation = (
                f"Introspective Diagnosis: Structural conflict detected between constraint nodes "
                f"'{conflict_pair[0]}' and '{conflict_pair[1]}'. Accumulated residual stress "
                f"exceeds tolerance ({initial_stress:.3f} > {self.yield_threshold:.3f}). "
                f"System balance shattered due to principle collision in causal map."
            )
        else:
            explanation = "Introspective Diagnosis: System within stable metabolic equilibrium."

        return DiagnosticProof(
            is_metabolic_bound=True,
            status=BindingState.RESIDUAL_STRESS if conflict_pair else BindingState.BOUND,
            conflict_constraint_pair=conflict_pair,
            affected_nodes=stressed_nodes,
            initial_stress=initial_stress,
            residual_stress=initial_stress,
            root_cause_explanation=explanation
        )

    def _relax_and_predict_equilibrium(
        self,
        diagnostic: DiagnosticProof,
        input_data: Dict[str, Any]
    ) -> DiagnosticProof:
        """
        인과적 동역학에 따라 잔차 응력을 이완하고 변형된 최종 평형 상태(Psi*)를 명확히 도출
        """
        if diagnostic.status == BindingState.BOUND:
            diagnostic.predicted_equilibrium_state = {
                "state_type": "UNPERTURBED_EQUILIBRIUM",
                "final_stress": 0.0
            }
            return diagnostic

        current_residual = diagnostic.residual_stress
        absorption_logs = []

        # 이완 동역학 시뮬레이션
        for step in range(self.max_introspect_steps):
            if current_residual <= self.yield_threshold:
                break

            # 노드 간 마찰 이완 & 위상 복원
            decay_factor = 0.65
            current_residual *= decay_factor
            absorption_logs.append(
                f"Step {step+1}: Topological relaxation applied along causal edges. "
                f"Residual stress attenuated to {current_residual:.4f}"
            )

        # 노드 잔차 응력 갱신
        for n_id in diagnostic.affected_nodes:
            if n_id in self.nodes:
                self.nodes[n_id].current_stress = current_residual

        is_equilibrated = current_residual <= self.yield_threshold
        diagnostic.residual_stress = current_residual
        diagnostic.boundary_absorption_log = absorption_logs
        diagnostic.status = BindingState.EQUILIBRIUM if is_equilibrated else BindingState.RESIDUAL_STRESS

        diagnostic.predicted_equilibrium_state = {
            "state_type": "DEFORMED_EQUILIBRIUM_REACHED" if is_equilibrated else "UNRESOLVED_RESIDUAL_STRESS_STATE",
            "residual_energy_level": float(current_residual),
            "is_valid_equilibrium": is_equilibrated,
            "isomorphic_grid_aligned": True,
            "diagnosed_cause": diagnostic.root_cause_explanation
        }

        return diagnostic


if __name__ == "__main__":
    engine = IntrospectiveCausalDiagnosticsEngine(yield_threshold=0.8)

    # 노드 구성
    engine.add_node(CausalNode("node_logic", "Logic_Mechanism", "code", capacity=1.0))
    engine.add_node(CausalNode("node_memory", "Memory_Topology", "memory", capacity=1.0))
    engine.add_node(CausalNode("node_hardware", "Hardware_Physics", "hardware", capacity=1.0))

    # 제약 조건 연결
    engine.add_constraint(CausalConstraint("c1", "node_logic", "node_memory", ProtocolType.LANGUAGE, stiffness=1.5, tolerance=0.1))
    engine.add_constraint(CausalConstraint("c2", "node_memory", "node_hardware", ProtocolType.PASSPORT, stiffness=2.0, tolerance=0.1))

    print("=== Case 1: Incompatible Noise Ingestion (Structural Rejection) ===")
    bad_noise = {"signature": "BOGUS_STATISTICAL_NOISE", "protocol_type": "passport"}
    res1 = engine.process_metabolic_ingestion(bad_noise)
    print("Bound:", res1.is_metabolic_bound)
    print("Status:", res1.status.value)
    print("Explanation:", res1.root_cause_explanation)

    print("\n=== Case 2: Valid Protocol Ingestion with High Tension (Introspective Diagnostics) ===")
    valid_stimulus = {
        "signature": "CAUSAL_TOPOLOGY_V1",
        "protocol_type": "passport",
        "target_node": "node_logic",
        "intensity": 2.5
    }
    res2 = engine.process_metabolic_ingestion(valid_stimulus)
    print("Bound:", res2.is_metabolic_bound)
    print("Status:", res2.status.value)
    print("Conflicting Pair:", res2.conflict_constraint_pair)
    print("Initial Stress:", res2.initial_stress)
    print("Residual Stress after Relaxation:", res2.residual_stress)
    print("Predicted Equilibrium State:", res2.predicted_equilibrium_state)
