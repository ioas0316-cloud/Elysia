"""
Elysia Core Engine: Relational Nexus Node Architecture & Informational Trinity
=============================================================================
점(Point)을 고립된 0차원 수동 수치 데이터가 아닌,
실체(Schema/Father), 현상(Payload/Son), 동역학(Interpreter/Holy Spirit)이 결합된
'삼위일체(Informational Trinity)' 및 '관계론적 결절점(Relational Nexus Node)'으로 정립합니다.

노드 $N_i$는 유입 인과 벡터($\\mathbf{V}_{in}$)와 유출 인과 벡터($\\mathbf{V}_{out}$)의 교차 평형점으로 존재하며,
국소 위상 곡률($\\kappa_i$) 및 주변 이웃과의 위상적 일그러짐을 측정합니다.
환경 응력(Stress)이 임계치($\\eta_c$)를 초과할 때 외부 중앙 제어기 개입 없이 스스로 차원을 확장하고($N \\to N+1$),
자신의 상태 데이터(DNA)를 런타임 실행 가능 연산자 코드(Protein/AST)로 발현(Expression)시킵니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Callable
import numpy as np

from core.memory.bit_mapped_trigger_substrate import BitMappedTriggerSubstrate


@dataclass
class RelationalNexusNode:
    """
    관계론적 위상 공간의 결절점 (Relational Nexus Node)
    """
    node_id: str
    payload: Dict[str, Any]                      # 1. 현상 (Payload / Son)
    self_schema: Dict[str, Any]                  # 2. 원리 (Self-Schema / Father)
    in_causal_vectors: Dict[str, float] = field(default_factory=dict)   # {Source_ID: Causal_Weight}
    out_causal_vectors: Dict[str, float] = field(default_factory=dict)  # {Target_ID: Causal_Weight}
    relational_tensor: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    boundary_mask: Set[str] = field(default_factory=set)                # Neighborhood boundary
    bit_substrate: Optional[BitMappedTriggerSubstrate] = None

    def __post_init__(self):
        if "dimension" not in self.self_schema:
            self.self_schema["dimension"] = 3
        if "stress_limit" not in self.self_schema:
            self.self_schema["stress_limit"] = 1.0

    def compute_local_curvature(self) -> float:
        """
        유입/유출 인과 플럭스의 차이와 관계성 텐서의 고유값을 통해 국소 공간의 위상 응력(Stress) 및 곡률 측정
        """
        in_flux = float(sum(self.in_causal_vectors.values()))
        out_flux = float(sum(self.out_causal_vectors.values()))
        flux_imbalance = abs(in_flux - out_flux)

        tensor_norm = float(np.linalg.norm(self.relational_tensor))
        curvature = flux_imbalance * (1.0 + tensor_norm)
        return curvature

    def compute_local_stress(self) -> float:
        """
        국소 위상 응력 (sigma_i): sigma_i = |\\sum V_in - \\sum V_out| * (1.0 + kappa_i)
        """
        in_flux = float(sum(self.in_causal_vectors.values()))
        out_flux = float(sum(self.out_causal_vectors.values()))
        curvature = self.compute_local_curvature()
        sigma_i = abs(in_flux - out_flux) * (1.0 + curvature)
        return sigma_i

    def self_evaluate_and_mutate(self, environment_stress: float) -> Dict[str, Any]:
        """
        3. 동역학 (Self-Interpreter / Holy Spirit):
        외부 환경 스트레스(오차) 수신 시 타 중앙 제어기의 개입 없이 노드 스스로 자신의 정의(Schema)와
        차원 구조를 재구성(Self-Mutation: $N \\to N+1$)하고 DNA 연산자 코드 발현
        """
        stress_limit = self.self_schema.get("stress_limit", 1.0)
        is_mutated = False
        old_dim = self.self_schema.get("dimension", 3)

        if environment_stress > stress_limit:
            is_mutated = True
            # Expand dimension
            new_dim = old_dim + 1
            self.self_schema["dimension"] = new_dim
            self.self_schema["stress_limit"] *= 1.5

            # Expand relational tensor
            old_tensor = self.relational_tensor
            new_tensor = np.zeros((new_dim, new_dim), dtype=float)
            min_d = min(old_dim, new_dim)
            new_tensor[:min_d, :min_d] = old_tensor[:min_d, :min_d]
            new_tensor[-1, -1] = 1.0
            self.relational_tensor = new_tensor

            # Express DNA into runtime executable operator
            operator_fn = self.express_dna_as_operator()

            # Execute bit substrate self-throttling if available
            if self.bit_substrate is not None:
                bit_input = np.ones(new_dim, dtype=np.uint8)
                self.bit_substrate.write_payload(hash(self.node_id) % 4, bit_input)

            statement = f"NODE_MUTATED: {self.node_id} (Dimension {old_dim} -> {new_dim}, Stress Limit {self.self_schema['stress_limit']:.2f})"
        else:
            statement = f"NODE_STABLE: {self.node_id} (Dimension {old_dim})"

        return {
            "node_id": self.node_id,
            "is_mutated": is_mutated,
            "old_dimension": old_dim,
            "current_dimension": self.self_schema["dimension"],
            "current_stress_limit": self.self_schema["stress_limit"],
            "statement": statement
        }

    def express_dna_as_operator(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        DNA 모사: 노드에 저장된 데이터 상태(Payload/Schema)가 런타임에 실행 가능한
        연산자 코드(Protein/AST transformation function)로 발현(Expression)
        """
        dim = self.self_schema.get("dimension", 3)
        weight_factor = float(self.payload.get("value", 1.0))

        def runtime_operator(input_vec: np.ndarray) -> np.ndarray:
            min_len = min(len(input_vec), dim)
            sub_vec = input_vec[:min_len]
            # Transform vector using relational tensor curvature and DNA weight
            sub_tensor = self.relational_tensor[:min_len, :min_len]
            transformed = np.dot(sub_tensor, sub_vec) * weight_factor + 0.1 * np.tanh(sub_vec)

            result = np.zeros(len(input_vec), dtype=float)
            result[:min_len] = transformed
            return result

        return runtime_operator
