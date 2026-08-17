"""
[Higher-Order Hypergraph Engine: 고차 재귀적 하이퍼그래프 및 위상망 순회 평가 엔진]

데이터(Atom)와 연산/과정(Process)의 구분을 파괴하고,
모든 연결, 연산, 인과 과정을 재귀적 식별자(ID)를 갖는 제1급 개체(First-Class Entity)로 승격시킵니다.
상위 프로세스가 하위 프로세스의 결과나 과정을 입력/출력으로 다루며,
지연된 재귀 평가(Lazy Recursive Resolution), 위상적 맥락 주입(Context Injection),
상태 전이의 동적 영향력(Contextual Re-calibration), 인과 궤적 영구화(Causal Trace Provenance)를 지원합니다.

또한 의도(Intent) 및 선언적 제약(Declarative Constraint) 기반의 포텐셜 이완 연산(Potential Relaxation Engine)을
통해 상태 공간 내 제약 만족 및 에너지 평형점(Equilibrium)으로의 자율 수렴을 지원합니다.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional, Callable, Tuple
import uuid
import copy
import math
import numpy as np

from synaptic_architecture.inverse_mechanism_engine import (
    BoundaryCondition,
    GeneratingMechanism,
    InverseMechanismEngine,
    ObservedTrajectory
)


@dataclass
class HyperEntity:
    """모든 노드, 관계, 과정의 최상위 추상 개체 (First-Class Entity)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    meta_attributes: Dict[str, Any] = field(default_factory=dict)
    causal_trace: List[str] = field(default_factory=list)  # 인과 생성 이력 (Process IDs)


@dataclass
class AtomicNode(HyperEntity):
    """더 이상 쪼개지지 않는 단일 데이터/상태 점 (Atom)"""
    value: Any = None


@dataclass
class ProcessNode(HyperEntity):
    """
    과정/관계 자체가 노드가 되는 하이퍼 개체 (Hyper-Node / First-Class Process)
    입력(inputs) 및 출력(outputs)은 AtomicNode일 수도, 다른 ProcessNode일 수도 있음.
    """
    inputs: List[str] = field(default_factory=list)    # 참조할 HyperEntity IDs
    outputs: List[str] = field(default_factory=list)   # 생성/변이될 HyperEntity IDs

    # 상태 전이를 일으키는 인과 연산자 및 규칙
    causal_operator: str = "IDENTITY"                  # "IDENTITY", "ADD", "MULTIPLY", "MECHANISM_EXTRAPOLATION", "RELAXATION", "CUSTOM"
    custom_fn: Optional[Callable[..., Any]] = None      # 커스텀 연산 함수
    mechanism: Optional[GeneratingMechanism] = None     # 역메커니즘 엔진의 Θ 방정식 연동

    boundary_conditions: Dict[str, Any] = field(default_factory=dict) # 위상적 경계 조건 / 맥락 가중치
    stiffness: float = 1.0                             # 동적 조정 파라미터 / 결합 강성


class PotentialRelaxationEngine:
    """
    [포텐셜 이완 연산기 (Potential Relaxation Engine)]
    의도 포텐셜 E_intent(S)와 경계 제약 B(S) = 0 의 합성 에너지를 정의하고,
    수평/수직 제약 필드 내에서 이완(Relaxation / Gradient Descent)하여 에너지 평형 상태를 유도합니다.
    """

    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.state = np.random.randn(state_dim)
        self.constraints: List[Tuple[Callable[[np.ndarray], float], float]] = []
        self.intent_energy_fn: Optional[Callable[[np.ndarray], float]] = None

    def set_intent(self, energy_fn: Callable[[np.ndarray], float]):
        """달성하고자 하는 의도를 포텐셜 최소화 함수 E_intent(S)로 정의"""
        self.intent_energy_fn = energy_fn

    def add_boundary_constraint(
        self,
        constraint_fn: Callable[[np.ndarray], float],
        penalty_weight: float = 1000.0
    ):
        """위배할 수 없는 인과적 경계 제약 B(S) 추가"""
        self.constraints.append((constraint_fn, penalty_weight))

    def compute_total_energy(self, state: np.ndarray) -> float:
        """전체 시스템 에너지 V(S) = E_intent(S) + sum(weight * B(S)^2) 계산"""
        energy = 0.0
        if self.intent_energy_fn:
            energy += self.intent_energy_fn(state)
        for constraint_fn, weight in self.constraints:
            violation = constraint_fn(state)
            energy += weight * (violation ** 2)
        return energy

    def relax_to_equilibrium(
        self,
        initial_state: Optional[np.ndarray] = None,
        steps: int = 200,
        lr: float = 0.01
    ) -> np.ndarray:
        """
        절차적 조건문 없이, 경계 제약 필드 내에서 에너지가 최소화되는 평형 상태로 이완
        """
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=float)

        state = self.state.copy()
        eps = 1e-5

        for _ in range(steps):
            # 수치 미분(Numerical Gradient)을 통한 포텐셜 기울기 계산
            grad = np.zeros_like(state)
            base_energy = self.compute_total_energy(state)

            for i in range(len(state)):
                state_plus = state.copy()
                state_plus[i] += eps
                energy_plus = self.compute_total_energy(state_plus)
                grad[i] = (energy_plus - base_energy) / eps

            # 포텐셜 이완 (기울기 반대 방향 이동)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 10.0:
                grad = grad * (10.0 / grad_norm)

            state -= lr * grad

        self.state = state
        return self.state


class HigherOrderHypergraphEngine:
    """
    [Higher-Order Hypergraph & Traversal Engine]
    고차 하이퍼그래프 데이터 구조의 등록, 재귀 순회, 지연 평가, 맥락 주입 및 동적 재조정을 관리합니다.
    """

    def __init__(self):
        self.entities: Dict[str, HyperEntity] = {}
        self.inverse_engine = InverseMechanismEngine()

    def register_entity(self, entity: HyperEntity) -> HyperEntity:
        """엔티티(AtomicNode 또는 ProcessNode)를 하이퍼그래프 필드에 등록합니다."""
        self.entities[entity.id] = entity
        return entity

    def get_entity(self, entity_id: str) -> Optional[HyperEntity]:
        """ID로 엔티티를 조회합니다."""
        return self.entities.get(entity_id)

    def connect_process(
        self,
        process_node: ProcessNode,
        inputs: List[HyperEntity],
        outputs: Optional[List[HyperEntity]] = None
    ) -> ProcessNode:
        """
        프로세스 노드에 입력 및 출력 엔티티들을 결합합니다.
        입/출력 엔티티 자체가 다른 ProcessNode일 수도 있습니다 (재귀적 고차 연결).
        """
        for inp in inputs:
            if inp.id not in self.entities:
                self.register_entity(inp)
            if inp.id not in process_node.inputs:
                process_node.inputs.append(inp.id)

        if outputs:
            for outp in outputs:
                if outp.id not in self.entities:
                    self.register_entity(outp)
                if outp.id not in process_node.outputs:
                    process_node.outputs.append(outp.id)

        self.register_entity(process_node)
        return process_node

    def evaluate_lazy(
        self,
        target_entity_id: str,
        context_override: Optional[Dict[str, Any]] = None,
        visited: Optional[set] = None
    ) -> Any:
        """
        [지연된 재귀 평가 (Lazy Recursive Resolution)]
        상위 프로세스 또는 결과 노드의 연산을 요구받으면,
        해당 엔티티를 생성하는 하위 프로세스 및 입력 노드들의 연산자와 경계 조건을 역트리 구조로 추적하여 연쇄 전이시킵니다.
        """
        if visited is None:
            visited = set()

        if target_entity_id in visited:
            # 순환 참조 방지 및 현재 저장된 상태 반환
            entity = self.entities.get(target_entity_id)
            if isinstance(entity, AtomicNode):
                return entity.value
            return entity

        visited.add(target_entity_id)
        entity = self.entities.get(target_entity_id)
        if entity is None:
            raise KeyError(f"Entity with ID {target_entity_id} not found in Hypergraph.")

        # AtomicNode 인 경우: 자신을 생성한 하위 ProcessNode가 있다면 그 연산 결과로 지연 업데이트
        if isinstance(entity, AtomicNode):
            # target_entity_id 가 어떤 ProcessNode의 outputs에 들어있는지 검색
            producer_processes = [
                p for p in self.entities.values()
                if isinstance(p, ProcessNode) and target_entity_id in p.outputs
            ]
            if not producer_processes:
                return entity.value

            # producer process 들의 평가를 통해 엔티티 값 갱신
            for proc in producer_processes:
                computed_val = self._evaluate_process(proc, context_override, visited.copy())
                entity.value = computed_val
                if proc.id not in entity.causal_trace:
                    entity.causal_trace.append(proc.id)
            return entity.value

        # ProcessNode 인 경우: 해당 프로세스의 연산을 수행하여 평가 결과 반환
        elif isinstance(entity, ProcessNode):
            return self._evaluate_process(entity, context_override, visited.copy())

        return None

    def _evaluate_process(
        self,
        process: ProcessNode,
        context_override: Optional[Dict[str, Any]],
        visited: set
    ) -> Any:
        """
        단일 ProcessNode의 인과 연산 평가 및 맥락 주입 (Context Injection)
        """
        # 1. 입력 엔티티들을 재귀적으로 평가하여 값 추출
        resolved_inputs = []
        for inp_id in process.inputs:
            resolved_val = self.evaluate_lazy(inp_id, context_override, visited)
            resolved_inputs.append(resolved_val)

        # 2. 위상적 맥락 주입 (Context Injection & Re-calibration)
        effective_boundary = copy.deepcopy(process.boundary_conditions)
        if context_override:
            effective_boundary.update(context_override)

        # 맥락 주입에 따른 스티프니스/가중치 재조정 (Dynamic Re-calibration)
        scale_mod = effective_boundary.get("scale", 1.0) * process.stiffness
        friction_mod = effective_boundary.get("friction", 1.0)

        # 3. 인과 연산자 (Causal Operator) 적용
        op = process.causal_operator.upper()
        result_value = None

        if op == "IDENTITY":
            result_value = resolved_inputs[0] if resolved_inputs else None

        elif op == "ADD":
            total = 0.0
            for val in resolved_inputs:
                if isinstance(val, (int, float)):
                    total += val
                elif isinstance(val, list):
                    total += sum(val)
                elif isinstance(val, dict) and "output" in val:
                    v = val["output"]
                    if isinstance(v, (int, float)):
                        total += v
            result_value = total * scale_mod / friction_mod

        elif op == "MULTIPLY":
            prod = 1.0
            for val in resolved_inputs:
                if isinstance(val, (int, float)):
                    prod *= val
            result_value = prod * scale_mod

        elif op == "MECHANISM_EXTRAPOLATION":
            # 역메커니즘 Θ 기반 궤적 생성
            if process.mechanism is None:
                raise ValueError(f"ProcessNode {process.id} missing GeneratingMechanism for MECHANICAL_EXTRAPOLATION.")

            init_state = resolved_inputs[0] if (resolved_inputs and isinstance(resolved_inputs[0], list)) else [1.0, 1.0]
            bc = BoundaryCondition(
                condition_id=f"bc_{process.id}",
                friction=effective_boundary.get("friction", 1.0),
                scale=effective_boundary.get("scale", 1.0),
                gravity=effective_boundary.get("gravity", 9.81),
                temperature=effective_boundary.get("temperature", 1.0)
            )
            traj = self.inverse_engine.generate_trajectory(
                mechanism=process.mechanism,
                boundary=bc,
                initial_state=init_state,
                steps=effective_boundary.get("steps", 5)
            )
            result_value = traj

        elif op == "RELAXATION":
            # 포텐셜 이완 기반 평형 수렴
            state_dim = effective_boundary.get("state_dim", 2)
            relax_engine = PotentialRelaxationEngine(state_dim=state_dim)

            intent_target = effective_boundary.get("intent_target", np.zeros(state_dim))
            relax_engine.set_intent(lambda s: float(np.sum((s - intent_target) ** 2)))

            if "barrier" in effective_boundary:
                barrier_val = effective_boundary["barrier"]
                relax_engine.add_boundary_constraint(lambda s: float(np.maximum(0.0, np.sum(s) - barrier_val)))

            init_st = resolved_inputs[0] if (resolved_inputs and isinstance(resolved_inputs[0], (list, np.ndarray))) else None
            steps = effective_boundary.get("relaxation_steps", 100)
            eq_state = relax_engine.relax_to_equilibrium(initial_state=init_st, steps=steps)
            result_value = eq_state.tolist()

        elif op == "CUSTOM" and process.custom_fn is not None:
            result_value = process.custom_fn(resolved_inputs, effective_boundary)

        else:
            # 기본 결합: 입력 리스트 및 맥락 파라미터 반환
            result_value = {
                "inputs": resolved_inputs,
                "effective_boundary": effective_boundary,
                "stiffness": process.stiffness,
                "operator": op
            }

        # 4. 인과 궤적 영구화 (Causal Trace Provenance)
        for outp_id in process.outputs:
            outp_entity = self.entities.get(outp_id)
            if outp_entity:
                outp_entity.value = result_value
                if process.id not in outp_entity.causal_trace:
                    outp_entity.causal_trace.append(process.id)

        return result_value

    def trigger_contextual_recalibration(
        self,
        trigger_entity_id: str,
        new_boundary_delta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        [상태 전이의 동적 영향력 (Contextual Re-calibration)]
        새로운 관계/엔티티 변이가 네트워크에 주입되었을 때,
        연쇄 반응처럼 직·간접적으로 얽힌 기존 프로세스 노드들의 경계 조건을 미세 재조정(Re-calibrate)합니다.
        """
        recalibrated_processes = []
        entity = self.entities.get(trigger_entity_id)
        if not entity:
            return {"recalibrated_count": 0, "affected_processes": []}

        # 트리거 엔티티와 상위/하위 연결된 모든 ProcessNode 탐색
        for proc in self.entities.values():
            if isinstance(proc, ProcessNode):
                if trigger_entity_id in proc.inputs or trigger_entity_id in proc.outputs or proc.id == trigger_entity_id:
                    proc.boundary_conditions.update(new_boundary_delta)
                    # 경계 조건 변경에 따른 스티프니스 재조정
                    if "friction" in new_boundary_delta:
                        proc.stiffness *= (1.0 / max(new_boundary_delta["friction"], 1e-5))
                    recalibrated_processes.append(proc.id)

        return {
            "recalibrated_count": len(recalibrated_processes),
            "affected_processes": recalibrated_processes
        }
