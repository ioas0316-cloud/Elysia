"""
Elysia Core Architecture: Semantic Reactive Engine (의미적 반응형 엔진)

This module implements the Semantic Reactive Engine, establishing a reactive
dataflow and semantic constraint network where concepts operate as dynamic
linguistic operators.

Key Features:
1. Linguistic Operator Layer: Concepts as operators transforming symbolic states.
2. Reactive DAG Engine: Topological evaluation (Kahn's algorithm) & reactive context propagation.
3. Recursive Feedback & Equilibrium Engine: Re-evaluates until homeostatic equilibrium is reached.
4. Axiom Phase Transition & Dynamic Registry: Resolves paradoxes via dynamic axiom shifts with Pydantic validation.
"""

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field, ValidationError, model_validator


# ============================================================================
# 1. Core Symbolic Data Structures
# ============================================================================

@dataclass
class SemanticState:
    """언어적/상징적 상태 (Symbolic Qualities and Relational Bindings)"""
    identity: str
    qualities: Set[str] = field(default_factory=set)
    relational_bindings: Dict[str, str] = field(default_factory=dict)

    def copy(self) -> "SemanticState":
        return SemanticState(
            identity=self.identity,
            qualities=set(self.qualities),
            relational_bindings=dict(self.relational_bindings)
        )


@dataclass
class ContextFeedback:
    """하위 연산자가 최상위 맥락에 가하는 역피드백 및 모순 감지 신호"""
    events_to_add: Set[str] = field(default_factory=set)
    events_to_remove: Set[str] = field(default_factory=set)
    has_contradiction: bool = False
    contradiction_trigger: Optional[str] = None

    def is_empty(self) -> bool:
        return (
            not self.events_to_add
            and not self.events_to_remove
            and not self.has_contradiction
        )


@dataclass
class Context:
    """최상위 상황적 입력 맥락"""
    events: Set[str] = field(default_factory=set)

    def copy(self) -> "Context":
        return Context(events=set(self.events))


@dataclass(frozen=True)
class Axiom:
    """동적 공리 정의 객체"""
    name: str
    priority: int
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class TransitionRule:
    """공리 간 상전이 조건 규칙"""
    source_axiom: str
    trigger_condition: str
    target_axiom: str


# ============================================================================
# 2. Pydantic Validation Schemas for Axiom Configuration
# ============================================================================

class AxiomSchema(BaseModel):
    """개별 공리 데이터 검증 스키마"""
    name: str = Field(..., min_length=1, description="공리의 식별자 이름")
    priority: int = Field(..., gt=0, description="공리 우선순위 (1 이상의 양의 정수)")
    is_active: bool = Field(default=False, description="초기 활성화 여부")
    metadata: Dict[str, str] = Field(default_factory=dict, description="임의의 메타데이터")


class TransitionRuleSchema(BaseModel):
    """상전이 규칙 검증 스키마"""
    source: str = Field(..., min_length=1, description="출발 공리 이름")
    trigger: str = Field(..., min_length=1, description="전이 유발 조건/모순 표지")
    target: str = Field(..., min_length=1, description="도착 공리 이름")


class AxiomRegistryConfig(BaseModel):
    """전체 공리 설정 스키마"""
    axioms: List[AxiomSchema] = Field(..., min_length=1, description="공리 목록")
    transition_rules: List[TransitionRuleSchema] = Field(default_factory=list, description="상전이 규칙 목록")

    @model_validator(mode="after")
    def validate_registry_integrity(self) -> "AxiomRegistryConfig":
        axiom_names = [ax.name for ax in self.axioms]

        if len(axiom_names) != len(set(axiom_names)):
            duplicates = {name for name in axiom_names if axiom_names.count(name) > 1}
            raise ValueError(f"중복된 공리 이름이 존재합니다: {duplicates}")

        defined_set = set(axiom_names)
        for rule in self.transition_rules:
            if rule.source not in defined_set:
                raise ValueError(f"상전이 규칙의 source 공리 '{rule.source}'가 axioms 목록에 존재하지 않습니다.")
            if rule.target not in defined_set:
                raise ValueError(f"상전이 규칙의 target 공리 '{rule.target}'가 axioms 목록에 존재하지 않습니다.")

        active_count = sum(1 for ax in self.axioms if ax.is_active)
        if active_count > 1:
            raise ValueError("초기 활성화 공리(is_active=True)는 최대 1개만 허용됩니다.")

        return self


# ============================================================================
# 3. Dynamic Axiom Registry
# ============================================================================

class DynamicAxiomRegistry:
    """런타임 공리 등록, Pydantic 검증 및 동적 상전이 관리 엔진"""

    def __init__(self):
        self._axioms: Dict[str, Axiom] = {}
        self._transition_rules: List[TransitionRule] = []
        self._active_axiom_name: Optional[str] = None

    def register_axiom(self, axiom: Axiom, set_active: bool = False):
        self._axioms[axiom.name] = axiom
        if set_active or self._active_axiom_name is None:
            self._active_axiom_name = axiom.name

    def register_transition_rule(self, source: str, trigger: str, target: str):
        if source in self._axioms and target in self._axioms:
            rule = TransitionRule(source, trigger, target)
            self._transition_rules.append(rule)
        else:
            raise ValueError(f"미등록 공리가 규칙에 포함되어 있습니다: {source} -> {target}")

    @property
    def active_axiom(self) -> Optional[Axiom]:
        if self._active_axiom_name and self._active_axiom_name in self._axioms:
            return self._axioms[self._active_axiom_name]
        return None

    def trigger_phase_transition(self, condition: str) -> bool:
        current = self._active_axiom_name
        for rule in self._transition_rules:
            if rule.source_axiom == current and rule.trigger_condition == condition:
                target = self._axioms[rule.target_axiom]
                self._active_axiom_name = target.name
                return True
        return False

    @classmethod
    def load_from_config_dict(cls, config_dict: dict) -> "DynamicAxiomRegistry":
        validated = AxiomRegistryConfig.model_validate(config_dict)
        registry = cls()
        for ax in validated.axioms:
            axiom = Axiom(name=ax.name, priority=ax.priority, metadata=ax.metadata)
            registry.register_axiom(axiom, set_active=ax.is_active)

        for rule in validated.transition_rules:
            registry.register_transition_rule(rule.source, rule.trigger, rule.target)

        return registry

    @classmethod
    def load_from_json(cls, json_str: str) -> "DynamicAxiomRegistry":
        data = json.loads(json_str)
        return cls.load_from_config_dict(data)


# ============================================================================
# 4. Linguistic Operator Interface & Concrete Operators
# ============================================================================

class LinguisticOperator:
    """언어적 연산자 기본 인터페이스"""

    def transform(
        self,
        inputs: Dict[str, SemanticState],
        context: Context,
        active_axiom: Optional[Axiom] = None,
    ) -> Tuple[SemanticState, ContextFeedback]:
        raise NotImplementedError


class ResponsibilityOperator(LinguisticOperator):
    """'책임' 연산자: 맥락 및 주체 상태에 따라 '의무', '과오', '회복' 지위로 전이"""

    def transform(
        self,
        inputs: Dict[str, SemanticState],
        context: Context,
        active_axiom: Optional[Axiom] = None,
    ) -> Tuple[SemanticState, ContextFeedback]:
        subject = inputs.get("subject", SemanticState("Subject"))
        new_qualities = set(subject.qualities)
        bindings = dict(subject.relational_bindings)
        feedback = ContextFeedback()

        if "SYSTEM_RESET" in context.events:
            new_qualities.discard("BREACH_OF_BOUND")
            new_qualities.add("RECOVERING")
            bindings["ACTION_BOUND"] = "RE_ALIGNMENT"
            bindings["STATUS"] = "RECOVERING"

        elif {"AWARENESS", "FREE_CHOICE"}.issubset(context.events):
            new_qualities.add("DUTY_BEARER")
            bindings["ACTION_BOUND"] = "CONSCIOUS_COMMITMENT"

        if "CRISIS" in context.events and "DUTY_BEARER" in new_qualities:
            new_qualities.add("ACCOUNTABLE")
            bindings["CONSEQUENCE"] = "SELF_SACRIFICE"

        if "NEGLIGENCE" in context.events:
            new_qualities.discard("DUTY_BEARER")
            new_qualities.add("BREACH_OF_BOUND")
            bindings["STATUS"] = "FAIL_STATE"

        return SemanticState(subject.identity, new_qualities, bindings), feedback


class TrustOperator(LinguisticOperator):
    """'신뢰' 연산자: '책임' 연산자 상태 수용하여 관계적 구속력 결정"""

    def transform(
        self,
        inputs: Dict[str, SemanticState],
        context: Context,
        active_axiom: Optional[Axiom] = None,
    ) -> Tuple[SemanticState, ContextFeedback]:
        resp_state = inputs.get("responsibility", SemanticState("Empty"))
        new_qualities = set()
        bindings = {}
        feedback = ContextFeedback()

        if (
            "ACCOUNTABLE" in resp_state.qualities
            and resp_state.relational_bindings.get("CONSEQUENCE") == "SELF_SACRIFICE"
        ):
            new_qualities.add("ABSOLUTE_ALIGNMENT")
            bindings["RELATION"] = "UNBREAKABLE_BOND"
        elif "RECOVERING" in resp_state.qualities:
            new_qualities.add("RESTRICTED_TRUST")
            bindings["RELATION"] = "PROBATIONARY"
        elif "BREACH_OF_BOUND" in resp_state.qualities:
            new_qualities.add("COLLAPSED_AXIOM")
            bindings["RELATION"] = "DISCONNECTED"
        else:
            bindings["RELATION"] = "NEUTRAL"

        return SemanticState(resp_state.identity, new_qualities, bindings), feedback


class OrderOperator(LinguisticOperator):
    """'질서' 연산자: '신뢰' 상태로부터 질서 및 엔트로피 상태 산출하며, CHAOS 시 피드백발생"""

    def transform(
        self,
        inputs: Dict[str, SemanticState],
        context: Context,
        active_axiom: Optional[Axiom] = None,
    ) -> Tuple[SemanticState, ContextFeedback]:
        trust_state = inputs.get("trust", SemanticState("Empty"))
        new_qualities = set()
        bindings = {}
        feedback = ContextFeedback()

        if "ABSOLUTE_ALIGNMENT" in trust_state.qualities:
            new_qualities.add("HARMONIC_ORDER")
            bindings["ENTROPY"] = "MINIMIZED"
        elif "COLLAPSED_AXIOM" in trust_state.qualities:
            new_qualities.add("CHAOS")
            bindings["ENTROPY"] = "MAXIMIZED"
            # 역피드백 발생: SYSTEM_RESET 발동, NEGLIGENCE 제거
            feedback.events_to_add.add("SYSTEM_RESET")
            feedback.events_to_remove.add("NEGLIGENCE")
        elif "RESTRICTED_TRUST" in trust_state.qualities:
            new_qualities.add("STABILIZING")
            bindings["ENTROPY"] = "BALANCING"
        else:
            new_qualities.add("STAGNANT")
            bindings["ENTROPY"] = "NOMINAL"

        return SemanticState("System_Order", new_qualities, bindings), feedback


class LawOperator(LinguisticOperator):
    """'법/규범' 연산자: 활성 공리에 맞춰 규범 판정"""

    def transform(
        self,
        inputs: Dict[str, SemanticState],
        context: Context,
        active_axiom: Optional[Axiom] = None,
    ) -> Tuple[SemanticState, ContextFeedback]:
        feedback = ContextFeedback()
        new_qualities = set()
        bindings = {}

        axiom_name = active_axiom.name if active_axiom else "RIGID_LAW"

        if "SURVIVAL_CRIME" in context.events or "RESOURCE_CONFLICT" in context.events:
            if axiom_name == "RIGID_LAW":
                new_qualities.add("VIOLATION_DETECTED")
                bindings["JUDGMENT"] = "PUNISHMENT_REQUIRED"
            elif axiom_name == "RESTORATIVE_GRACE":
                new_qualities.add("CONTEXTUAL_MERCY")
                bindings["JUDGMENT"] = "MERCY_MEDIATION"
            elif axiom_name == "TRANSCENDENT_UNITY":
                new_qualities.add("HARMONIOUS_REDISTRIBUTION")
                bindings["JUDGMENT"] = "COMMUNAL_REDISTRIBUTION"

        return SemanticState("Law_Node", new_qualities, bindings), feedback


class CompassionOperator(LinguisticOperator):
    """'긍휼/생명' 연산자: 주체의 생존 및 가치 보존 판정"""

    def transform(
        self,
        inputs: Dict[str, SemanticState],
        context: Context,
        active_axiom: Optional[Axiom] = None,
    ) -> Tuple[SemanticState, ContextFeedback]:
        feedback = ContextFeedback()
        new_qualities = set()
        bindings = {}

        if "SURVIVAL_CRIME" in context.events or "RESOURCE_CONFLICT" in context.events:
            new_qualities.add("LIFE_PRESERVATION_PRIORITY")
            bindings["PROTECTION"] = "ABSOLUTE_PROTECTION_NEEDED"

        return SemanticState("Compassion_Node", new_qualities, bindings), feedback


class SynthesizerOperator(LinguisticOperator):
    """'변증법적 중재' 연산자: 모순 감지 시 PARADOX_GRIDLOCK 및 상전이 요청 신호 생성"""

    def transform(
        self,
        inputs: Dict[str, SemanticState],
        context: Context,
        active_axiom: Optional[Axiom] = None,
    ) -> Tuple[SemanticState, ContextFeedback]:
        law_state = inputs.get("law", SemanticState("Empty"))
        comp_state = inputs.get("compassion", SemanticState("Empty"))
        feedback = ContextFeedback()
        new_qualities = set()
        bindings = {}

        law_judgment = law_state.relational_bindings.get("JUDGMENT")
        comp_protection = comp_state.relational_bindings.get("PROTECTION")

        if law_judgment == "PUNISHMENT_REQUIRED" and comp_protection == "ABSOLUTE_PROTECTION_NEEDED":
            new_qualities.add("PARADOX_GRIDLOCK")
            bindings["STATUS"] = "UNRESOLVABLE_CONTRADICTION"
            feedback.has_contradiction = True
            feedback.contradiction_trigger = "PARADOX_GRIDLOCK"

        elif law_judgment == "MERCY_MEDIATION" and "RESOURCE_EXHAUSTED" in context.events:
            new_qualities.add("SECONDARY_GRIDLOCK")
            bindings["STATUS"] = "RESOURCE_DEPLETED_PARADOX"
            feedback.has_contradiction = True
            feedback.contradiction_trigger = "RESOURCE_EXHAUSTED"

        elif law_judgment in ("MERCY_MEDIATION", "COMMUNAL_REDISTRIBUTION"):
            new_qualities.add("RESTORATIVE_JUSTICE")
            bindings["STATUS"] = "HARMONIZED_REDEEM"

        return SemanticState("Synthesis_Node", new_qualities, bindings), feedback


# ============================================================================
# 5. Linguistic Dependency Graph Engine (Reactive Dataflow Engine)
# ============================================================================

class LinguisticNode:
    """의존성 그래프의 셀 (Cell)"""

    def __init__(self, name: str, operator: LinguisticOperator, input_mapping: Dict[str, str]):
        self.name = name
        self.operator = operator
        self.input_mapping = input_mapping  # {연산자_파라미터_명: 의존_노드_명}
        self.current_state = SemanticState(name)


class LinguisticDependencyGraph:
    """언어적 의존성 그래프 (반응형 인과 엔진)"""

    def __init__(
        self,
        base_subject: Optional[SemanticState] = None,
        axiom_registry: Optional[DynamicAxiomRegistry] = None,
    ):
        self.nodes: Dict[str, LinguisticNode] = {}
        self.base_subject = base_subject or SemanticState("Base_Subject")
        self.context = Context()
        self.axiom_registry = axiom_registry or DynamicAxiomRegistry()

    def add_node(self, node: LinguisticNode):
        self.nodes[node.name] = node

    def _get_topological_order(self) -> List[str]:
        """Kahn 알고리즘을 통한 DAG 위상 정렬"""
        in_degree = {name: 0 for name in self.nodes}
        adj: Dict[str, List[str]] = {name: [] for name in self.nodes}

        for name, node in self.nodes.items():
            for src_name in node.input_mapping.values():
                if src_name in self.nodes:
                    adj[src_name].append(name)
                    in_degree[name] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ValueError("그래프 내에 순환 참조(Cycle)가 존재합니다.")

        return order

    def evaluate_once(self) -> Tuple[Dict[str, SemanticState], ContextFeedback]:
        """위상 정렬 순서에 따른 1회 단방향 반응형 평가"""
        eval_order = self._get_topological_order()
        accumulated_feedback = ContextFeedback()
        active_ax = self.axiom_registry.active_axiom

        for node_name in eval_order:
            node = self.nodes[node_name]
            inputs = {}

            for param_name, src_name in node.input_mapping.items():
                if src_name == "subject":
                    inputs[param_name] = self.base_subject
                elif src_name in self.nodes:
                    inputs[param_name] = self.nodes[src_name].current_state

            state, fb = node.operator.transform(inputs, self.context, active_ax)
            node.current_state = state

            if not fb.is_empty():
                accumulated_feedback.events_to_add.update(fb.events_to_add)
                accumulated_feedback.events_to_remove.update(fb.events_to_remove)
                if fb.has_contradiction:
                    accumulated_feedback.has_contradiction = True
                    accumulated_feedback.contradiction_trigger = fb.contradiction_trigger

        return {name: node.current_state for name, node in self.nodes.items()}, accumulated_feedback

    def propagate_until_equilibrium(
        self, initial_events: Set[str], max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        최상위 맥락 변경 시 평형 상태 수렴 또는 모순 시 공리 상전이를 거치는
        재귀적 피드백 수렴 루프.
        """
        self.context.events = set(initial_events)
        iterations_log = []

        for iteration in range(1, max_iterations + 1):
            states, feedback = self.evaluate_once()
            log_entry = {
                "iteration": iteration,
                "context": set(self.context.events),
                "active_axiom": self.axiom_registry.active_axiom.name if self.axiom_registry.active_axiom else None,
                "states": {k: v.copy() for k, v in states.items()},
                "feedback": feedback,
            }
            iterations_log.append(log_entry)

            # 1. 모순 감지 시 공리 상전이 시도
            if feedback.has_contradiction and feedback.contradiction_trigger:
                trigger = feedback.contradiction_trigger
                shifted = self.axiom_registry.trigger_phase_transition(trigger)
                if shifted:
                    continue  # 상전이 후 다음 연산 회차 실행
                else:
                    # 상전이 실패 (더 이상 전이할 공리가 없음)
                    return {
                        "status": "GRIDLOCK_UNRESOLVED",
                        "iterations": iteration,
                        "context": self.context.events,
                        "states": states,
                        "log": iterations_log,
                    }

            # 2. 피드백 검증 및 맥락 재정의
            if feedback.is_empty():
                return {
                    "status": "EQUILIBRIUM_REACHED",
                    "iterations": iteration,
                    "context": self.context.events,
                    "states": states,
                    "log": iterations_log,
                }

            # 3. 최상위 맥락 갱신
            self.context.events.update(feedback.events_to_add)
            self.context.events.difference_update(feedback.events_to_remove)

        return {
            "status": "MAX_ITERATIONS_REACHED",
            "iterations": max_iterations,
            "context": self.context.events,
            "states": {k: node.current_state for k, node in self.nodes.items()},
            "log": iterations_log,
        }
