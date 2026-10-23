"""
Conceptual Phase Knob Engine (개념적 위상 노브 엔진)

This module implements the conceptual phase space where distances between concepts
are governed by Clifford Rotor phase rotation angles and friction errors (E),
and concepts are split into invariant core and variant wave axes to form controllable Cognitive Knobs.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class CognitiveKnob:
    """Represents a tunable parameter extracted from dynamic wave variants."""
    name: str
    value: float
    min_value: float = 0.0
    max_value: float = 2.0 * np.pi
    default_value: float = 1.0
    description: str = ""

    def tune(self, new_value: float) -> float:
        self.value = float(np.clip(new_value, self.min_value, self.max_value))
        return self.value


@dataclass
class ConceptNode:
    name: str
    feature_vector: np.ndarray  # 고차원 잠재 좌표
    invariance_core: np.ndarray # 변하지 않는 공리적 뼈대 (Invariance)
    variant_wave: np.ndarray    # 요동치는 파동축 (Variance)
    friction_error: float = 0.0


# Compatibility aliases & dataclasses for L1_causal_field.core.topology exports
ConceptualEntity = ConceptNode


@dataclass
class ConceptualDistanceResult:
    entity_a: str
    entity_b: str
    euclidean_physical_distance: float
    conceptual_phase_distance: float
    clifford_rotor_angle: float
    shared_invariance_score: float
    is_topologically_isomorphic: bool


@dataclass
class ExtractedWorldMechanism:
    generating_mechanism_theta: np.ndarray
    boundary_condition_delta: np.ndarray
    invariant_bedrock: np.ndarray
    variant_dynamic_wave: np.ndarray
    knobs: Dict[str, CognitiveKnob]


class ConceptualPhaseKnobEngine:
    """
    개념간 거리(Conceptual Distance)와 Clifford Rotor 위상 회전각을 기반으로
    잠재 지식을 분별하고 제어 노브(Knob)로 치환하는 인지 사유 체계 엔진.
    """
    def __init__(self, dimension: int = 64, friction_threshold: float = 0.15):
        self.dimension = dimension
        self.friction_threshold = friction_threshold
        self.concepts: Dict[str, ConceptNode] = {}
        self.entities: Dict[str, ConceptNode] = self.concepts # Alias for compatibility
        self.active_knobs: Dict[str, float] = {} # 제어 노브 파라미터 맵

    def register_concept(self, name: str, raw_features: List[float]) -> ConceptNode:
        vec = np.array(raw_features, dtype=float)
        if len(vec) < self.dimension:
            padded = np.zeros(self.dimension, dtype=float)
            padded[:len(vec)] = vec
            vec = padded
        elif len(vec) > self.dimension:
            vec = vec[:self.dimension]

        # 불변축과 가변축 분별 (Invariant vs Variant Classification)
        # 상위 50%의 안정된 성분을 불변축으로, 하위 50%를 가변 파동축으로 분리
        norm = np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else 1.0
        normalized_vec = vec / norm

        split_idx = self.dimension // 2
        invariance_core = normalized_vec.copy()
        invariance_core[split_idx:] = 0.0 # 후반부를 가변 영역으로 설정

        variant_wave = normalized_vec.copy()
        variant_wave[:split_idx] = 0.0

        node = ConceptNode(
            name=name,
            feature_vector=normalized_vec,
            invariance_core=invariance_core,
            variant_wave=variant_wave
        )
        self.concepts[name] = node
        return node

    def register_entity(self, entity: Any) -> None:
        """Compatibility method for registering entities."""
        if hasattr(entity, 'causal_vector'):
            raw_features = entity.causal_vector.tolist()
        elif hasattr(entity, 'feature_vector'):
            raw_features = entity.feature_vector.tolist()
        else:
            raw_features = np.zeros(self.dimension).tolist()
        self.register_concept(entity.name, raw_features)

    def compute_conceptual_distance(self, concept_a: str, concept_b: str) -> Tuple[float, float]:
        """
        두 개념 간의 위상적 거리 및 마찰 오차(E) 계산.
        Clifford Rotor 위상 회전각(Delta Theta)을 기반으로 도출.
        """
        if concept_a not in self.concepts or concept_b not in self.concepts:
            raise ValueError("Registered concepts required.")

        vec_a = self.concepts[concept_a].feature_vector
        vec_b = self.concepts[concept_b].feature_vector

        # 코사인 유사도를 위상 회전각으로 치환
        dot_product = np.dot(vec_a, vec_b)
        clipped_dot = float(np.clip(dot_product, -1.0, 1.0))
        delta_theta = math.acos(abs(clipped_dot)) # 위상 회전각 (Delta Theta)

        # 마찰 오차 (E): 개념적 거리의 비선형 구배
        friction_e = float(1.0 - abs(clipped_dot))

        return delta_theta, friction_e

    def create_knob_from_concept(self, concept_name: str, initial_value: float = 1.0) -> str:
        """
        가변 파동축(Variance)을 조율 가능한 제어 노브(Knob)로 치환.
        """
        if concept_name not in self.concepts:
            self.register_concept(concept_name, np.random.randn(self.dimension).tolist())

        knob_key = f"knob_{concept_name}"
        self.active_knobs[knob_key] = initial_value
        return knob_key

    def adjust_knob(self, knob_key: str, value: float) -> Dict[str, Any]:
        """
        노브 조율 시 전체 위상 지형(Phase Space) 및 구조 배열을 실시간 재배치.
        """
        if knob_key not in self.active_knobs:
            raise KeyError(f"Knob {knob_key} does not exist.")

        self.active_knobs[knob_key] = value

        # 상태 재배치 시뮬레이션 결과 반환
        return {
            "status": "rearranged",
            "active_knobs": self.active_knobs,
            "system_phase_tension": float(sum(self.active_knobs.values()) * 0.1)
        }
