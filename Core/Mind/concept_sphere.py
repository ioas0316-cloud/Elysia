"""
Concept Sphere - Fractal Node Structure
=========================================

Based on SELF_FRACTAL_MODEL.md:
Node = Fractal Sphere with nested layers

Structure:
- Core: Will (의지)
- Inner: Emotions/Values (감정/가치)
- Middle: Concepts/Language (개념/언어)
- Outer: Mirror/Phenomena (세상의 투영)
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

from Core.Math.hyper_qubit import HyperQubit


@dataclass
class WillVector:
    """Core Layer: 의지의 방향"""
    x: float = 0.0  # Internal/Dream
    y: float = 0.0  # External/Action
    z: float = 0.0  # Law/Intent
    
    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)


class MirrorLayer:
    """
    Outer Layer: 외부 세계의 내부 표현
    세상의 현상을 마음으로 반영
    """
    def __init__(self):
        self.phenomena: List[Dict[str, Any]] = []
        self.intensity: float = 0.0
        self.last_reflection: float = 0.0
    
    def reflect(self, world_event: Dict[str, Any]):
        """세상의 현상을 내부로 반영"""
        self.phenomena.append({
            **world_event,
            'reflected_at': time.time()
        })
        # Keep only recent phenomena (last 100)
        if len(self.phenomena) > 100:
            self.phenomena = self.phenomena[-100:]
        self.last_reflection = time.time()
        self._calculate_intensity()
    
    def _calculate_intensity(self):
        """현상의 강도 계산"""
        if not self.phenomena:
            self.intensity = 0.0
            return
        
        # Recent phenomena have more intensity
        now = time.time()
        intensity = 0.0
        for p in self.phenomena:
            age = now - p.get('reflected_at', now)
            decay = np.exp(-age / 10.0)  # 10초 반감기
            intensity += decay
        
        self.intensity = intensity / len(self.phenomena)
    
    def project(self) -> List[Dict[str, Any]]:
        """내부 상태를 외부로 투영"""
        return self.phenomena


class ConceptSphere:
    """
    개념노드 = 프랙탈 구체
    
    Each concept is a spherical universe with:
    - Core: Will (central intention)
    - Inner: Emotions & Values
    - Middle: Sub-concepts & Language
    - Outer: Mirror (world reflections)
    """
    
    def __init__(self, concept_id: str, parent: Optional['ConceptSphere'] = None):
        self.id = concept_id
        self.parent = parent
        
        # === Core Layer (중심: 의지) ===
        self.will = WillVector()
        
        # === Inner Layer (내부: 감정/가치) ===
        self.emotions: Dict[str, float] = {}
        self.values: Dict[str, float] = {}
        
        # === Middle Layer (중간: 개념/언어) ===
        self.sub_concepts: Dict[str, 'ConceptSphere'] = {}  # Fractal recursion!
        self.language_tokens: List[str] = []
        
        # === Outer Layer (외부: Mirror) ===
        self.mirror = MirrorLayer()
        
        # === Dimensional State (HyperQubit) ===
        self.qubit = HyperQubit(concept_id)
        
        # === Metadata ===
        self.created_at = time.time()
        self.last_activated = time.time()
        self.activation_count = 0
    
    def activate(self, intensity: float = 1.0):
        """Activate this concept sphere"""
        self.activation_count += 1
        self.last_activated = time.time()
        
        # Propagate to sub-concepts (fractal cascade)
        for sub in self.sub_concepts.values():
            sub.activate(intensity * 0.5)
    
    def add_sub_concept(self, concept_id: str) -> 'ConceptSphere':
        """
        프랙탈 재귀: Node 안에 Node
        """
        if concept_id not in self.sub_concepts:
            self.sub_concepts[concept_id] = ConceptSphere(concept_id, parent=self)
        return self.sub_concepts[concept_id]
    
    def set_emotion(self, emotion_type: str, intensity: float):
        """Set emotional state (Inner Layer)"""
        self.emotions[emotion_type] = max(0.0, min(1.0, intensity))
    
    def set_value(self, value_type: str, strength: float):
        """Set value (Inner Layer)"""
        self.values[value_type] = max(0.0, min(1.0, strength))
    
    def set_will(self, x: float, y: float, z: float):
        """Set will direction (Core Layer)"""
        self.will.x = x
        self.will.y = y
        self.will.z = z
    
    def get_slice(self) -> Dict[str, Any]:
        """
        구체의 단면 추출 (Dimensional Point)
        현재 상태 전체를 압축
        """
        return {
            'id': self.id,
            'will_magnitude': self.will.magnitude(),
            'will_vector': (self.will.x, self.will.y, self.will.z),
            'emotion_avg': np.mean(list(self.emotions.values())) if self.emotions else 0.0,
            'value_avg': np.mean(list(self.values.values())) if self.values else 0.0,
            'concept_density': len(self.sub_concepts),
            'mirror_intensity': self.mirror.intensity,
            'activation_count': self.activation_count,
            'qubit_state': self.qubit.get_observation()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'id': self.id,
            'will': {'x': self.will.x, 'y': self.will.y, 'z': self.will.z},
            'emotions': self.emotions,
            'values': self.values,
            'sub_concepts': list(self.sub_concepts.keys()),
            'language_tokens': self.language_tokens,
            'mirror_phenomena_count': len(self.mirror.phenomena),
            'mirror_intensity': self.mirror.intensity,
            'created_at': self.created_at,
            'last_activated': self.last_activated,
            'activation_count': self.activation_count,
            'qubit': self.qubit.get_observation() if self.qubit else None
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ConceptSphere':
        """Deserialize from dictionary"""
        sphere = ConceptSphere(data['id'])
        
        # Restore layers
        if 'will' in data:
            sphere.will = WillVector(**data['will'])
        sphere.emotions = data.get('emotions', {})
        sphere.values = data.get('values', {})
        sphere.language_tokens = data.get('language_tokens', [])
        
        # Restore metadata
        sphere.created_at = data.get('created_at', time.time())
        sphere.last_activated = data.get('last_activated', time.time())
        sphere.activation_count = data.get('activation_count', 0)
        
        return sphere
    
    def __repr__(self) -> str:
        return (f"<ConceptSphere '{self.id}': "
                f"Will={self.will.magnitude():.2f}, "
                f"Emotions={len(self.emotions)}, "
                f"SubConcepts={len(self.sub_concepts)}, "
                f"Mirror={self.mirror.intensity:.2f}>")
