"""
Concept Formation (개념 형성)
===========================

"악보(Score)를 읽고, 다시 쓰다"

이 모듈은 '개념(Concept)'을 형성하고 발전시킵니다.
기존의 AestheticWisdom이 고정된 '규칙'이었다면,
ConceptFormation은 경험(MemoryStream)을 통해 진화하는 '가설'입니다.

- Aesthetic: 미학적 개념 (Vector)
- Logic: 논리적 개념 (Rule)
- Linguistic: 언어적 개념 (Grammar)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
import math
import json
import os
from pathlib import Path

# Domains
from Core.Philosophy.aesthetic_principles import AestheticVector, AestheticWisdom, get_aesthetic_wisdom
from Core.Cognitive.memory_stream import MemoryStream, get_memory_stream, ExperienceType

@dataclass
class ConceptScore:
    """
    개념의 악보 (Concept Score)
    
    Polymorphic Score Container.
    Can hold Art (Vector), Math (Rule), or Language (Pattern).
    """
    name: str
    domain: str                    # "aesthetic", "logic", "linguistic", "meta", "social"
    value: Any                     # Polymorphic value (Vector, Rule, etc.)
    confidence: float = 0.5        # 확신도 (0~1)
    evolution_stage: int = 0       # 진화 단계
    
    # Synesthesia (Unified Field)
    meta_properties: List[str] = field(default_factory=list) # e.g. ["Recursive", "Symmetry"]
    synaptic_links: List[str] = field(default_factory=list)  # e.g. ["Logic:Fibonacci"]
    
    # Cognitive Chemistry (Valence)
    valence: List[str] = field(default_factory=list) # e.g. needed elements for bonding
    
    # 연관된 경험들의 ID (근거)
    supporting_memories: List[str] = field(default_factory=list)

    @property
    def vector(self) -> Optional[AestheticVector]:
        """Backward compatibility for Aesthetic domain"""
        if self.domain == "aesthetic" and isinstance(self.value, AestheticVector):
            return self.value
        return None

    def describe(self) -> str:
        val_str = str(self.value)
        if self.domain == "aesthetic":
            val_str = "Vector(...)"
        return f"Concept '{self.name}' [{self.domain}] (Conf: {self.confidence:.2f}, Links: {len(self.synaptic_links)}, Valence: {self.valence})"

    def to_dict(self) -> Dict:
        """Serialize for JSON storage"""
        data = asdict(self)
        # Handle special types like AestheticVector
        if self.domain == "aesthetic" and isinstance(self.value, AestheticVector):
            data['value'] = {"w": self.value.w, "x": self.value.x, "y": self.value.y, "z": self.value.z, "__type__": "AestheticVector"}
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConceptScore':
        """Deserialize from JSON"""
        # Restore special types
        val = data.get('value')
        if isinstance(val, dict) and val.get("__type__") == "AestheticVector":
            data['value'] = AestheticVector(val['w'], val['x'], val['y'], val['z'])
        return cls(**data)


class ConceptFormation:
    """
    개념 형성 엔진
    """
    
    def __init__(self, persistence_path: str = "data/memory/concepts.json"):
        self.memory = get_memory_stream()
        self.wisdom = get_aesthetic_wisdom()
        self.persistence_path = Path(persistence_path)
        
        # 형성된 개념들 (Cache)
        self.concepts: Dict[str, ConceptScore] = {}
        
        # Load attempts
        self.load_concepts()
        
    def get_concept(self, name: str) -> ConceptScore:
        """
        특정 개념에 대한 '악보'를 가져옵니다.
        없으면 초기화(Initial Score)합니다.
        """
        # Case insensitive check could be added here
        if name not in self.concepts:
            # Default to aesthetic if unknown, or infer from name?
            # For now, default to aesthetic for backward compat.
            self._initialize_concept(name, domain="aesthetic")
            
        return self.concepts[name]
    
    def learn_concept(self, name: str, context: str, domain: str = "aesthetic", meta_tags: List[str] = None, valence: List[str] = None):
        """
        새로운 개념 학습 (Explicit Learning)
        """
        if name not in self.concepts:
            self._initialize_concept(name, domain, meta_tags, valence)
        else:
            # Update existing?
            pass
        self.save_concepts()
            
    def _initialize_concept(self, name: str, domain: str = "aesthetic", meta_tags: List[str] = None, valence: List[str] = None):
        """
        개념 최초 형성
        """
        initial_value = None
        
        if domain == "aesthetic":
            initial_value = AestheticVector(0.5, 0.5, 0.5, 0.5) 
        elif domain == "logic":
            initial_value = {"rule": "unknown", "consistency": 0.0}
        elif domain == "linguistic":
            initial_value = {"pattern": [], "frequency": 0}
        elif domain == "social" or domain == "meta":
            initial_value = {"status": "forming"}
            
        self.concepts[name] = ConceptScore(
            name=name,
            domain=domain,
            value=initial_value,
            confidence=0.1, # New concepts are weak
            evolution_stage=0,
            meta_properties=meta_tags or [],
            valence=valence or []
        )
        
    def evolve_concept(self, name: str):
        """
        개념 진화 (The Reflection)
        
        Domain-specific evolution logic.
        """
        concept = self.get_concept(name)
        
        # 1. 관련 기억 회상
        recent_memories = self.memory.get_recent_experiences(limit=100)
        relevant_memories = [m for m in recent_memories if m.score.get("intent") == name]
                
        if not relevant_memories:
            return

        # 2. Domain Specific Logic
        if concept.domain == "aesthetic":
            self._evolve_aesthetic(concept, relevant_memories)
        elif concept.domain == "logic":
            self._evolve_logic(concept, relevant_memories)
            
        # 3. Save
        self.save_concepts()
            
    def _evolve_aesthetic(self, concept: ConceptScore, memories: List[Any]):
        """Evolve Aesthetic Vector (Legacy Logic)"""
        success_weight = 0.0
        vec = concept.value # AestheticVector
        new_w, new_x, new_y, new_z = 0.0, 0.0, 0.0, 0.0
        
        for mem in memories:
            rating = mem.sound.get("aesthetic_score", 0) / 100.0
            if rating > 0.5:
                # Mock reinforcement
                new_w += vec.w * rating
                new_x += vec.x * rating
                new_y += vec.y * rating
                new_z += vec.z * rating
                success_weight += rating
        
        if success_weight > 0:
            vec.w = (vec.w * 0.7) + ((new_w / success_weight) * 0.3)
            vec.x = (vec.x * 0.7) + ((new_x / success_weight) * 0.3)
            vec.y = (vec.y * 0.7) + ((new_y / success_weight) * 0.3)
            vec.z = (vec.z * 0.7) + ((new_z / success_weight) * 0.3)
            
            concept.confidence = min(concept.confidence + 0.05, 1.0)
            concept.evolution_stage += 1
            print(f"✨ Aesthetic Concept '{concept.name}' evolved. (Conf {concept.confidence:.2f})")

    def _evolve_logic(self, concept: ConceptScore, memories: List[Any]):
        """Evolve Logic Rule (New)"""
        # Logic evolution: Check consistency (True/False)
        true_count = 0
        total = 0
        
        for mem in memories:
            # In Logic, Sound is 'is_correct' (True/False)
            is_correct = mem.sound.get("is_correct", False)
            if is_correct:
                true_count += 1
            total += 1
            
        consistency = true_count / total if total > 0 else 0.0
        
        # Update Hypotheses
        concept.value["consistency"] = consistency
        if consistency > 0.9:
            concept.confidence = min(concept.confidence + 0.1, 1.0)
            concept.value["rule"] = "proven_true"
        else:
            concept.confidence = max(concept.confidence - 0.1, 0.0)
            
        concept.evolution_stage += 1
        print(f"📐 Logic Concept '{concept.name}' evolved. Consistency: {consistency:.2f}")

    def save_concepts(self):
        """Persist concepts to disk"""
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: concept.to_dict() for name, concept in self.concepts.items()}
        with open(self.persistence_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load_concepts(self):
        """Load concepts from disk"""
        if not self.persistence_path.exists():
            return
            
        try:
            with open(self.persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for name, item in data.items():
                    self.concepts[name] = ConceptScore.from_dict(item)
            # print(f"🧠 Loaded {len(self.concepts)} concepts from memory.")
        except Exception as e:
            print(f"⚠️ Failed to load concepts: {e}")

# 싱글톤
_formation_instance: Optional[ConceptFormation] = None

def get_concept_formation() -> ConceptFormation:
    global _formation_instance
    if _formation_instance is None:
        _formation_instance = ConceptFormation()
    return _formation_instance
