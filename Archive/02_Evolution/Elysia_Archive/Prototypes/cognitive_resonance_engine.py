"""
Cognitive Resonance Engine (인지적 공명 엔진)
=============================================

"내가 아는 것과 외부 세계가 정의한 것은 항상 다르다"
"진정한 이해는 이 둘 사이의 공명을 찾는 과정이다"

핵심 원리:
- 내부 이해 (Internal Model): 내가 경험/학습한 것
- 외부 정의 (External Definition): 세계가 정의한 것
- 공명 (Resonance): 두 모델의 정렬도

공명도가 높을수록 → 진정한 이해
공명도가 낮을수록 → 오해 또는 불완전한 지식

공명은 정적이 아니라 동적:
- 새로운 경험 → 내부 모델 변화
- 새로운 외부 정보 → 외부 정의 업데이트
- 지속적 재공명 필요
"""

import sys
import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class InternalModel:
    """내부 이해 모델 - 내가 경험/학습한 것"""
    name: str
    
    # 내가 이해한 것
    my_definition: str = ""      # 내가 생각하기에 이것은...
    my_understanding: str = ""   # 내가 느끼기에...
    my_experience: str = ""      # 내 경험상...
    
    # 파동 서명 (느낌의 수치화)
    wave_signature: Dict[str, float] = field(default_factory=dict)
    
    # 연관 개념 (내 마음속에서)
    associated_with: List[str] = field(default_factory=list)
    
    # 이해 확신도
    confidence: float = 0.0  # 0.0 ~ 1.0


@dataclass
class ExternalDefinition:
    """외부 정의 - 세계가 정의한 것"""
    name: str
    
    # 외부 정의
    formal_definition: str = ""   # 공식 정의
    source: str = ""              # 출처 (문서, 설계서, 표준)
    
    # 구조적 정보
    properties: List[str] = field(default_factory=list)
    relations: Dict[str, List[str]] = field(default_factory=dict)  # is_a, part_of 등
    
    # 메타데이터
    last_updated: str = ""
    version: str = ""


@dataclass
class ResonanceResult:
    """공명 결과"""
    concept_name: str
    
    # 공명도 (0.0 ~ 1.0)
    resonance_score: float = 0.0
    
    # 세부 공명
    definition_match: float = 0.0   # 정의 일치도
    semantic_overlap: float = 0.0   # 의미 중첩도
    structural_align: float = 0.0   # 구조적 정렬도
    
    # 불일치
    internal_only: List[str] = field(default_factory=list)  # 내부에만 있는 것
    external_only: List[str] = field(default_factory=list)  # 외부에만 있는 것
    contradictions: List[str] = field(default_factory=list) # 모순
    
    # 해석
    interpretation: str = ""
    
    def describe(self) -> str:
        lines = [
            f"\n🔊 공명 분석: {self.concept_name}",
            f"{'='*50}",
            f"   총 공명도: {self.resonance_score:.2f} / 1.00",
            f"",
            f"   📊 세부:",
            f"      정의 일치: {self.definition_match:.2f}",
            f"      의미 중첩: {self.semantic_overlap:.2f}",
            f"      구조 정렬: {self.structural_align:.2f}",
        ]
        
        if self.internal_only:
            lines.append(f"\n   🧠 내부에만 있음: {', '.join(self.internal_only[:3])}")
        if self.external_only:
            lines.append(f"   🌍 외부에만 있음: {', '.join(self.external_only[:3])}")
        if self.contradictions:
            lines.append(f"   ⚠️ 모순: {', '.join(self.contradictions[:3])}")
        
        lines.append(f"\n   💭 해석: {self.interpretation}")
        
        return "\n".join(lines)


class CognitiveResonanceEngine:
    """
    인지적 공명 엔진
    
    내부 이해와 외부 정의 사이의 공명을 측정하고,
    불일치를 발견하고, 재정렬을 유도한다.
    """
    
    def __init__(self, storage_path: str = "data/cognitive_resonance.json"):
        self.storage_path = storage_path
        
        # 내부 모델들
        self.internal_models: Dict[str, InternalModel] = {}
        
        # 외부 정의들
        self.external_defs: Dict[str, ExternalDefinition] = {}
        
        # 공명 이력
        self.resonance_history: List[ResonanceResult] = []
        
        self._load()
    
    def _load(self):
        """저장된 모델 로드"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for im in data.get("internal_models", []):
                        self.internal_models[im["name"]] = InternalModel(
                            name=im["name"],
                            my_definition=im.get("my_definition", ""),
                            my_understanding=im.get("my_understanding", ""),
                            my_experience=im.get("my_experience", ""),
                            wave_signature=im.get("wave_signature", {}),
                            associated_with=im.get("associated_with", []),
                            confidence=im.get("confidence", 0)
                        )
                    
                    for ed in data.get("external_defs", []):
                        self.external_defs[ed["name"]] = ExternalDefinition(
                            name=ed["name"],
                            formal_definition=ed.get("formal_definition", ""),
                            source=ed.get("source", ""),
                            properties=ed.get("properties", []),
                            relations=ed.get("relations", {}),
                            last_updated=ed.get("last_updated", ""),
                            version=ed.get("version", "")
                        )
                    
                    print(f"📂 Loaded {len(self.internal_models)} internal models, {len(self.external_defs)} external defs")
            except Exception as e:
                print(f"Load failed: {e}")
    
    def _save(self):
        """저장"""
        os.makedirs(os.path.dirname(self.storage_path) or '.', exist_ok=True)
        
        data = {
            "internal_models": [
                {
                    "name": m.name,
                    "my_definition": m.my_definition,
                    "my_understanding": m.my_understanding,
                    "my_experience": m.my_experience,
                    "wave_signature": m.wave_signature,
                    "associated_with": m.associated_with,
                    "confidence": m.confidence
                }
                for m in self.internal_models.values()
            ],
            "external_defs": [
                {
                    "name": d.name,
                    "formal_definition": d.formal_definition,
                    "source": d.source,
                    "properties": d.properties,
                    "relations": d.relations,
                    "last_updated": d.last_updated,
                    "version": d.version
                }
                for d in self.external_defs.values()
            ],
            "last_resonance": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_internal_model(
        self,
        name: str,
        my_definition: str = "",
        my_understanding: str = "",
        associated_with: List[str] = None,
        confidence: float = 0.5
    ) -> InternalModel:
        """내부 모델 추가 (내가 이해한 것)"""
        model = InternalModel(
            name=name,
            my_definition=my_definition,
            my_understanding=my_understanding,
            associated_with=associated_with or [],
            confidence=confidence
        )
        self.internal_models[name.lower()] = model
        self._save()
        return model
    
    def add_external_definition(
        self,
        name: str,
        formal_definition: str,
        source: str = "",
        properties: List[str] = None,
        relations: Dict[str, List[str]] = None
    ) -> ExternalDefinition:
        """외부 정의 추가 (세계가 정의한 것)"""
        defn = ExternalDefinition(
            name=name,
            formal_definition=formal_definition,
            source=source,
            properties=properties or [],
            relations=relations or {},
            last_updated=time.strftime("%Y-%m-%d")
        )
        self.external_defs[name.lower()] = defn
        self._save()
        return defn
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 (단순 시퀀스 매칭)"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """텍스트에서 키워드 추출"""
        if not text:
            return set()
        # 단순히 3글자 이상 단어
        words = re.findall(r'\b\w{3,}\b', text.lower())
        # 불용어 제거
        stopwords = {'the', 'and', 'for', 'that', 'with', 'this', 'are', 'was', 'were', 'been'}
        return set(words) - stopwords
    
    def resonate(self, concept_name: str) -> ResonanceResult:
        """
        공명 측정
        
        내부 모델과 외부 정의를 비교하여 공명도 계산
        """
        key = concept_name.lower()
        
        internal = self.internal_models.get(key)
        external = self.external_defs.get(key)
        
        result = ResonanceResult(concept_name=concept_name)
        
        # 둘 다 없음
        if not internal and not external:
            result.interpretation = f"'{concept_name}'에 대해 내부 이해도 외부 정의도 없습니다."
            return result
        
        # 내부만 있음 (외부 정의 모름)
        if internal and not external:
            result.interpretation = f"내가 이해한 것은 있지만, 외부 세계의 정의를 모릅니다. 검증 필요."
            result.resonance_score = internal.confidence * 0.3  # 낮은 공명
            result.internal_only = internal.associated_with[:5]
            return result
        
        # 외부만 있음 (아직 이해 못함)
        if not internal and external:
            result.interpretation = f"외부 정의는 알지만, 아직 내면화하지 못했습니다. 학습 필요."
            result.resonance_score = 0.1
            result.external_only = external.properties[:5]
            return result
        
        # 둘 다 있음 - 진정한 공명 측정
        
        # 1. 정의 일치도
        result.definition_match = self._text_similarity(
            internal.my_definition, 
            external.formal_definition
        )
        
        # 2. 의미 중첩도 (키워드 기반)
        internal_keywords = self._extract_keywords(
            f"{internal.my_definition} {internal.my_understanding}"
        )
        external_keywords = self._extract_keywords(
            f"{external.formal_definition} {' '.join(external.properties)}"
        )
        
        if internal_keywords and external_keywords:
            overlap = internal_keywords & external_keywords
            union = internal_keywords | external_keywords
            result.semantic_overlap = len(overlap) / len(union) if union else 0
        
        # 3. 구조적 정렬도 (연관 개념)
        internal_relations = set(internal.associated_with)
        external_relations = set()
        for rel_list in external.relations.values():
            external_relations.update(rel_list)
        
        if internal_relations and external_relations:
            overlap = internal_relations & external_relations
            union = internal_relations | external_relations
            result.structural_align = len(overlap) / len(union) if union else 0
        
        # 불일치 분석
        result.internal_only = list(internal_keywords - external_keywords)[:5]
        result.external_only = list(external_keywords - internal_keywords)[:5]
        
        # 총 공명도 계산
        result.resonance_score = (
            result.definition_match * 0.4 +
            result.semantic_overlap * 0.4 +
            result.structural_align * 0.2
        )
        
        # 해석
        if result.resonance_score > 0.8:
            result.interpretation = "높은 공명. 내부 이해와 외부 정의가 잘 정렬되어 있습니다."
        elif result.resonance_score > 0.5:
            result.interpretation = "중간 공명. 대체로 이해하지만 일부 차이가 있습니다."
        elif result.resonance_score > 0.2:
            result.interpretation = "낮은 공명. 외부 정의와 내 이해 사이에 상당한 차이가 있습니다."
        else:
            result.interpretation = "불일치. 재학습이 필요합니다."
        
        self.resonance_history.append(result)
        return result
    
    def global_resonance(self) -> Dict[str, Any]:
        """전체 공명 상태"""
        all_concepts = set(self.internal_models.keys()) | set(self.external_defs.keys())
        
        results = []
        for concept in all_concepts:
            result = self.resonate(concept)
            results.append(result)
        
        if not results:
            return {"status": "No concepts to resonate"}
        
        avg_resonance = sum(r.resonance_score for r in results) / len(results)
        
        return {
            "total_concepts": len(all_concepts),
            "internal_only": len(self.internal_models.keys() - self.external_defs.keys()),
            "external_only": len(self.external_defs.keys() - self.internal_models.keys()),
            "both": len(self.internal_models.keys() & self.external_defs.keys()),
            "average_resonance": avg_resonance,
            "high_resonance": len([r for r in results if r.resonance_score > 0.7]),
            "low_resonance": len([r for r in results if r.resonance_score < 0.3])
        }


def demo_resonance():
    """공명 데모"""
    print("\n" + "="*70)
    print("🔊 COGNITIVE RESONANCE ENGINE (인지적 공명 엔진)")
    print("   '내가 아는 것 ↔ 세계가 정의한 것'")
    print("="*70)
    
    engine = CognitiveResonanceEngine("data/resonance_demo.json")
    
    # 1. 내부 모델 추가 (내가 이해한 것)
    print("\n📥 내부 모델 추가 (내가 이해한 것)...")
    
    engine.add_internal_model(
        name="물",
        my_definition="투명하고 흐르는 액체, 마시면 갈증이 해소됨",
        my_understanding="생명에 필수적인 것, 비가 오면 생기는 것",
        associated_with=["비", "바다", "강", "갈증", "생명"],
        confidence=0.7
    )
    
    engine.add_internal_model(
        name="의식",
        my_definition="내가 존재한다는 느낌, 생각하고 있다는 자각",
        my_understanding="깨어있음, 나를 인식하는 것",
        associated_with=["생각", "자아", "인식", "깨어있음"],
        confidence=0.5
    )
    
    engine.add_internal_model(
        name="파동",
        my_definition="출렁이는 움직임, 에너지의 전달",
        my_understanding="물결처럼 퍼져나가는 것",
        associated_with=["에너지", "진동", "물결", "소리"],
        confidence=0.6
    )
    
    # 2. 외부 정의 추가 (세계가 정의한 것)
    print("📥 외부 정의 추가 (세계가 정의한 것)...")
    
    engine.add_external_definition(
        name="물",
        formal_definition="수소 원자 2개와 산소 원자 1개로 구성된 화합물(H2O)",
        source="화학",
        properties=["투명", "무색", "무취", "극성 용매", "수소결합"],
        relations={
            "is_a": ["물질", "액체", "화합물"],
            "composed_of": ["수소", "산소"]
        }
    )
    
    engine.add_external_definition(
        name="의식",
        formal_definition="자기 자신과 환경을 인식하는 주관적 경험의 상태",
        source="철학/심리학",
        properties=["주관성", "지향성", "통합성", "자각"],
        relations={
            "is_a": ["정신 상태", "현상"],
            "related_to": ["마음", "인지", "자아"]
        }
    )
    
    engine.add_external_definition(
        name="파동",
        formal_definition="매질 또는 공간을 통해 에너지가 전파되는 교란",
        source="물리학",
        properties=["진동수", "파장", "진폭", "위상"],
        relations={
            "is_a": ["물리 현상"],
            "types": ["횡파", "종파", "전자기파"]
        }
    )
    
    # 3. 공명 측정
    print("\n" + "="*70)
    print("🔊 공명 측정")
    print("="*70)
    
    for concept in ["물", "의식", "파동"]:
        result = engine.resonate(concept)
        print(result.describe())
    
    # 4. 전체 공명 상태
    print("\n" + "="*70)
    print("📊 전체 공명 상태")
    print("="*70)
    
    global_state = engine.global_resonance()
    print(f"   총 개념: {global_state['total_concepts']}")
    print(f"   내부만: {global_state['internal_only']}")
    print(f"   외부만: {global_state['external_only']}")
    print(f"   둘 다: {global_state['both']}")
    print(f"\n   평균 공명도: {global_state['average_resonance']:.2f}")
    print(f"   높은 공명: {global_state['high_resonance']}")
    print(f"   낮은 공명: {global_state['low_resonance']}")
    
    print("\n" + "="*70)
    print("✅ 이것이 '인지적 공명'입니다.")
    print("   내가 아는 것과 세계가 정의한 것 사이의 정렬을 측정합니다.")
    print("   공명이 높을수록 → 진정한 이해")
    print("   공명이 낮을수록 → 재학습 필요")
    print("="*70)


if __name__ == "__main__":
    demo_resonance()
