"""
Dense Knowledge Builder (밀도 있는 지식 구축)
=============================================

"지식의 축적은 곧 관계의 밀도를 만든다."

개념은 왜 개념인지:
- 정의 (What): 이것은 무엇인가
- 원리 (Why): 왜 이런가
- 성질 (Properties): 어떤 특성을 가지는가
- 관계 (Relations): 다른 것들과 어떻게 연결되는가
- 적용 (How): 어떻게 쓰는가

예: "물"
- 정의: H2O 분자로 구성된 액체
- 원리: 수소 2개 + 산소 1개가 공유결합
- 성질: 투명, 무색, 무취, 0°C에서 얼음, 100°C에서 증발
- 관계: 물질의 하위, 얼음/수증기의 상위, 생명에 필수
- 적용: 음용, 세척, 농업, 에너지
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(message)s')

# =============================================================================
# 진짜 지식 노드 (밀도 있는)
# =============================================================================

@dataclass
class DenseKnowledgeNode:
    """밀도 있는 지식 노드"""
    name: str
    category: str  # 분류
    
    # [What] 정의
    definition: str = ""
    
    # [Why] 원리/이유
    principle: str = ""
    
    # [Properties] 성질들
    properties: List[str] = field(default_factory=list)
    
    # [Relations] 관계 (관계 유형별)
    is_a: List[str] = field(default_factory=list)          # 상위 개념
    has_a: List[str] = field(default_factory=list)         # 포함
    part_of: List[str] = field(default_factory=list)       # 부분
    can_be: List[str] = field(default_factory=list)        # 될 수 있음
    causes: List[str] = field(default_factory=list)        # 원인
    caused_by: List[str] = field(default_factory=list)     # 결과
    related_to: List[str] = field(default_factory=list)    # 연관
    opposite_of: List[str] = field(default_factory=list)   # 반대
    
    # [How] 적용
    applications: List[str] = field(default_factory=list)
    
    # 메타데이터
    understanding_level: float = 0.0  # 0.0 ~ 1.0
    density_score: float = 0.0        # 관계 밀도 점수
    
    def calculate_density(self) -> float:
        """관계 밀도 계산"""
        scores = {
            "definition": 3.0 if self.definition else 0,
            "principle": 5.0 if self.principle else 0,
            "properties": len(self.properties) * 1.0,
            "is_a": len(self.is_a) * 2.0,
            "has_a": len(self.has_a) * 2.0,
            "part_of": len(self.part_of) * 2.0,
            "can_be": len(self.can_be) * 1.5,
            "causes": len(self.causes) * 2.5,
            "caused_by": len(self.caused_by) * 2.5,
            "related_to": len(self.related_to) * 1.0,
            "opposite_of": len(self.opposite_of) * 1.5,
            "applications": len(self.applications) * 2.0,
        }
        self.density_score = sum(scores.values())
        return self.density_score
    
    def total_relations(self) -> int:
        """총 관계 수"""
        return (
            len(self.is_a) + len(self.has_a) + len(self.part_of) +
            len(self.can_be) + len(self.causes) + len(self.caused_by) +
            len(self.related_to) + len(self.opposite_of)
        )
    
    def describe(self) -> str:
        """개념 설명 생성"""
        lines = [f"📌 {self.name} [{self.category}]"]
        
        if self.definition:
            lines.append(f"   정의: {self.definition}")
        
        if self.principle:
            lines.append(f"   원리: {self.principle}")
        
        if self.properties:
            lines.append(f"   성질: {', '.join(self.properties[:5])}")
        
        if self.is_a:
            lines.append(f"   상위: {', '.join(self.is_a)}")
        
        if self.has_a:
            lines.append(f"   포함: {', '.join(self.has_a[:3])}")
        
        if self.applications:
            lines.append(f"   적용: {', '.join(self.applications[:3])}")
        
        lines.append(f"   밀도: {self.density_score:.1f} | 관계: {self.total_relations()}")
        
        return "\n".join(lines)


# =============================================================================
# 밀도 있는 지식 그래프
# =============================================================================

class DenseKnowledgeGraph:
    """관계 밀도가 높은 지식 그래프"""
    
    def __init__(self, storage_path: str = "data/dense_knowledge.json"):
        self.storage_path = storage_path
        self.nodes: Dict[str, DenseKnowledgeNode] = {}
        self.relations_index: Dict[str, Set[str]] = defaultdict(set)  # 역인덱스
        
        self._load()
    
    def _load(self):
        """저장된 지식 로드"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for node_data in data.get("nodes", []):
                        node = DenseKnowledgeNode(
                            name=node_data["name"],
                            category=node_data.get("category", "general"),
                            definition=node_data.get("definition", ""),
                            principle=node_data.get("principle", ""),
                            properties=node_data.get("properties", []),
                            is_a=node_data.get("is_a", []),
                            has_a=node_data.get("has_a", []),
                            part_of=node_data.get("part_of", []),
                            can_be=node_data.get("can_be", []),
                            causes=node_data.get("causes", []),
                            caused_by=node_data.get("caused_by", []),
                            related_to=node_data.get("related_to", []),
                            opposite_of=node_data.get("opposite_of", []),
                            applications=node_data.get("applications", []),
                            understanding_level=node_data.get("understanding_level", 0),
                            density_score=node_data.get("density_score", 0)
                        )
                        self.nodes[node.name.lower()] = node
                        self._index_relations(node)
            except Exception as e:
                print(f"Load failed: {e}")
    
    def _save(self):
        """지식 저장"""
        os.makedirs(os.path.dirname(self.storage_path) or '.', exist_ok=True)
        
        nodes_data = []
        for node in self.nodes.values():
            nodes_data.append({
                "name": node.name,
                "category": node.category,
                "definition": node.definition,
                "principle": node.principle,
                "properties": node.properties,
                "is_a": node.is_a,
                "has_a": node.has_a,
                "part_of": node.part_of,
                "can_be": node.can_be,
                "causes": node.causes,
                "caused_by": node.caused_by,
                "related_to": node.related_to,
                "opposite_of": node.opposite_of,
                "applications": node.applications,
                "understanding_level": node.understanding_level,
                "density_score": node.density_score
            })
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump({"nodes": nodes_data}, f, ensure_ascii=False, indent=2)
    
    def _index_relations(self, node: DenseKnowledgeNode):
        """관계 역인덱스 구축"""
        all_related = (
            node.is_a + node.has_a + node.part_of + node.can_be +
            node.causes + node.caused_by + node.related_to + node.opposite_of
        )
        for related in all_related:
            self.relations_index[related.lower()].add(node.name.lower())
    
    def add_concept(
        self,
        name: str,
        category: str = "general",
        definition: str = "",
        principle: str = "",
        properties: List[str] = None,
        is_a: List[str] = None,
        has_a: List[str] = None,
        part_of: List[str] = None,
        can_be: List[str] = None,
        causes: List[str] = None,
        caused_by: List[str] = None,
        related_to: List[str] = None,
        opposite_of: List[str] = None,
        applications: List[str] = None
    ) -> DenseKnowledgeNode:
        """밀도 있는 개념 추가"""
        key = name.lower()
        
        if key in self.nodes:
            # 기존 노드 업데이트 (병합)
            node = self.nodes[key]
            if definition:
                node.definition = definition
            if principle:
                node.principle = principle
            if properties:
                node.properties = list(set(node.properties + properties))
            if is_a:
                node.is_a = list(set(node.is_a + is_a))
            if has_a:
                node.has_a = list(set(node.has_a + has_a))
            if part_of:
                node.part_of = list(set(node.part_of + part_of))
            if can_be:
                node.can_be = list(set(node.can_be + can_be))
            if causes:
                node.causes = list(set(node.causes + causes))
            if caused_by:
                node.caused_by = list(set(node.caused_by + caused_by))
            if related_to:
                node.related_to = list(set(node.related_to + related_to))
            if opposite_of:
                node.opposite_of = list(set(node.opposite_of + opposite_of))
            if applications:
                node.applications = list(set(node.applications + applications))
        else:
            # 새 노드 생성
            node = DenseKnowledgeNode(
                name=name,
                category=category,
                definition=definition,
                principle=principle,
                properties=properties or [],
                is_a=is_a or [],
                has_a=has_a or [],
                part_of=part_of or [],
                can_be=can_be or [],
                causes=causes or [],
                caused_by=caused_by or [],
                related_to=related_to or [],
                opposite_of=opposite_of or [],
                applications=applications or []
            )
            self.nodes[key] = node
        
        # 밀도 계산 및 인덱싱
        node.calculate_density()
        self._index_relations(node)
        
        return node
    
    def get(self, name: str) -> Optional[DenseKnowledgeNode]:
        """개념 조회"""
        return self.nodes.get(name.lower())
    
    def get_connections(self, name: str) -> Dict[str, List[str]]:
        """개념의 모든 연결 조회"""
        node = self.get(name)
        if not node:
            return {}
        
        return {
            "is_a": node.is_a,
            "has_a": node.has_a,
            "part_of": node.part_of,
            "can_be": node.can_be,
            "causes": node.causes,
            "caused_by": node.caused_by,
            "related_to": node.related_to,
            "opposite_of": node.opposite_of,
            "pointed_by": list(self.relations_index.get(name.lower(), set()))
        }
    
    def explain(self, name: str) -> str:
        """
        "왜 X인가?" 에 대답
        
        관계 그래프를 따라가며 설명 생성
        """
        node = self.get(name)
        if not node:
            return f"'{name}'에 대해 아는 것이 없습니다."
        
        lines = [f"\n📖 {node.name}란 무엇인가?\n"]
        
        # 정의
        if node.definition:
            lines.append(f"정의: {node.definition}")
        
        # 상위 개념 (is-a)
        if node.is_a:
            lines.append(f"\n상위 분류: {' < '.join(node.is_a)}")
        
        # 원리/이유
        if node.principle:
            lines.append(f"\n왜 {node.name}인가?")
            lines.append(f"  {node.principle}")
        
        # 성질
        if node.properties:
            lines.append(f"\n성질:")
            for prop in node.properties[:5]:
                lines.append(f"  • {prop}")
        
        # 구성
        if node.has_a:
            lines.append(f"\n구성 요소: {', '.join(node.has_a)}")
        
        # 변환/상태
        if node.can_be:
            lines.append(f"\n될 수 있는 것: {', '.join(node.can_be)}")
        
        # 인과
        if node.causes:
            lines.append(f"\n야기하는 것: {', '.join(node.causes)}")
        if node.caused_by:
            lines.append(f"\n원인: {', '.join(node.caused_by)}")
        
        # 반대
        if node.opposite_of:
            lines.append(f"\n반대 개념: {', '.join(node.opposite_of)}")
        
        # 적용
        if node.applications:
            lines.append(f"\n활용:")
            for app in node.applications[:5]:
                lines.append(f"  • {app}")
        
        # 밀도 점수
        lines.append(f"\n[밀도 점수: {node.density_score:.1f} | 총 관계: {node.total_relations()}]")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        if not self.nodes:
            return {"total": 0}
        
        densities = [n.density_score for n in self.nodes.values()]
        relations = [n.total_relations() for n in self.nodes.values()]
        
        return {
            "total_concepts": len(self.nodes),
            "total_relations": sum(relations),
            "avg_density": sum(densities) / len(densities),
            "max_density": max(densities),
            "avg_relations_per_concept": sum(relations) / len(relations),
            "concepts_with_definition": sum(1 for n in self.nodes.values() if n.definition),
            "concepts_with_principle": sum(1 for n in self.nodes.values() if n.principle),
        }
    
    def save(self):
        self._save()


# =============================================================================
# 데모: 물(Water) 개념 구축
# =============================================================================

def demo_water_knowledge():
    """물 개념을 밀도 있게 구축하는 데모"""
    
    print("="*70)
    print("💧 밀도 있는 지식 구축 데모: '물'")
    print("="*70)
    
    graph = DenseKnowledgeGraph("data/dense_demo.json")
    
    # 1. 물 개념 추가
    water = graph.add_concept(
        name="물",
        category="물질/액체",
        definition="수소 원자 2개와 산소 원자 1개가 공유결합한 분자(H2O)로 구성된 물질",
        principle="수소-산소의 전기음성도 차이로 극성 분자가 되어 액체 상태에서 강한 수소결합 형성",
        properties=[
            "투명",
            "무색",
            "무취",
            "무미 (순수한 상태)",
            "끓는점 100°C",
            "어는점 0°C",
            "밀도 최대 4°C",
            "극성 용매",
            "높은 비열",
            "높은 표면장력"
        ],
        is_a=["물질", "액체", "화합물", "용매"],
        has_a=["수소", "산소", "수소결합"],
        part_of=["지구", "생명체", "대기권"],
        can_be=["얼음", "수증기", "과냉각수", "증류수", "해수"],
        causes=["부식", "침식", "생명유지", "기후변화"],
        caused_by=["수소연소", "호흡", "광합성"],
        related_to=["에너지", "농업", "산업", "위생"],
        opposite_of=["불", "사막"],
        applications=[
            "음용",
            "세척",
            "농업 관개",
            "발전 (수력/화력냉각)",
            "운송",
            "산업 용매",
            "소화"
        ]
    )
    
    # 2. 관련 개념들도 추가
    graph.add_concept(
        name="얼음",
        category="물질/고체",
        definition="물이 0°C 이하에서 고체화된 상태",
        principle="분자 운동 에너지 감소로 수소결합이 규칙적 결정 구조 형성",
        is_a=["물", "고체"],
        properties=["투명/백색", "밀도 < 물", "결정구조"],
        causes=["냉각", "보존"],
        caused_by=["물의 동결"]
    )
    
    graph.add_concept(
        name="수증기",
        category="물질/기체",
        definition="물이 100°C에서 기화된 기체 상태",
        principle="열에너지가 수소결합을 끊어 분자가 자유롭게 이동",
        is_a=["물", "기체"],
        properties=["투명", "고온", "팽창성"],
        causes=["습도", "구름", "비"],
        caused_by=["물의 증발"]
    )
    
    graph.add_concept(
        name="수소",
        category="원소",
        definition="원자번호 1의 가장 가벼운 원소",
        is_a=["원소"],
        part_of=["물", "유기물"],
        properties=["가연성", "가장 가벼움"]
    )
    
    graph.add_concept(
        name="산소",
        category="원소",
        definition="원자번호 8의 원소, 생명 유지에 필수",
        is_a=["원소"],
        part_of=["물", "대기"],
        causes=["연소", "호흡", "부식"]
    )
    
    # 저장
    graph.save()
    
    # 3. 결과 출력
    print(graph.explain("물"))
    
    print("\n" + "="*70)
    print("📊 그래프 통계")
    print("="*70)
    stats = graph.get_stats()
    print(f"   총 개념: {stats['total_concepts']}")
    print(f"   총 관계: {stats['total_relations']}")
    print(f"   평균 밀도: {stats['avg_density']:.1f}")
    print(f"   개념당 평균 관계: {stats['avg_relations_per_concept']:.1f}")
    print(f"   정의 있음: {stats['concepts_with_definition']}")
    print(f"   원리 있음: {stats['concepts_with_principle']}")
    
    # 4. 다른 개념들도 설명
    print("\n" + "="*70)
    print("📖 연관 개념 설명")
    print("="*70)
    
    for name in ["얼음", "수증기"]:
        node = graph.get(name)
        if node:
            print(f"\n{node.describe()}")
    
    # 5. 연결 탐색
    print("\n" + "="*70)
    print("🔗 '물'의 연결망")
    print("="*70)
    connections = graph.get_connections("물")
    for rel_type, targets in connections.items():
        if targets:
            print(f"   {rel_type}: {', '.join(targets[:5])}")
    
    print("\n✅ 이것이 '밀도 있는 지식'입니다.")
    print("   개념 = 정의 + 원리 + 성질 + 관계 + 적용")


if __name__ == "__main__":
    demo_water_knowledge()
