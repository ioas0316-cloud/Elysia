"""
[Project Elysia] Principle Extractor
====================================
Phase 3: 점에서 섭리로

"물 순환 체인에서 'energy_drives_cycles' 공리를 발견한다"

이 모듈은 연결 체인에서 순환/패턴을 인식하여 원리(Axiom)로 승화한다.
이것이 "왜?"에서 시작해 "모든 것이 연결되어 있구나"에 도달하는 과정.
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter
import time
import hashlib

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L5_Mental.Reasoning.connection_explorer import ConnectionChain


@dataclass
class Axiom:
    """
    공리 (발견된 원리)
    
    "모든 것이 순환한다" 같은 보편적 진리
    아이가 "아, 물은 순환하는구나!"라고 깨닫는 순간
    """
    axiom_id: str
    name: str                    # 예: "energy_drives_cycles"
    description: str             # 자연어 설명
    source_chains: List[str]     # 이 공리를 도출한 체인 ID들
    pattern_type: str            # "cycle", "hierarchy", "causation"
    confidence: float            # 0.0 ~ 1.0
    related_nodes: List[str]     # 관련 노드들
    timestamp: float = field(default_factory=time.time)
    applications: int = 0        # 다른 영역에 적용된 횟수
    
    def to_natural_language(self) -> str:
        """자연어 설명으로 변환"""
        if self.pattern_type == "cycle":
            nodes = ", ".join(self.related_nodes[:3])
            return f"{nodes} 등이 순환 구조를 이룬다. {self.description}"
        elif self.pattern_type == "hierarchy":
            return f"계층 구조가 발견되었다: {self.description}"
        else:
            return self.description


@dataclass
class PatternSignature:
    """패턴의 특성"""
    pattern_type: str
    relations: List[str]
    node_count: int
    is_cycle: bool


class PrincipleExtractor:
    """
    원리 추출기
    
    연결 체인에서 반복되는 패턴(순환)을 인식하여
    새로운 Axiom(원리)으로 기록한다.
    
    핵심 원리:
    - 물 순환 발견 → "순환" 패턴 인식
    - 생명 순환도 발견 → 같은 패턴!
    - → "에너지가 변화를 일으킨다" 공리 생성
    """
    
    # 패턴 유형별 키워드
    CYCLE_RELATIONS = {"causes", "leads_to", "creates", "enables", "produces", "flows_to"}
    HIERARCHY_RELATIONS = {"is_a", "part_of", "belongs_to", "contains"}
    
    def __init__(self):
        self.axiom_registry: Dict[str, Axiom] = {}
        self.pattern_signatures: List[PatternSignature] = []
        self.axiom_counter = 0
    
    def extract_principle(self, chains: List[ConnectionChain]) -> List[Axiom]:
        """
        연결 체인들에서 원리 추출
        
        1. 순환 구조가 있으면 → 순환 공리 생성
        2. 유사한 패턴이 반복되면 → 보편 공리 생성
        """
        new_axioms = []
        
        # 순환 체인에서 공리 추출
        cycles = [c for c in chains if c.is_cycle]
        for cycle in cycles:
            axiom = self._extract_from_cycle(cycle)
            if axiom and axiom.axiom_id not in self.axiom_registry:
                self.axiom_registry[axiom.axiom_id] = axiom
                new_axioms.append(axiom)
        
        # 패턴 유사성 분석
        if len(chains) >= 2:
            pattern_axioms = self._find_repeated_patterns(chains)
            for axiom in pattern_axioms:
                if axiom.axiom_id not in self.axiom_registry:
                    self.axiom_registry[axiom.axiom_id] = axiom
                    new_axioms.append(axiom)
        
        return new_axioms
    
    def _extract_from_cycle(self, cycle: ConnectionChain) -> Optional[Axiom]:
        """순환 체인에서 공리 추출"""
        if not cycle.is_cycle or len(cycle) < 2:
            return None
        
        # 관계 유형 분석
        relations = [c.relation for c in cycle.connections]
        nodes = cycle.get_path()
        
        # 순환 공리 생성
        self.axiom_counter += 1
        
        # 핵심 관계 추출
        relation_counts = Counter(relations)
        dominant_relation = relation_counts.most_common(1)[0][0] if relation_counts else "flows"
        
        # 공리 이름 생성
        name = self._generate_axiom_name(cycle, dominant_relation)
        
        return Axiom(
            axiom_id=f"AX_{self.axiom_counter:04d}",
            name=name,
            description=f"{len(nodes)}개 요소가 순환 구조를 형성",
            source_chains=[cycle.chain_id],
            pattern_type="cycle",
            confidence=min(1.0, len(cycle) / 5.0),  # 길수록 높은 확신
            related_nodes=nodes[:10]
        )
    
    def _find_repeated_patterns(self, chains: List[ConnectionChain]) -> List[Axiom]:
        """반복되는 패턴에서 보편 공리 추출"""
        axioms = []
        
        # 패턴 시그니처 추출
        signatures = {}
        for chain in chains:
            sig = self._get_signature(chain)
            sig_key = (sig.pattern_type, tuple(sorted(sig.relations)))
            
            if sig_key not in signatures:
                signatures[sig_key] = []
            signatures[sig_key].append(chain)
        
        # 2번 이상 반복되는 패턴 → 공리
        for sig_key, matching_chains in signatures.items():
            if len(matching_chains) >= 2:
                pattern_type, relations = sig_key
                
                self.axiom_counter += 1
                axiom = Axiom(
                    axiom_id=f"AX_{self.axiom_counter:04d}",
                    name=f"pattern_{pattern_type}_{self.axiom_counter}",
                    description=f"'{pattern_type}' 패턴이 {len(matching_chains)}개 영역에서 반복됨",
                    source_chains=[c.chain_id for c in matching_chains],
                    pattern_type="universal",
                    confidence=min(1.0, len(matching_chains) / 3.0),
                    related_nodes=list(set(
                        node for chain in matching_chains 
                        for node in chain.get_path()[:3]
                    ))
                )
                axioms.append(axiom)
        
        return axioms
    
    def _get_signature(self, chain: ConnectionChain) -> PatternSignature:
        """체인의 패턴 시그니처 추출"""
        relations = [c.relation.lower() for c in chain.connections]
        
        # 패턴 유형 결정
        if chain.is_cycle:
            pattern_type = "cycle"
        elif any(r in self.HIERARCHY_RELATIONS for r in relations):
            pattern_type = "hierarchy"
        else:
            pattern_type = "causation"
        
        return PatternSignature(
            pattern_type=pattern_type,
            relations=relations,
            node_count=len(chain.get_path()),
            is_cycle=chain.is_cycle
        )
    
    def _generate_axiom_name(self, chain: ConnectionChain, dominant_relation: str) -> str:
        """공리 이름 생성"""
        nodes = chain.get_path()
        
        # 노드들의 공통 주제 추출 시도
        if any("water" in n.lower() or "rain" in n.lower() for n in nodes):
            return "water_cycle_principle"
        elif any("life" in n.lower() or "death" in n.lower() for n in nodes):
            return "life_cycle_principle"
        elif any("energy" in n.lower() or "sun" in n.lower() for n in nodes):
            return "energy_transformation_principle"
        else:
            # 해시 기반 이름
            hash_input = "".join(nodes[:3])
            short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
            return f"cycle_principle_{short_hash}"
    
    def apply_axiom(self, axiom_id: str, new_domain: str) -> bool:
        """기존 공리를 새 영역에 적용"""
        if axiom_id in self.axiom_registry:
            axiom = self.axiom_registry[axiom_id]
            axiom.applications += 1
            axiom.related_nodes.append(new_domain)
            return True
        return False
    
    def get_all_axioms(self) -> List[Axiom]:
        """등록된 모든 공리 반환"""
        return list(self.axiom_registry.values())
    
    def get_stats(self) -> Dict:
        """통계"""
        axioms = self.get_all_axioms()
        return {
            "total_axioms": len(axioms),
            "cycle_axioms": sum(1 for a in axioms if a.pattern_type == "cycle"),
            "universal_axioms": sum(1 for a in axioms if a.pattern_type == "universal"),
            "total_applications": sum(a.applications for a in axioms)
        }


# Singleton
_principle_extractor = None

def get_principle_extractor() -> PrincipleExtractor:
    global _principle_extractor
    if _principle_extractor is None:
        _principle_extractor = PrincipleExtractor()
    return _principle_extractor


if __name__ == "__main__":
    print("💡 Testing Principle Extractor...")
    
    from connection_explorer import ConnectionChain, Connection
    
    # 테스트용 순환 체인 생성
    water_cycle = ConnectionChain(
        chain_id="TEST_CYCLE_001",
        connections=[
            Connection("sun", "evaporation", "causes"),
            Connection("evaporation", "cloud", "creates"),
            Connection("cloud", "rain", "produces"),
            Connection("rain", "ocean", "flows_to"),
            Connection("ocean", "evaporation", "enables"),
        ],
        is_cycle=True,
        cycle_start="evaporation"
    )
    
    life_cycle = ConnectionChain(
        chain_id="TEST_CYCLE_002",
        connections=[
            Connection("birth", "growth", "leads_to"),
            Connection("growth", "reproduction", "enables"),
            Connection("reproduction", "death", "followed_by"),
            Connection("death", "birth", "enables"),
        ],
        is_cycle=True,
        cycle_start="birth"
    )
    
    extractor = get_principle_extractor()
    axioms = extractor.extract_principle([water_cycle, life_cycle])
    
    print(f"\n📊 Extracted {len(axioms)} axioms:")
    for axiom in axioms:
        print(f"  [{axiom.pattern_type}] {axiom.name}")
        print(f"    → {axiom.to_natural_language()}")
        print(f"    Confidence: {axiom.confidence:.2f}")
    
    print(f"\n✅ Principle Extractor operational!")
    print(f"   Stats: {extractor.get_stats()}")
