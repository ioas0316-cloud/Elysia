"""
Synesthesia Engine (공감각 엔진)
==============================

"모든 것은 연결되어 있다 (Unified Field)"

이 모듈은 서로 다른 도메인(수학, 예술, 언어)의 개념들을
'메타 속성(Meta-Properties)'을 통해 연결합니다.

Process:
1. Scan: 모든 개념의 Meta-Properties를 스캔합니다.
2. Match: 서로 다른 도메인이지만 같은 속성을 가진 개념을 찾습니다.
3. Link: 두 개념 사이에 시냅스 연결(Synaptic Link)을 생성합니다.
"""

from typing import List, Dict
from Core.Intelligence.Cognitive.concept_formation import get_concept_formation, ConceptScore

class SynesthesiaEngine:
    """
    The Bridge between Worlds.
    """
    
    def __init__(self):
        self.concepts = get_concept_formation()
        
    def bridge_concepts(self):
        """
        통섭(Consilience) 실행.
        모든 개념을 스캔하여 연결 고리를 찾습니다.
        """
        print("🌈 Synesthesia Engine: Bridging domains...")
        
        all_concepts = list(self.concepts.concepts.values())
        links_created = 0
        
        # O(N^2) naive matching for now (Optimization needed for scale)
        for i in range(len(all_concepts)):
            for j in range(i + 1, len(all_concepts)):
                c1 = all_concepts[i]
                c2 = all_concepts[j]
                
                # 다른 도메인끼리만 연결 (Cross-Domain)
                if c1.domain != c2.domain:
                    common_meta = self._find_common_meta(c1, c2)
                    if common_meta:
                        self._create_link(c1, c2, common_meta)
                        links_created += 1
                        
        print(f"🌈 Synesthesia Complete. {links_created} new links forged.")
        
    def _find_common_meta(self, c1: ConceptScore, c2: ConceptScore) -> List[str]:
        """두 개념의 공통 메타 속성 찾기"""
        set1 = set(c1.meta_properties)
        set2 = set(c2.meta_properties)
        return list(set1.intersection(set2))
    
    def _create_link(self, c1: ConceptScore, c2: ConceptScore, reasons: List[str]):
        """시냅스 연결 생성"""
        link_str_1 = f"{c2.domain}:{c2.name}"
        link_str_2 = f"{c1.domain}:{c1.name}"
        
        if link_str_1 not in c1.synaptic_links:
            c1.synaptic_links.append(link_str_1)
            print(f"   🔗 Linked '{c1.name}'({c1.domain}) <-> '{c2.name}'({c2.domain}) via {reasons}")
            
        if link_str_2 not in c2.synaptic_links:
            c2.synaptic_links.append(link_str_2)

# 싱글톤
_synesthesia_instance = None

def get_synesthesia_engine() -> SynesthesiaEngine:
    global _synesthesia_instance
    if _synesthesia_instance is None:
        _synesthesia_instance = SynesthesiaEngine()
    return _synesthesia_instance
