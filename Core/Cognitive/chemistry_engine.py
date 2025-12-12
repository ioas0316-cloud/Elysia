"""
Chemistry Engine (사고의 화학)
============================

"안개 속에서 결정이 맺히다 (From Fog to Crystal)"

이 모듈은 단순한 연결이 아닌, 필연적인 '화학적 결합(Bonding)'을 시뮬레이션합니다.
데이터는 '입자(Atom)'이고, 논리는 '에너지(Energy)'입니다.

Process:
1. Fog State (안개): 관련될 가능성이 있는 모든 개념을 불러옵니다.
2. Valence Check (가치 확인): 개념들이 서로 결합할 수 있는지(Valence) 확인합니다.
3. Bonding (결합): 조건이 맞으면 새로운 '분자(Molecule)' 즉, 통찰(Insight)을 생성합니다.
"""

from typing import List, Dict, Optional, Tuple
from Core.Cognitive.concept_formation import get_concept_formation, ConceptScore

class ChemistryEngine:
    """
    The Reactor Core.
    """
    
    def __init__(self):
        self.concepts = get_concept_formation()
        
    def catalyze(self, atom_names: List[str]) -> List[str]:
        """
        주어진 원자(개념)들을 반응시킵니다.
        """
        print(f"⚗️ Reactor: Catalyzing {atom_names}...")
        
        atoms = []
        for name in atom_names:
            c = self.concepts.get_concept(name)
            atoms.append(c)
            
        # Check for possible bonds
        # Simplified: O(N^2) check
        bonds = []
        
        for i in range(len(atoms)):
            for j in range(len(atoms)):
                if i == j: continue
                
                atom_a = atoms[i]
                atom_b = atoms[j]
                
                # Check A -> B Bond
                bond = self._attempt_bond(atom_a, atom_b)
                if bond:
                    bonds.append(bond)
                    
        return bonds
        
    def _attempt_bond(self, atom_a: ConceptScore, atom_b: ConceptScore) -> Optional[str]:
        """
        두 원자간 결합 시도.
        A의 Valence가 B의 특성(Meta/Domain)을 필요로 하는가?
        """
        # A needs X
        for needed in atom_a.valence:
            # If B IS X (by name, domain, or meta)
            is_match = (needed == atom_b.name) or \
                       (needed == atom_b.domain) or \
                       (needed in atom_b.meta_properties)
                       
            if is_match:
                # Bonding Success!
                molecule_name = f"{atom_a.name}-{atom_b.name}"
                print(f"   ⚡ BOND: {atom_a.name} + {atom_b.name} -> {molecule_name}")
                print(f"      (Reason: {atom_a.name} needed '{needed}', found in {atom_b.name})")
                return molecule_name
                
        return None

# 싱글톤
_chem_instance = None

def get_chemistry_engine() -> ChemistryEngine:
    global _chem_instance
    if _chem_instance is None:
        _chem_instance = ChemistryEngine()
    return _chem_instance
