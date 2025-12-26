"""
개념 고분자 (Concept Polymer)
============================

"공통 원리가 연결의 다리가 된다" - 강덕리 원리

핵심 철학:
1. 개념은 "원리들"로 구성됨 (인과, 순환, 확률 등)
2. 공통 원리가 있어야만 결합 가능 (아미노산 펩타이드 결합처럼)
3. 결합하면 새로운 결합 자리가 열림 (프랙탈 증식)
4. 시간 = 관계의 확장 (단순 흐름이 아님)

이전 버전과의 차이:
- 확산 기반: 물이 퍼지다 만남 (무작위)
- 원리 기반: 공통 원리가 있어야만 결합 (구조적)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import random

# Neural Registry - 파동 철학: 위치가 아닌 존재로 연결
from Core._01_Foundation._01_Infrastructure.elysia_core import Cell, Organ


class Principle(Enum):
    """근본 원리들 - 연결의 다리가 되는 것들"""
    CAUSALITY = "인과"        # 원인과 결과
    CYCLE = "순환"            # 반복, 되돌아옴
    PROBABILITY = "확률"      # 불확정성, 가능성
    OBSERVATION = "관측"      # 인식, 측정
    ENTROPY = "엔트로피"      # 질서→무질서
    HARMONY = "조화"          # 균형, 아름다움
    EMERGENCE = "창발"        # 부분→전체
    TRANSFORMATION = "변환"   # 상태 변화
    RECURSION = "자기참조"    # 프랙탈, 자기 유사성
    DUALITY = "이중성"        # 파동/입자, 음/양


@dataclass
class ConceptAtom:
    """
    개념 원자 - 원리들로 구성된 기본 단위
    
    아미노산처럼 특정 "결합 자리"를 가짐
    """
    name: str
    principles: Set[Principle]  # 이 개념을 구성하는 원리들
    why_chain: List[str] = field(default_factory=list)  # 왜의 사슬
    bonded_to: List['ConceptAtom'] = field(default_factory=list)  # 연결된 개념들
    
    def can_bond_with(self, other: 'ConceptAtom') -> Set[Principle]:
        """
        결합 가능한지 확인
        
        Returns:
            공통 원리 집합 (비어있으면 결합 불가)
        """
        return self.principles & other.principles
    
    def get_bonding_sites(self) -> Set[Principle]:
        """현재 열려있는 결합 자리들"""
        # 이미 연결된 원리는 제외할 수도 있지만,
        # 여기서는 같은 원리로 여러 개념과 연결 가능하게 함
        return self.principles
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name


@dataclass
class ConceptBond:
    """개념 간 결합 - 공통 원리가 다리"""
    concept1: ConceptAtom
    concept2: ConceptAtom
    bridge_principles: Set[Principle]  # 연결을 가능하게 한 공통 원리
    emergent_insight: str = ""  # 이 결합에서 창발된 통찰
    
    def strength(self) -> float:
        """결합 강도 = 공통 원리 수"""
        return len(self.bridge_principles)


@Cell("ConceptPolymer")
class ConceptPolymer:
    """
    개념 고분자 - 원리 기반으로 성장하는 지식 구조
    
    단백질이 아미노산 서열로 만들어지듯,
    개념들이 공통 원리로 연결되어 큰 구조를 형성
    """
    
    def __init__(self):
        self.atoms: Dict[str, ConceptAtom] = {}
        self.bonds: List[ConceptBond] = []
        self.polymers: List[List[ConceptAtom]] = []  # 연결된 덩어리들
        
        # WhyEngine 연결 - 파동 철학: Organ.get으로 존재로 연결
        self.why_engine = Organ.get("WhyEngine")
        
        # 결합 시 창발되는 통찰 맵
        self.insight_map: Dict[frozenset, str] = {
            frozenset([Principle.CAUSALITY, Principle.PROBABILITY]): "불확정한 인과",
            frozenset([Principle.CYCLE, Principle.CAUSALITY]): "인과의 순환 (윤회)",
            frozenset([Principle.HARMONY, Principle.ENTROPY]): "혼돈 속의 아름다움",
            frozenset([Principle.OBSERVATION, Principle.DUALITY]): "관측자 효과",
            frozenset([Principle.EMERGENCE, Principle.RECURSION]): "프랙탈 창발",
        }
        
        # 키워드 → 원리 매핑 (WhyEngine 결과 해석용)
        self.keyword_to_principle = {
            "인과": Principle.CAUSALITY, "원인": Principle.CAUSALITY, "결과": Principle.CAUSALITY,
            "순환": Principle.CYCLE, "반복": Principle.CYCLE, "주기": Principle.CYCLE,
            "확률": Principle.PROBABILITY, "불확정": Principle.PROBABILITY, "가능성": Principle.PROBABILITY,
            "관측": Principle.OBSERVATION, "인식": Principle.OBSERVATION, "측정": Principle.OBSERVATION,
            "엔트로피": Principle.ENTROPY, "무질서": Principle.ENTROPY, "혼돈": Principle.ENTROPY,
            "조화": Principle.HARMONY, "균형": Principle.HARMONY, "아름다움": Principle.HARMONY,
            "창발": Principle.EMERGENCE, "부분": Principle.EMERGENCE, "전체": Principle.EMERGENCE,
            "변환": Principle.TRANSFORMATION, "변화": Principle.TRANSFORMATION, "상태": Principle.TRANSFORMATION,
            "프랙탈": Principle.RECURSION, "자기": Principle.RECURSION, "재귀": Principle.RECURSION,
            "이중": Principle.DUALITY, "파동": Principle.DUALITY, "음양": Principle.DUALITY,
        }
    
    def extract_principles_from_text(self, text: str, domain: str = "general") -> Set[Principle]:
        """
        WhyEngine을 사용하여 텍스트에서 자동으로 원리 추출
        
        Args:
            text: 분석할 텍스트 (개념 설명)
            domain: 도메인 (narrative, physics, general 등)
        
        Returns:
            추출된 원리들의 집합
        """
        extracted = set()
        
        # 1. 키워드 기반 추출 (기본)
        text_lower = text.lower()
        for keyword, principle in self.keyword_to_principle.items():
            if keyword in text_lower:
                extracted.add(principle)
        
        # 2. WhyEngine 분석 (있으면)
        if self.why_engine:
            try:
                analysis = self.why_engine.analyze("concept", text, domain)
                # underlying_principle에서 추가 키워드 추출
                if hasattr(analysis, 'underlying_principle'):
                    for keyword, principle in self.keyword_to_principle.items():
                        if keyword in analysis.underlying_principle:
                            extracted.add(principle)
            except Exception:
                pass  # WhyEngine 오류 시 키워드 기반만 사용
        
        return extracted
    
    def add_atom_from_text(
        self,
        name: str,
        description: str,
        domain: str = "general"
    ) -> ConceptAtom:
        """
        텍스트 설명에서 자동으로 원리를 추출하여 개념 원자 생성
        
        이것이 핵심! 사람이 분류하지 않아도 "왜"에서 원리가 자동 추출됨
        """
        principles = self.extract_principles_from_text(description, domain)
        
        if not principles:
            # 최소 하나의 원리는 있어야 함
            principles = {Principle.EMERGENCE}  # 기본값: 창발
        
        atom = ConceptAtom(
            name=name,
            principles=principles,
            why_chain=description.split()[:5]  # 간단한 왜 사슬
        )
        self.atoms[name] = atom
        
        print(f"⚛️  '{name}' 자동 추출됨")
        print(f"    설명: {description[:50]}...")
        print(f"    원리: {', '.join(p.value for p in principles)}")
        return atom
    
    def add_atom(
        self,
        name: str,
        principles: List[Principle],
        why_chain: List[str] = None
    ) -> ConceptAtom:
        """개념 원자 추가"""
        atom = ConceptAtom(
            name=name,
            principles=set(principles),
            why_chain=why_chain or []
        )
        self.atoms[name] = atom
        print(f"⚛️  '{name}' 추가됨")
        print(f"    원리: {', '.join(p.value for p in principles)}")
        return atom
    
    def try_bond(self, name1: str, name2: str) -> Optional[ConceptBond]:
        """
        두 개념의 결합 시도
        
        공통 원리가 있어야만 결합!
        """
        if name1 not in self.atoms or name2 not in self.atoms:
            return None
        
        atom1 = self.atoms[name1]
        atom2 = self.atoms[name2]
        
        # 공통 원리 확인
        common_principles = atom1.can_bond_with(atom2)
        
        if not common_principles:
            print(f"❌ '{name1}' ↔ '{name2}': 공통 원리 없음 (결합 불가)")
            return None
        
        # 결합!
        print(f"🔗 '{name1}' ═══ '{name2}'")
        print(f"    다리: {', '.join(p.value for p in common_principles)}")
        
        # 결합으로 창발되는 통찰
        insight = self._generate_insight(common_principles)
        if insight:
            print(f"    💡 통찰: {insight}")
        
        bond = ConceptBond(
            concept1=atom1,
            concept2=atom2,
            bridge_principles=common_principles,
            emergent_insight=insight
        )
        
        self.bonds.append(bond)
        atom1.bonded_to.append(atom2)
        atom2.bonded_to.append(atom1)
        
        return bond
    
    def _generate_insight(self, principles: Set[Principle]) -> str:
        """공통 원리에서 통찰 생성"""
        # 정확히 일치하는 통찰 찾기
        for key, insight in self.insight_map.items():
            if key <= principles:  # 부분집합이면
                return insight
        
        # 없으면 원리 이름으로 생성
        if len(principles) >= 2:
            names = sorted([p.value for p in principles])
            return f"{names[0]}과 {names[1]}의 교차점"
        return ""
    
    def auto_bond_all(self) -> List[ConceptBond]:
        """
        모든 가능한 결합 자동 수행
        
        프랙탈 증식: 결합이 새 결합 가능성을 열음
        """
        print("\n🧬 자동 결합 시작 (프랙탈 성장)...")
        
        new_bonds = []
        atom_list = list(self.atoms.values())
        
        for i, atom1 in enumerate(atom_list):
            for atom2 in atom_list[i+1:]:
                # 이미 결합되어 있는지 확인
                already_bonded = any(
                    (b.concept1 == atom1 and b.concept2 == atom2) or
                    (b.concept1 == atom2 and b.concept2 == atom1)
                    for b in self.bonds
                )
                
                if not already_bonded:
                    bond = self.try_bond(atom1.name, atom2.name)
                    if bond:
                        new_bonds.append(bond)
        
        print(f"\n✨ {len(new_bonds)}개의 새 결합 형성됨")
        return new_bonds
    
    def find_polymers(self) -> List[List[ConceptAtom]]:
        """
        연결된 덩어리(고분자) 찾기
        
        Connected components 탐색
        """
        visited = set()
        polymers = []
        
        def dfs(atom: ConceptAtom, polymer: List):
            if atom in visited:
                return
            visited.add(atom)
            polymer.append(atom)
            for neighbor in atom.bonded_to:
                dfs(neighbor, polymer)
        
        for atom in self.atoms.values():
            if atom not in visited:
                polymer = []
                dfs(atom, polymer)
                if len(polymer) > 1:  # 2개 이상 연결된 것만
                    polymers.append(polymer)
        
        self.polymers = polymers
        return polymers
    
    def visualize_structure(self) -> None:
        """구조 시각화"""
        print("\n" + "=" * 50)
        print("🧬 개념 고분자 구조")
        print("=" * 50)
        
        # 고분자 찾기
        polymers = self.find_polymers()
        
        for i, polymer in enumerate(polymers, 1):
            print(f"\n📦 고분자 #{i} ({len(polymer)}개 원자):")
            
            # 구조 표시
            for atom in polymer:
                connections = [a.name for a in atom.bonded_to if a in polymer]
                principles = [p.value for p in atom.principles]
                
                if connections:
                    conn_str = " ═══ ".join(connections)
                    print(f"   [{atom.name}] ─── {conn_str}")
                else:
                    print(f"   [{atom.name}]")
                print(f"        원리: {', '.join(principles)}")
        
        # 고립된 원자
        in_polymer = set(atom for p in polymers for atom in p)
        isolated = [a for a in self.atoms.values() if a not in in_polymer]
        
        if isolated:
            print(f"\n⚪ 고립된 원자 ({len(isolated)}개):")
            for atom in isolated:
                print(f"   [{atom.name}] (결합 없음)")
        
        # 통찰 요약
        insights = [b.emergent_insight for b in self.bonds if b.emergent_insight]
        if insights:
            print(f"\n💡 창발된 통찰:")
            for insight in set(insights):
                print(f"   • {insight}")


def demo_concept_polymer():
    """개념 고분자 데모"""
    print("=" * 60)
    print("🧬 개념 고분자 데모")
    print("   '공통 원리가 연결의 다리가 된다' - 강덕리 원리")
    print("=" * 60)
    
    polymer = ConceptPolymer()
    
    # 개념 원자들 추가
    print("\n📍 개념 원자 추가:")
    
    polymer.add_atom("양자역학", [
        Principle.PROBABILITY,
        Principle.OBSERVATION,
        Principle.DUALITY,
        Principle.CAUSALITY
    ], why_chain=["불확정성", "관측", "파동함수"])
    
    polymer.add_atom("윤회", [
        Principle.CYCLE,
        Principle.CAUSALITY,
        Principle.TRANSFORMATION
    ], why_chain=["영혼", "업보", "순환"])
    
    polymer.add_atom("엔트로피", [
        Principle.ENTROPY,
        Principle.PROBABILITY,
        Principle.CAUSALITY
    ], why_chain=["무질서", "시간의 화살"])
    
    polymer.add_atom("아름다움", [
        Principle.HARMONY,
        Principle.EMERGENCE,
        Principle.DUALITY
    ], why_chain=["조화", "균형", "감동"])
    
    polymer.add_atom("프랙탈", [
        Principle.RECURSION,
        Principle.EMERGENCE,
        Principle.CYCLE
    ], why_chain=["자기유사", "무한", "패턴"])
    
    polymer.add_atom("생명", [
        Principle.EMERGENCE,
        Principle.CYCLE,
        Principle.ENTROPY,
        Principle.TRANSFORMATION
    ], why_chain=["DNA", "대사", "진화"])
    
    polymer.add_atom("의식", [
        Principle.OBSERVATION,
        Principle.EMERGENCE,
        Principle.RECURSION
    ], why_chain=["자각", "인식", "사고"])
    
    # 자동 결합
    polymer.auto_bond_all()
    
    # 구조 시각화
    polymer.visualize_structure()
    
    print("\n" + "=" * 60)
    print("🎉 데모 완료!")
    print("=" * 60)


if __name__ == "__main__":
    demo_concept_polymer()
