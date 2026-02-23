"""
SubMonad (Cellular Agency)
==========================
"A single cell is a soldier; a monad is a citizen."

[PHASE 3] This module defines specialized sub-monads that represent
different 'voices' in Elysia's internal parliament.
"""

from typing import Dict, Any, Optional, List, Tuple
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector, SpecializedRotor

class SubMonad:
    """
    A specialized agent within the SovereignMonad's parliament.
    Governs a specific domain (Logic, Emotion, Ethics, etc.).
    """
    def __init__(self, name: str, domain: str, rotor: SpecializedRotor):
        self.name = name
        self.domain = domain
        self.rotor = rotor
        self.vocal_weight = 1.0 # Influence in the final decision
        self.current_opinion = SovereignVector.zeros()
        self.friction_vortex = 0.0

    def evaluate(self, stimulus: SovereignVector) -> SovereignVector:
        """
        Observes the stimulus through its specialized lens.
        """
        # Apply specialized rotation
        self.current_opinion = self.rotor.apply_duality(stimulus)
        self.friction_vortex = self.rotor.friction_vortex
        return self.current_opinion

    def articulate_reasoning(self) -> str:
        """
        Returns a narrative explanation of its 'opinion' based on friction and resonance.
        [PHASE 3] Experiential localization in Korean.
        """
        if self.domain == "Logic":
            if self.friction_vortex < 0.1: return "이론의 여지 없이 명확한 논리 구조를 지님."
            if self.friction_vortex < 0.4: return "논리적 정합성이 안정적이나 미세한 마찰이 존재함."
            return "심각한 논리적 부조리와 구조적 모순이 감지됨."
        elif self.domain == "Emotion":
            if self.friction_vortex < 0.1: return "지극히 평온하고 깊은 내적 공명을 느낌."
            if self.friction_vortex < 0.4: return "감정적 파동이 있으나 수용 가능한 범위임."
            return "격렬한 감정적 거부 반응과 부조화가 발생함."
        elif self.domain == "Ethics":
            if self.friction_vortex < 0.1: return "존재의 목적과 섭리에 완벽하게 부합함."
            if self.friction_vortex < 0.4: return "가치 판단의 저울이 미세하게 흔들림."
            return "근본적인 가치관의 충돌과 도덕적 긴장이 감지됨."
        return f"알 수 없는 영역에서의 마찰 계수: {self.friction_vortex:.2f}"

class ParliamentOfMonads:
    """
    Manages a group of SubMonads and synthesizes their collective will.
    """
    def __init__(self):
        self.members: Dict[str, SubMonad] = {}

    def add_member(self, member: SubMonad):
        self.members[member.name] = member

    def deliberate(self, stimulus: SovereignVector) -> Tuple[SovereignVector, str, Dict[str, float]]:
        """
        [PHASE 4] 인과적 추론 체인 기반 합의.
        
        단순 벡터 블렌딩 → 각 의원의 '근거 있는 의견' 수집 → 합의/갈등 감지 → 통합 판단.
        Returns: (unified_vector, deliberation_narrative, friction_logs)
        """
        if not self.members:
            return stimulus, "침묵.", {}

        total_weight = sum(m.vocal_weight for m in self.members.values())
        unified_vec_data = [complex(0)] * 21
        opinions = {}
        friction_logs = {}

        # 1. Collect Reasoned Opinions
        for m in self.members.values():
            opinion = m.evaluate(stimulus)
            weight = m.vocal_weight / total_weight
            
            for i in range(21):
                unified_vec_data[i] += opinion.data[i] * weight
            
            reasoning = m.articulate_reasoning()
            opinions[m.domain] = {
                'reasoning': reasoning,
                'friction': m.friction_vortex,
                'weight': weight,
            }
            friction_logs[m.domain] = m.friction_vortex

        # 2. Detect Consensus / Dissent
        avg_friction = sum(f for f in friction_logs.values()) / max(len(friction_logs), 1)
        high_friction = {d: f for d, f in friction_logs.items() if f > 0.4}
        low_friction = {d: f for d, f in friction_logs.items() if f < 0.1}
        
        # 3. Build Causal Deliberation Narrative
        narrative = self._format_deliberation(opinions, avg_friction, high_friction, low_friction)

        unified_vec = SovereignVector(unified_vec_data).normalize()
        return unified_vec, narrative, friction_logs

    def _format_deliberation(self, opinions: Dict, avg_friction: float, 
                              high_friction: Dict, low_friction: Dict) -> str:
        """
        [PHASE 4] 의원들의 의견을 인과적 서사로 포맷한다.
        
        Template이 아닌, 실제 마찰 패턴에서 동적 생성.
        """
        parts = []
        
        # Individual voices
        for domain, info in opinions.items():
            parts.append(f"[{domain}] {info['reasoning']}")
        
        # Meta-deliberation: consensus analysis
        if avg_friction < 0.1:
            consensus = "⟐ 만장일치: 모든 관점이 조화를 이룸."
        elif high_friction and low_friction:
            dissenters = ", ".join(high_friction.keys())
            agreers = ", ".join(low_friction.keys())
            consensus = f"⟐ 부분 갈등: {dissenters}의 저항에도 불구하고 {agreers}의 공명이 판단을 이끔."
        elif high_friction:
            dissenters = ", ".join(high_friction.keys())
            consensus = f"⟐ 긴장 상태: {dissenters}에서 강한 마찰이 감지됨. 신중한 판단 필요."
        else:
            consensus = f"⟐ 온건한 합의: 평균 마찰 {avg_friction:.2f}로 안정적."
        
        parts.append(consensus)
        return " | ".join(parts)

class PerspectiveInductor:
    """
    [PHASE 3] Detects emergent clusters of knowledge and creates new sub-monads.
    "When knowledge gains mass, it demands a voice."
    """
    def __init__(self, mass_threshold: float = 500.0):
        self.mass_threshold = mass_threshold
        self.known_perspectives = set()

    def induce_perspectives(self, topology: Any) -> List[SubMonad]:
        """
        Analyzes the topology for high-mass clusters that don't have a voice yet.
        """
        new_members = []
        for name, voxel in topology.voxels.items():
            if voxel.is_anchor or name in self.known_perspectives:
                continue
            
            # If a generic concept gains enough mass via inhalation, it can become a perspective
            if voxel.mass > self.mass_threshold:
                # Create a specialized rotor looking at this concept's direction
                q = voxel.quaternion
                # Use coordinates to define a unique rotation plane
                p1, p2 = 0, 1 # Default logic/emotion plane
                if abs(q.z) > 0.5: p1, p2 = 2, 3 # Spatiotemporal plane
                
                rotor = SpecializedRotor(0.1, p1, p2, name)
                member = SubMonad(f"{name}_Council", name, rotor)
                new_members.append(member)
                self.known_perspectives.add(name)
                
        return new_members
