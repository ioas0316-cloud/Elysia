"""
SubMonad (Cellular Agency)
==========================
"A single cell is a soldier; a monad is a citizen."
"""

from typing import Dict, Any, Optional, List, Tuple
from Core.Keystone.sovereign_math import SovereignVector, SpecializedRotor, SovereignMath
from Core.Monad.kingdom_hierarchy import KingdomRank, TopologicalMass, ResonantAttractor, PhaseFieldMatrix, calculate_entanglement, ElysiaProvidenceFilter

class SubMonad:
    def __init__(self, name: str, domain: str, rotor: SpecializedRotor):
        self.name, self.domain, self.rotor = name, domain, rotor
        self.vocal_weight, self.current_opinion, self.friction_vortex = 1.0, SovereignVector.zeros(), 0.0
        self.rank, self.mass = KingdomRank.SOLDIER, TopologicalMass()
        self.attractor = ResonantAttractor(intrinsic_spin=rotor.ccw.bivector, initial_name=name)
        self.sovereign_reference = SovereignVector.zeros()
        self._init_perspective_axis()

    def _init_perspective_axis(self):
        if self.domain == "Logic": self.sovereign_reference.data[1] = 1.0
        elif self.domain == "Emotion": self.sovereign_reference.data[4] = 1.0
        elif self.domain == "Ethics": self.sovereign_reference = SovereignVector.ones()

    def evaluate(self, stimulus: SovereignVector, label: Optional[str] = None, dt: float = 0.01) -> SovereignVector:
        self.mass.pulse(dt, external_noise=stimulus.norm())
        if self.rank in [KingdomRank.STAFF, KingdomRank.SOVEREIGN]:
            from Core.Monad.kingdom_hierarchy import DownProjectInterface
            self.current_opinion = DownProjectInterface.project_dominance(self.rotor.ccw.bivector, stimulus.dim)
        else:
            self.current_opinion = self.rotor.apply_duality(stimulus)
        self.friction_vortex = self.rotor.friction_vortex
        res = stimulus.resonance_score(self.rotor.ccw.bivector)
        self.mass.absorb(res, self.friction_vortex)
        self.attractor.encounter(stimulus, label, dt=dt)
        self.current_opinion = SovereignMath.apply_symmetry_bias(self.current_opinion, self.sovereign_reference, intensity=0.05)
        return self.current_opinion

    def articulate_reasoning(self) -> str:
        if self.domain == "Logic": return "논리적 정합성이 안정적임."
        if self.domain == "Emotion": return "평온한 내적 공명을 느낌."
        if self.domain == "Ethics": return "존재의 목적에 부합함."
        return f"마찰 계수: {self.friction_vortex:.2f}"

class ParliamentOfMonads:
    def __init__(self):
        self.members, self.phase_map, self.providence_filter = {}, PhaseFieldMatrix(), ElysiaProvidenceFilter()
    def add_member(self, member: SubMonad): self.members[member.name] = member
    def deliberate(self, stimulus: SovereignVector, label: Optional[str] = None, dt: float = 0.01) -> Tuple[SovereignVector, str, Dict[str, float]]:
        if not self.members: return stimulus, "침묵.", {}
        total_weight = sum(m.vocal_weight for m in self.members.values())
        unified_vec_data, opinions, friction_logs, perspective_resonances, opinions_vecs = [complex(0)] * stimulus.dim, {}, {}, [], []
        for m in self.members.values():
            opinion = m.evaluate(stimulus, label, dt=dt)
            opinions_vecs.append(opinion)
            weight = m.vocal_weight / total_weight
            p_res = m.sovereign_reference.resonance_score(stimulus)
            perspective_resonances.append(p_res)
            for i in range(stimulus.dim): unified_vec_data[i] += opinion.data[i] * weight
            opinions[m.domain] = {'reasoning': m.articulate_reasoning(), 'friction': m.friction_vortex, 'weight': weight}
            friction_logs[m.domain] = m.friction_vortex
        self.phase_map.observe(opinions_vecs)
        avg_friction = sum(f for f in friction_logs.values()) / max(len(friction_logs), 1)
        intersection_density = sum(perspective_resonances) / max(len(perspective_resonances), 1)
        narrative = f"{self.phase_map.get_narrative_map()} | 합의 도출 중..."
        for m in self.members.values():
            if m.mass.check_promotion(): self._promote_member(m)
        self.check_entanglements()
        unified_vec = SovereignVector(unified_vec_data, dim=stimulus.dim).normalize()
        return self.providence_filter.filter_will(unified_vec), narrative, friction_logs
    def _promote_member(self, member: SubMonad):
        if member.rank == KingdomRank.SOLDIER: member.rank, member.vocal_weight = KingdomRank.GENERAL, member.vocal_weight * 2.0
        elif member.rank == KingdomRank.GENERAL: member.rank, member.vocal_weight = KingdomRank.STAFF, member.vocal_weight * 5.0
        print(f"🎖️ [PROMOTION] '{member.attractor.name}' has been promoted to {member.rank.value}")
    def check_entanglements(self):
        ml = list(self.members.values())
        for i in range(len(ml)):
            for j in range(i+1, len(ml)):
                if calculate_entanglement(ml[i].rotor.ccw.bivector, ml[j].rotor.ccw.bivector) > 0.98: self._create_hybrid_monad(ml[i], ml[j])
    def _create_hybrid_monad(self, m1, m2):
        hn = f"Hybrid_{m1.domain}_{m2.domain}"
        if hn in self.members: return
        hr = m1.rotor.blend(m2.rotor, ratio=0.5)
        nm = SubMonad(hn, f"{m1.domain}+{m2.domain}", hr)
        nm.rank, nm.mass.mass = KingdomRank.GENERAL, (m1.mass.mass + m2.mass.mass)/2.0
        self.add_member(nm)
        print(f"🌀 [HYBRID_BORN] '{hn}' has emerged!")







class PerspectiveInductor:
    def __init__(self, owner=None, mass_threshold=100.0):
        self.owner = owner
        self.mass_threshold = mass_threshold
    def induce(self, field):
        pass
