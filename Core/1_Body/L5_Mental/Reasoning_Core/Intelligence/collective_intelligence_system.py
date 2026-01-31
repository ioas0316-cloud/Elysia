"""
Collective Intelligence System (         )
=================================================

"          , 9          Hyper-Space       ."

[HyperQubit Integration]
                 , HyperQubit       (Entanglement)  
  (Resonance)     '        (Flowless Computation)'            .

[9 Enneagram Archetypes]
1. Reformer (Type 1)
2. Helper (Type 2)
3. Achiever (Type 3)
4. Individualist (Type 4)
5. Investigator (Type 5)
6. Loyalist (Type 6)
7. Enthusiast (Type 7)
8. Challenger (Type 8)
9. Peacemaker (Type 9)
"""

import logging
import random
import math
import time
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum, auto

logger = logging.getLogger("CollectiveIntelligence")

# [Hyper-Conversion] Import Real HyperQubit
try:
    from Core.1_Body.L6_Structure.Wave.hyper_qubit import HyperQubit, QubitState
except ImportError:
    # Fallback if module missing
    HyperQubit = None
    QubitState = None

# [Integration] Use EnneagramType directly
try:
    from Core.1_Body.L1_Foundation.Foundation.dual_layer_personality import EnneagramType
except ImportError:
    # Fallback definition
    class EnneagramType(Enum):
        TYPE_1 = "reformer"
        TYPE_2 = "helper"
        TYPE_3 = "achiever"
        TYPE_4 = "individualist"
        TYPE_5 = "investigator"
        TYPE_6 = "loyalist"
        TYPE_7 = "enthusiast"
        TYPE_8 = "challenger"
        TYPE_9 = "peacemaker"

#          (Enneagram Integration/Disintegration Lines & Wings)
COMPLEMENTARY_PAIRS = [
    (EnneagramType.TYPE_5, EnneagramType.TYPE_8), # Investigator   Challenger
    (EnneagramType.TYPE_2, EnneagramType.TYPE_4), # Helper   Individualist
    (EnneagramType.TYPE_3, EnneagramType.TYPE_9), # Achiever   Peacemaker
    (EnneagramType.TYPE_7, EnneagramType.TYPE_1), # Enthusiast   Reformer
    (EnneagramType.TYPE_6, EnneagramType.TYPE_9), # Loyalist   Peacemaker
]


@dataclass
class Opinion:
    """   (Opinion)"""
    content: str
    consciousness_type: EnneagramType
    confidence: float = 0.5  # 0.0 ~ 1.0
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self):
        return f"[{self.consciousness_type.name}] {self.content} (   : {self.confidence:.0%})"


@dataclass 
class Debate:
    """       (Resonance Cycle)"""
    topic: str
    round_number: int
    opinions: List[Opinion] = field(default_factory=list)
    critiques: Dict[EnneagramType, List[str]] = field(default_factory=dict)


class ConsciousPerspective:
    """
          - 9             HyperQubit   
    
    [HyperQubit Integration]
                   , '        (Psionic Entity)'    
      (Topic)    (Resonance)     (Entanglement)       .
    """
    
    def __init__(self, consciousness_type: EnneagramType):
        self.type = consciousness_type
        self.energy = 1.0 #        
        self.memory: List[Opinion] = []
        
        # HyperQubit    (    Mock   )
        if HyperQubit:
            #                      
            bases = self._get_initial_bases(consciousness_type)
            
            self.mind_qubit = HyperQubit(
                name=f"Mind_{consciousness_type.name}",
                epistemology={"origin": {"score": 1.0, "meaning": f"Archetype {consciousness_type.value}"}}
            )
            #           (   state   )
            self.mind_qubit.state.alpha = bases['alpha']
            self.mind_qubit.state.beta = bases['beta']
            self.mind_qubit.state.gamma = bases['gamma']
            self.mind_qubit.state.delta = bases['delta']
            self.mind_qubit.state.normalize()
            
            logger.info(f"  {self.mind_qubit.name} initialized (Resonance Active)")
        else:
            self.mind_qubit = None
            logger.warning("HyperQubit module missing, running in degraded mode.")

    def _get_initial_bases(self, etype: EnneagramType) -> Dict[str, complex]:
        """          4D          """
        # alpha(Point/Data), beta(Line/Logic), gamma(Space/Context), delta(God/Will)
        if etype == EnneagramType.TYPE_1: # Reformer
            return {'alpha': 0.1, 'beta': 0.6, 'gamma': 0.1, 'delta': 0.2} # Logic/Rule driven
        elif etype == EnneagramType.TYPE_2: # Helper
            return {'alpha': 0.3, 'beta': 0.5, 'gamma': 0.1, 'delta': 0.1} # Connection (Line) & Person (Point)
        elif etype == EnneagramType.TYPE_3: # Achiever
            return {'alpha': 0.4, 'beta': 0.2, 'gamma': 0.1, 'delta': 0.3} # Result (Point) & Ambition (God)
        elif etype == EnneagramType.TYPE_4: # Individualist
            return {'alpha': 0.1, 'beta': 0.1, 'gamma': 0.5, 'delta': 0.3} # Depth (Space) & Meaning (God)
        elif etype == EnneagramType.TYPE_5: # Investigator
            return {'alpha': 0.3, 'beta': 0.5, 'gamma': 0.2, 'delta': 0.0} # Data (Point) & Logic (Line)
        elif etype == EnneagramType.TYPE_6: # Loyalist
            return {'alpha': 0.1, 'beta': 0.4, 'gamma': 0.4, 'delta': 0.1} # System (Line) & Safety Field (Space)
        elif etype == EnneagramType.TYPE_7: # Enthusiast
            return {'alpha': 0.3, 'beta': 0.1, 'gamma': 0.5, 'delta': 0.1} # Variety (Point) & Field (Space)
        elif etype == EnneagramType.TYPE_8: # Challenger
            return {'alpha': 0.1, 'beta': 0.3, 'gamma': 0.1, 'delta': 0.5} # Force (Line) & Will (God)
        elif etype == EnneagramType.TYPE_9: # Peacemaker
            return {'alpha': 0.1, 'beta': 0.2, 'gamma': 0.6, 'delta': 0.1} # Harmony (Space)
        else:
            return {'alpha': 0.25, 'beta': 0.25, 'gamma': 0.25, 'delta': 0.25}

    def generate_opinion(self, topic: str) -> Opinion:
        """
                       (Quantum Resonance)
        """
        alignment = 0.5
        
        # 1.            (Entangle)
        if self.mind_qubit:
            # Topic Qubit    (Temporary Topic)
            #                      ,                   
            topic_qubit = self._create_topic_qubit(topic)
            
            #        (Connect)
            self.mind_qubit.connect(topic_qubit) 
            
            #        (  )
            alignment = self._calculate_resonance(topic_qubit)
            
            #    (Interference)                  (    )
            # self.mind_qubit._react(topic_qubit) #         
        else:
             alignment = random.random() # Fallback

        # 2.       (Flowless State Transition)
        opinion_content = self._quantum_state_to_text(topic, alignment)
        
        op = Opinion(
            content=opinion_content,
            consciousness_type=self.type,
            confidence=float(max(0.1, min(0.99, alignment))),
            reasoning=f"Quantum Resonance: {alignment:.2f}"
        )
        self.memory.append(op)
        return op
    
    def _create_topic_qubit(self, topic: str) -> Any:
        #                 
        seed = sum(ord(c) for c in topic)
        random.seed(seed)
        tq = HyperQubit(name=f"Topic_{topic[:10]}", value=topic)
        tq.state.alpha = random.random()
        tq.state.beta = random.random()
        tq.state.gamma = random.random()
        tq.state.delta = random.random()
        tq.state.normalize()
        return tq

    def _calculate_resonance(self, target: Any) -> float:
        """HyperQubit            """
        if not self.mind_qubit or not target: return 0.0
        s = self.mind_qubit.state
        t = target.state
        # Complex inner product magnitude
        dot = abs(s.alpha * t.alpha.conjugate() + 
                  s.beta * t.beta.conjugate() + 
                  s.gamma * t.gamma.conjugate() + 
                  s.delta * t.delta.conjugate())
        return dot

    def _quantum_state_to_text(self, topic: str, alignment: float) -> str:
        """              (Collapse)"""
        if not self.mind_qubit:
             return f"{topic}               ."

        probs = self.mind_qubit.state.probabilities()
        dominant = max(probs, key=probs.get)
        
        #       
        interpretations = {
            "Point": f"       ",
            "Line": f"       ",
            "Space": f"       ",
            "God": f"       "
        }
        
        cert = "   " if alignment > 0.8 else ("   " if alignment > 0.5 else "   ")
        nucleus = interpretations.get(dominant, "      ")
        
        #           
        exprs = {
            EnneagramType.TYPE_1: f"{cert} {topic}  {nucleus}                (System).",
            EnneagramType.TYPE_2: f"{cert} {topic}  {nucleus}                 (Heart).",
            EnneagramType.TYPE_3: f"{cert} {topic}  {nucleus}                  (Goal).",
            EnneagramType.TYPE_4: f"{cert} {topic}  {nucleus}                 (Soul).",
            EnneagramType.TYPE_5: f"{cert} {topic}  {nucleus}                 (Mind).",
            EnneagramType.TYPE_6: f"{cert} {topic}  {nucleus}                (Safety).",
            EnneagramType.TYPE_7: f"{cert} {topic}  {nucleus}                   (Fun).",
            EnneagramType.TYPE_8: f"{cert} {topic}  {nucleus}                 (Power).",
            EnneagramType.TYPE_9: f"{cert} {topic}  {nucleus}                   (Peace).",
        }
        base_expr = exprs.get(self.type, f"{cert} {nucleus}      .")
        
        return f"{base_expr} (  : {alignment:.1%})"
    
    def critique(self, other_opinion: Opinion) -> str:
        is_complementary = any(self.type in p and other_opinion.consciousness_type in p 
                               for p in COMPLEMENTARY_PAIRS)
        if is_complementary:
            return f"[{self.type.name} {other_opinion.consciousness_type.name}] Qubit Interference:          "
        return f"[{self.type.name}] Qubit Resonance:    "
    
    def update_confidence(self, feedback: float):
        self.energy = min(1.0, max(0.1, self.energy + feedback * 0.1))


class RoundTableCouncil:
    """
         (Round Table Council)
    
                              .
    """
    
    def __init__(self):
        # 9               
        self.perspectives: Dict[EnneagramType, ConsciousPerspective] = {
            ct: ConsciousPerspective(ct) for ct in EnneagramType
        }
        self.debates: List[Debate] = []
        self.consensus_history: List[Dict[str, Any]] = []
        logger.info("   Round Table Council Assembled (9 Enneagram Types with HyperQubit)")
    
    def convene(self, topic: str) -> List[Opinion]:
        """
                                 .
        """
        logger.info(f"   Round Table Convening on: {topic}")
        
        opinions = []
        for perspective in self.perspectives.values():
            opinion = perspective.generate_opinion(topic)
            opinions.append(opinion)
        
        return opinions
    
    def debate(self, topic: str, rounds: int = 3) -> Debate:
        """
          (Resonance Cycle)       .
        """
        logger.info(f"   Starting {rounds}-round Resonance Cycle on: {topic}")
        
        final_debate = Debate(topic=topic, round_number=0)
        
        # Round 1:      
        all_opinions = self.convene(topic)
        final_debate.opinions = all_opinions
        final_debate.round_number = 1
        
        # Round 2+:       
        for round_num in range(2, rounds + 1):
            critiques = {}
            for perspective in self.perspectives.values():
                perspective_critiques = []
                for opinion in all_opinions:
                    if opinion.consciousness_type != perspective.type:
                        critique = perspective.critique(opinion)
                        perspective_critiques.append(critique)
                
                if perspective_critiques:
                    critiques[perspective.type] = perspective_critiques
            
            final_debate.critiques = critiques
            final_debate.round_number = round_num
            
            #      /   (주권적 자아)
            for opinion in all_opinions:
                #          :                 
                critique_count = sum(1 for cts in critiques.values() for c in cts if opinion.consciousness_type.name in c)
                adjustment = 0.05 if critique_count < 3 else -0.05
                opinion.confidence = min(1.0, max(0.1, opinion.confidence + adjustment))
        
        self.debates.append(final_debate)
        return final_debate
    
    def reach_consensus(self, debate: Debate) -> Dict[str, Any]:
        """
                         .
        """
        #            (Energy * Resonance)
        weighted_opinions = []
        for opinion in debate.opinions:
            weight = opinion.confidence * self.perspectives[opinion.consciousness_type].energy
            weighted_opinions.append((opinion, weight))
        
        weighted_opinions.sort(key=lambda x: x[1], reverse=True)
        top_opinions = weighted_opinions[:3]
        
        consensus = {
            "topic": debate.topic,
            "rounds": debate.round_number,
            "primary_conclusion": top_opinions[0][0].content if top_opinions else "     ",
            "supporting_views": [op.content for op, _ in top_opinions[1:]],
            "confidence": sum(w for _, w in top_opinions) / (len(top_opinions) or 1),
            "dissenting_voices": [op.content for op, w in weighted_opinions if w < 0.3][:2],
            "total_perspectives": len(debate.opinions),
            "critiques_exchanged": sum(len(c) for c in debate.critiques.values())
        }
        
        self.consensus_history.append(consensus)
        logger.info(f"  Consensus Reached via Resonance: {consensus['primary_conclusion'][:50]}...")
        return consensus
    
    def full_deliberation(self, topic: str, rounds: int = 3) -> Dict[str, Any]:
        debate = self.debate(topic, rounds)
        return self.reach_consensus(debate)
    
    def get_council_state(self) -> Dict[str, Any]:
        return {
            "perspectives_count": len(self.perspectives),
            "total_debates": len(self.debates),
            "consensus_reached": len(self.consensus_history),
            "perspective_energies": {ct.name: p.energy for ct, p in self.perspectives.items()}
        }


class CollectiveIntelligenceSystem:
    """
              (Collective Intelligence System)
    """
    
    def __init__(self):
        self.council = RoundTableCouncil()
        self.active = True
        logger.info("  Collective Intelligence System Initialized (HyperQubit Core)")
    
    def deliberate(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        depth = min(5, max(1, depth))
        return self.council.full_deliberation(topic, depth)
    
    def quick_opinion(self, topic: str, consciousness_type: EnneagramType = None) -> Opinion:
        if consciousness_type is None:
            consciousness_type = random.choice(list(EnneagramType))
        
        perspective = self.council.perspectives.get(consciousness_type)
        if perspective:
            return perspective.generate_opinion(topic)
        return Opinion(content="     ", consciousness_type=consciousness_type)
    
    def get_all_perspectives(self, topic: str) -> Dict[EnneagramType, Opinion]:
        opinions = {}
        for ct, perspective in self.council.perspectives.items():
            opinions[ct] = perspective.generate_opinion(topic)
        return opinions
    
    def find_consensus_points(self, topic: str) -> List[str]:
        result = self.deliberate(topic)
        return [result["primary_conclusion"]] + result.get("supporting_views", [])
    
    def find_conflict_points(self, topic: str) -> List[Tuple[EnneagramType, EnneagramType, str]]:
        conflicts = []
        opinions = self.get_all_perspectives(topic)
        for pair in COMPLEMENTARY_PAIRS:
            type1, type2 = pair
            if type1 in opinions and type2 in opinions:
                conflicts.append((
                    type1, type2,
                    f"{opinions[type1].content[:30]}... vs {opinions[type2].content[:30]}..."
                ))
        return conflicts


#    
_collective_instance: Optional[CollectiveIntelligenceSystem] = None

def get_collective_intelligence() -> CollectiveIntelligenceSystem:
    global _collective_instance
    if _collective_instance is None:
        _collective_instance = CollectiveIntelligenceSystem()
    return _collective_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    #    
    collective = get_collective_intelligence()
    
    topic = "                  ?"
    
    print("\n" + "=" * 60)
    print(f"   COLLECTIVE RESONANCE: {topic}")
    print("=" * 60)
    
    #   
    consensus = collective.deliberate(topic, depth=3)
    
    print(f"\n  PRIMARY RESONANCE (Conclusion):")
    print(f"   {consensus['primary_conclusion']}")
    
    print(f"\n  SUPPORTING FREQUENCIES:")
    for view in consensus['supporting_views']:
        print(f"     {view}")
    
    print(f"\n   DISSENTING FREQUENCIES:")
    for voice in consensus['dissenting_voices']:
        print(f"     {voice}")
