# -*- coding: utf-8 -*-
"""
Grammar Emergence Engine
========================

Protocol 05: Emergent Language Grammar   
  (Star)                 (Constellation Rules)      .
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger("GrammarEngine")

class GrammarRole(Enum):
    """       (Protocol 05)"""
    AGENT = "agent"       #     (Subject)
    PATIENT = "patient"   #     (Object)
    ACTION = "action"     #    (Verb)
    RESULT = "result"     #   
    MODIFIER = "modifier" #   
    CONDITION = "condition" #    (If X)
    CONSEQUENCE = "consequence" #    (Then Y)
    STIMULUS = "stimulus" #   
    FEELING = "feeling"   #   
    UNKNOWN = "unknown"

@dataclass
class SentencePattern:
    """      ( : AGENT -> ACTION -> PATIENT)"""
    roles: Tuple[GrammarRole, ...]
    frequency: int = 0
    examples: List[str] = field(default_factory=list)
    
    def confidence(self) -> float:
        #                 (   1.0)
        return min(1.0, self.frequency / 10.0)

class GrammarEmergenceEngine:
    """        """
    
    def __init__(self):
        self.patterns = defaultdict(lambda: SentencePattern(roles=(), frequency=0))
        self.role_memory = defaultdict(lambda: defaultdict(int)) # concept -> role -> count
        self.korean_mode = False #        (SOV)
        
        #          (         )
        self.rel_type_mapping = {
            'creates': (GrammarRole.AGENT, GrammarRole.ACTION, GrammarRole.PATIENT),
            'causes': (GrammarRole.AGENT, GrammarRole.ACTION, GrammarRole.PATIENT),
            'enables': (GrammarRole.AGENT, GrammarRole.ACTION, GrammarRole.PATIENT),
            'prevents': (GrammarRole.AGENT, GrammarRole.ACTION, GrammarRole.PATIENT),
            'is_a': (GrammarRole.PATIENT, GrammarRole.ACTION, GrammarRole.RESULT),
            'has': (GrammarRole.AGENT, GrammarRole.ACTION, GrammarRole.PATIENT),
            # Advanced mappings
            'if': (GrammarRole.CONDITION, GrammarRole.CONSEQUENCE), # Hypothetical
        }

    def learn_from_relationship(self, source: str, rel_type: str, target: str):
        """             """
        
        # 1.      
        roles = self._infer_roles(source, rel_type, target)
        
        if not roles:
            return

        # 2.      
        pattern_key = tuple(roles)
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = SentencePattern(roles=roles)
        
        self.patterns[pattern_key].frequency += 1
        
        #       (   5     )
        example = f"{source} {rel_type} {target}"
        self.patterns[pattern_key].examples.append(example)
        if len(self.patterns[pattern_key].examples) > 5:
            self.patterns[pattern_key].examples.pop(0)
            
        # 3.           (      )
        if len(roles) >= 3:
            self.role_memory[source][roles[0]] += 1
            self.role_memory[target][roles[2]] += 1
        
        logger.debug(f"  Learned Grammar: {example} -> {roles}")

    def _infer_roles(self, source: str, rel_type: str, target: str) -> Optional[Tuple[GrammarRole, ...]]:
        """              """
        if rel_type in self.rel_type_mapping:
            return self.rel_type_mapping[rel_type]
        return None

    def suggest_structure(self, concepts: List[str], intent: str = "statement") -> List[str]:
        """                 (Constellation   )"""
        # 1.                   
        concept_roles = {}
        for concept in concepts:
            if concept in self.role_memory:
                best_role = max(self.role_memory[concept].items(), key=lambda x: x[1])[0]
                concept_roles[concept] = best_role
            else:
                concept_roles[concept] = GrammarRole.UNKNOWN

        # 2.              
        agent = None
        patient = None
        condition = None
        consequence = None
        
        for concept, role in concept_roles.items():
            if role == GrammarRole.AGENT and not agent:
                agent = concept
            elif role == GrammarRole.PATIENT and not patient:
                patient = concept
            elif role == GrammarRole.CONDITION and not condition:
                condition = concept
            elif role == GrammarRole.CONSEQUENCE and not consequence:
                consequence = concept
        
        remaining = [c for c in concepts if c not in [agent, patient, condition, consequence]]
        
        # 3.       (            )
        sentence_parts = []
        
        if self.korean_mode:
            # SOV: [Subject] [Object] [Verb]
            #           (      )
            if agent: sentence_parts.append(f"{agent}( / )")
            if patient: sentence_parts.append(f"{patient}( / )")
            #                ,                            
            #                    (     )
            sentence_parts.extend(remaining)
            sentence_parts.append("(  )") #       
            
        else:
            # SVO: [Subject] [Verb] [Object]
            if condition and consequence:
                # Conditional: If [Condition], then [Consequence]
                sentence_parts.append(f"If {condition}")
                sentence_parts.append(f"then {consequence}")
            else:
                if agent: sentence_parts.append(agent)
                # Verb placeholder or remaining concepts acting as verb
                # For now, just append remaining
                sentence_parts.extend(remaining) 
                if patient: sentence_parts.append(patient)
        
        return sentence_parts

    def set_korean_mode(self, enabled: bool):
        self.korean_mode = enabled