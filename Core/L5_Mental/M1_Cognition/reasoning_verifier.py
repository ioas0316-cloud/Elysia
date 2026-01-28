import torch
import numpy as np
from typing import List, Dict, Optional
from Core.L5_Mental.M1_Cognition.thought_fragment import CognitivePulse, ThoughtFragment
from Core.L5_Mental.M1_Cognition.cognitive_types import AuditGrade, ViolationType
from Core.L1_Foundation.M1_Keystone.d7_vector import D7Vector

class ReasoningVerifier:
    """
    [STEEL CORE] The Internal Auditor
    =================================
    Validates logical consistency and prevents halluciations/drift.
    """
    def __init__(self, consistency_threshold: float = 0.7):
        self.threshold = consistency_threshold

    def audit_pulse(self, pulse: CognitivePulse) -> AuditGrade:
        """
        Performs a full audit of a cognitive cycle.
        """
        if not pulse.fragments:
            return AuditGrade.FRACTURED

        # 1. Check for Semantic Drift (D7 distance between start and end)
        first_d7 = next((f.d7_projection for f in pulse.fragments if f.d7_projection), None)
        last_d7 = next((f.d7_projection for f in reversed(pulse.fragments) if f.d7_projection), None)

        if not first_d7 or not last_d7:
            return AuditGrade.HOLLOW

        resonance = first_d7.resonate(last_d7)
        pulse.fragments[-1].resonance_score = resonance # Attach for narrator

        # 2. Determine Grade
        if resonance > 0.9:
            grade = AuditGrade.RADIANT
        elif resonance > self.threshold:
            grade = AuditGrade.COHERENT
        elif resonance > 0.4:
            grade = AuditGrade.DISSONANT
        else:
            grade = AuditGrade.FRACTURED

        # 3. Success Flag
        pulse.success = (grade in [AuditGrade.RADIANT, AuditGrade.COHERENT])
        
        return grade

    def get_violation_report(self, pulse: CognitivePulse) -> ViolationType:
        """Identifies specific failure modes."""
        grade = self.audit_pulse(pulse)
        if grade == AuditGrade.RADIANT: return ViolationType.NONE
        
        # Heuristic for demo
        if pulse.fragments[-1].resonance_score < 0.3:
            return ViolationType.SEMANTIC_DRIFT
        return ViolationType.NONE
