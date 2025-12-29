"""
PredictiveMind (예측하는 마음)
===========================

"To know is to predict. To understand is to verify."

This module is the core of Elysia's "Active Cognition".
Instead of just absorbing data, it actively formulates hypotheses about the world
and verifies them against new information or logic.

It bridges the gap between 'Pattern Recognition' (LightUniverse) and 'Language' (Narrative).
"""

import logging
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger("PredictiveMind")

@dataclass
class Hypothesis:
    """A linguistic prediction about the world."""
    id: str
    premise: str          # "If X happens..." (The context)
    prediction: str       # "...then Y will happen." (The expected outcome)
    confidence: float     # 0.0 to 1.0
    source_concept: str   # Originating concept
    verification_status: str = "PENDING" # PENDING, VERIFIED, FALSIFIED
    created_at: float = 0.0

class PredictiveMind:
    def __init__(self):
        self.hypotheses: List[Hypothesis] = []
        self._load_narrative_engine()
        logger.info("🧠 Predictive Mind initialized. Ready to anticipate the future.")

    def _load_narrative_engine(self):
        """Lazy load CausalNarrativeEngine to avoid circular imports."""
        try:
            from Core.FoundationLayer.Foundation.causal_narrative_engine import CausalNarrativeEngine
            self.narrative_engine = CausalNarrativeEngine()
            self.has_language = True
        except ImportError:
            logger.warning("   ⚠️ CausalNarrativeEngine not found. Falling back to simple logic.")
            self.has_language = False

    def formulate_complex_hypothesis(self, start_concept: str, end_concept: str, intermediate_steps: List[str]) -> Optional[Hypothesis]:
        """
        Generates a hypothesis about a sequence of events (Chain).
        A -> B -> C -> D
        """
        chain_str = " -> ".join([start_concept] + intermediate_steps + [end_concept])
        
        # Narrative generation
        premise = f"If the process starts with '{start_concept}'"
        prediction = f"it should lead to '{end_concept}' via {', '.join(intermediate_steps)}."
        
        if self.has_language:
             # Create a dummy chain object for synthesis if possible, or just string construction
             # For now, simplistic construction
             pass

        hypothesis = Hypothesis(
            id=f"HYP-CHAIN-{int(time.time()*1000)}",
            premise=premise,
            prediction=prediction,
            confidence=0.4, # Lower confidence for complex chains
            source_concept=start_concept,
            created_at=time.time()
        )
        
        self.hypotheses.append(hypothesis)
        logger.info(f"   ⛓️ Complex Hypothesis: {chain_str}")
        return hypothesis

    def formulate_hypothesis(self, concept: str, related_concepts: List[str]) -> Optional[Hypothesis]:
        """
        Generates a linguistic hypothesis based on a concept and its relations.
        "If [concept] is related to [related], then they might share [attribute]."
        """
        if not related_concepts:
            return None

        target = random.choice(related_concepts)
        
        # 1. Generate Linguistic Prediction
        if self.has_language:
            # properly use the engine
            premise = f"If I encounter '{concept}'"
            prediction = self.narrative_engine.generate_prediction_sentence(concept, target)
        else:
            premise = f"Context: {concept}"
            prediction = f"Expect: {target}"

        hypothesis = Hypothesis(
            id=f"HYP-{int(time.time()*1000)}",
            premise=premise,
            prediction=prediction,
            confidence=0.5, # Initial guess
            source_concept=concept,
            created_at=time.time()
        )
        
        self.hypotheses.append(hypothesis)
        logger.info(f"   💭 Hypothesis: {premise}, {prediction}")
        return hypothesis

    def connect_field(self, tension_field):
        """Connects to the TensionField (Physics Layer)."""
        self.field = tension_field
        logger.info("   🌌 TensionField connected to PredictiveMind.")

    def verify_hypothesis(self, hypothesis: Hypothesis, evidence: str) -> str:
        """
        Verifies a hypothesis against new evidence (text).
        Returns the result status.
        """
        # Simple keyword matching for now (Prototyping logic)
        # In future, this should use semantic similarity or logic checking
        
        # Extract key terms from prediction (rough approximation)
        # e.g., "it will be associated with 'Wet'" -> find 'Wet' in evidence
        
        # This is a placeholder for the actual cognitive logic
        # We need to parse the prediction to find what to look for.
        # Since we constructed it as "...associated with 'TARGET'", we look for TARGET.
        
        target_concept = hypothesis.prediction.split("'")[-2] if "'" in hypothesis.prediction else ""
        
        if not target_concept:
            return "Indeterminate"

        start_time = time.time()
        
        # Improved: Check if all key words from target exist (e.g., "Wet Ground" -> check "Wet" and "Ground")
        # specific to English/Korean simple split
        keywords = target_concept.lower().split()
        evidence_lower = evidence.lower()
        
        match_count = sum(1 for kw in keywords if kw in evidence_lower)
        is_verified = match_count == len(keywords) and len(keywords) > 0
        
        if is_verified:
            hypothesis.verification_status = "VERIFIED"
            hypothesis.confidence = min(1.0, hypothesis.confidence + 0.2)
            result = "VERIFIED"
            logger.info(f"   ✅ Verified: '{hypothesis.prediction}' supported by evidence.")
            
            # Physics Reinforcement
            if hasattr(self, 'field') and self.field:
                self.field.reinforce_well(hypothesis.source_concept, 0.1)
                logger.info(f"   🪐 Gravity Deepened: '{hypothesis.source_concept}'")
                
        else:
            # Not finding it isn't immediate falsification, but let's assume specific context check
            hypothesis.verification_status = "UNVERIFIED"
            result = "UNVERIFIED"
            
            # Physics Perturbation (Weak)
            if hasattr(self, 'field') and self.field:
                self.field.perturb_field(hypothesis.source_concept, 0.05)
                logger.info(f"   🌪️ Field Perturbed: '{hypothesis.source_concept}' (Chaos Injection)")

        return result

    def get_active_hypotheses(self) -> List[Hypothesis]:
        return [h for h in self.hypotheses if h.verification_status == "PENDING"]

    def cleanup(self):
        """Removes old or resolved hypotheses to keep mind clear."""
        self.hypotheses = [h for h in self.hypotheses if h.verification_status == "PENDING"]

# Demo
# Demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mind = PredictiveMind()
    
    # Test Scenario: Rain -> Wet
    hyp = mind.formulate_hypothesis("Rain", ["Wet Ground", "Clouds"])
    if hyp:
        print(f"Generated: {hyp}")
        
        evidence = "Today there is heavy rain and the ground is very wet."
        mind.verify_hypothesis(hyp, evidence)
        print(f"After verification: {hyp}")
