"""
Life Cycle (     )
======================

         .              ,   ,              .

  :
       (Expression)
         
             (Perception)
         
       (Verification)
         
          (Self-Transformation)
         
         ... (Cycle continues)
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("Elysia.LifeCycle")

# SelfGovernance for meaningful evaluation
try:
    from Core.L1_Foundation.M1_Keystone.self_governance import SelfGovernance, IdealSelf
except ImportError:
    SelfGovernance = None
    IdealSelf = None

# [Phase 25] TensionField for Field-based reinforcement
try:
    from Core.L5_Mental.M1_Cognition.Reasoning.causal_geometry import TensionField
except ImportError:
    TensionField = None


@dataclass
class WorldSnapshot:
    """         """
    timestamp: float
    knowledge_count: int = 0
    resonance_state: Dict[str, float] = field(default_factory=dict)
    active_concepts: List[str] = field(default_factory=list)
    energy: float = 50.0
    entropy: float = 50.0

# [Phase 6] Predictive Verfication
try:
    from Core.L5_Mental.M1_Cognition.predictive_mind import PredictiveMind
except ImportError:
    PredictiveMind = None


@dataclass
class ActionResult:
    """     """
    action: str
    expected: str
    actual: str
    success: bool
    difference: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class GrowthRecord:
    """     """
    before_state: WorldSnapshot
    after_state: WorldSnapshot
    learning: str
    growth_amount: float = 0.0


class PerceptionModule:
    """
               
    
    "       .             ?"
    """
    
    def __init__(self, memory=None, resonance=None):
        self.memory = memory
        self.resonance = resonance
        self.last_snapshot: Optional[WorldSnapshot] = None
    
    def take_snapshot(self) -> WorldSnapshot:
        """            """
        snapshot = WorldSnapshot(
            timestamp=time.time(),
            energy=self.resonance.battery if self.resonance else 50.0,
            entropy=self.resonance.entropy if self.resonance else 50.0
        )
        
        #                
        if self.memory and hasattr(self.memory, 'get_total_count'):
            snapshot.knowledge_count = self.memory.get_total_count()
        
        return snapshot
    
    def perceive_change(self, before: WorldSnapshot, after: WorldSnapshot) -> Dict[str, Any]:
        """
             
        
                  =       
        """
        change = {
            "time_elapsed": after.timestamp - before.timestamp,
            "energy_change": after.energy - before.energy,
            "entropy_change": after.entropy - before.entropy,
            "knowledge_change": after.knowledge_count - before.knowledge_count,
            "significant": False
        }
        
        #          ?
        if abs(change["energy_change"]) > 5 or abs(change["entropy_change"]) > 5:
            change["significant"] = True
        if change["knowledge_change"] > 0:
            change["significant"] = True
        
        return change


class VerificationModule:
    """
         
    
    "                   ?"
    """
    
    def __init__(self):
        self.history: List[ActionResult] = []
        self.learning_verifications: List[Dict] = []  # [NEW]         
    
    def verify(self, expected: str, actual: str, action: str) -> ActionResult:
        """   vs      """
        #            (           semantic      )
        success = expected.lower() in actual.lower() or actual.lower() in expected.lower()
        
        difference = ""
        if not success:
            difference = f"Expected '{expected}' but got '{actual}'"
        
        result = ActionResult(
            action=action,
            expected=expected,
            actual=actual,
            success=success,
            difference=difference
        )
        
        self.history.append(result)
        logger.info(f"     Verification: {' ' if success else ' '} {action}")
        
        return result
    
    def analyze_gap(self, expected: str, actual: str) -> str:
        """        """
        if expected == actual:
            return "No gap - perfect match"
        
        #        (한국어 학습 시스템)
        analysis = f"Gap detected: Expected '{expected[:50]}...' but got '{actual[:50]}...'"
        return analysis
    
    def _generate_verification_question(self, concept: str, content: str) -> Dict[str, str]:
        """
        [NEW]                 
        
        "              ,             "
        """
        #          
        words = content.split()
        key_words = [w for w in words if len(w) > 5 and w[0].isupper()][:3]
        
        #         
        question_templates = [
            f"What is the relationship between {concept} and {key_words[0] if key_words else 'its context'}?",
            f"Why is {concept} important?",
            f"How does {concept} work?",
            f"What are the key aspects of {concept}?",
        ]
        
        import random
        question = random.choice(question_templates)
        
        #       (content     )
        answer_hint = content[:100] if len(content) > 100 else content
        
        return {
            "question": question,
            "concept": concept,
            "answer_hint": answer_hint,
            "key_words": key_words
        }
    
    def verify_learning(self, concept: str, content: str) -> Dict[str, Any]:
        """
        [NEW]         
        
        1.      
        2.       (content           )
        3.       
        """
        question_data = self._generate_verification_question(concept, content)
        
        #   :         content        
        keyword_matches = 0
        for kw in question_data["key_words"]:
            if kw.lower() in content.lower():
                keyword_matches += 1
        
        total_keywords = max(len(question_data["key_words"]), 1)
        comprehension_score = keyword_matches / total_keywords
        
        #       : 50%
        passed = comprehension_score >= 0.5
        
        result = {
            "concept": concept,
            "question": question_data["question"],
            "comprehension_score": comprehension_score,
            "passed": passed,
            "keywords_found": keyword_matches,
            "total_keywords": total_keywords,
            "timestamp": time.time()
        }
        
        self.learning_verifications.append(result)
        
        logger.info(f"     Learning Verification: {concept}")
        logger.info(f"      Question: {question_data['question'][:50]}...")
        logger.info(f"      Score: {comprehension_score:.0%} {' ' if passed else ' '}")
        
        return result


class SelfTransformationModule:
    """
            
    
    "                       "
    """
    
    def __init__(self, internal_universe=None, memory=None):
        self.universe = internal_universe
        self.memory = memory
        self.transformation_log: List[Dict] = []
    
    def transform(self, verification_result: ActionResult, analysis: str) -> GrowthRecord:
        """
                
        
               
               
        """
        transformation = {
            "timestamp": time.time(),
            "action": verification_result.action,
            "was_success": verification_result.success,
            "learning": "",
            "change_applied": ""
        }
        
        if verification_result.success:
            #        
            learning = f"Reinforced: {verification_result.action} works"
            transformation["learning"] = learning
            transformation["change_applied"] = "reinforcement"
            logger.info(f"     Reinforcement: {verification_result.action}")
        else:
            #        
            learning = f"Revised: {verification_result.action} needs adjustment because {analysis}"
            transformation["learning"] = learning
            transformation["change_applied"] = "revision"
            logger.info(f"     Revision needed: {analysis[:50]}...")
        
        self.transformation_log.append(transformation)
        
        #         
        growth = GrowthRecord(
            before_state=WorldSnapshot(timestamp=time.time() - 1),
            after_state=WorldSnapshot(timestamp=time.time()),
            learning=transformation["learning"],
            growth_amount=0.1 if verification_result.success else 0.05
        )
        
        return growth


class LifeCycle:
    """
             
    
                          
    
               ,                    .
    """
    
    def __init__(self, memory=None, resonance=None, internal_universe=None, tension_field=None):
        self.perception = PerceptionModule(memory, resonance)
        self.verification = VerificationModule()
        self.transformation = SelfTransformationModule(internal_universe, memory)
        
        # [SELF GOVERNANCE]            
        self.governance = SelfGovernance() if SelfGovernance else None
        
        # [Phase 25] TensionField for Field-based reinforcement
        self.tension_field = tension_field
        
        self.cycle_count = 0
        self.growth_history: List[GrowthRecord] = []
        self.current_snapshot: Optional[WorldSnapshot] = None
        
        logger.info("  LifeCycle initialized - continuous flow enabled")
        if self.governance:
            logger.info("     SelfGovernance connected for meaningful evaluation")
        if self.tension_field:
            logger.info("     TensionField connected for Field Physics reinforcement")

        # [Phase 6] Predictive Mind
        self.predictive_mind = PredictiveMind() if PredictiveMind else None
        if self.predictive_mind:
            logger.info("     PredictiveMind connected for Cognitive Verification")
            
            # [Phase 7] Field-Mind Unification
            if self.tension_field:
                self.predictive_mind.connect_field(self.tension_field)
    
    def begin_cycle(self) -> WorldSnapshot:
        """       -          """
        self.current_snapshot = self.perception.take_snapshot()
        self.cycle_count += 1
        logger.info(f"  Cycle #{self.cycle_count} begins")
        return self.current_snapshot
    
    def complete_cycle(self, action: str, expected: str, actual: str) -> GrowthRecord:
        """
               -         
        
        1.         
        2.   
        3.      
        """
        # 1.         
        before = self.current_snapshot or self.perception.take_snapshot()
        after = self.perception.take_snapshot()
        change = self.perception.perceive_change(before, after)
        
        logger.info(f"      Perceived: energy  {change['energy_change']:.1f}, entropy  {change['entropy_change']:.1f}")
        
        # 2.   
        result = self.verification.verify(expected, actual, action)
        analysis = self.verification.analyze_gap(expected, actual)
        
        # 3.      
        growth = self.transformation.transform(result, analysis)
        self.growth_history.append(growth)
        
        # [Phase 25] Field Physics Reinforcement
        if self.tension_field and action:
            # Extract concept from action (e.g., "LEARN:Python" -> "Python")
            concept_id = action.split(":")[-1] if ":" in action else action
            
            if result.success:
                # Success   Deepen the gravity well (habit formation)
                self.tension_field.reinforce_well(concept_id, 0.1)
                logger.info(f"     Gravity Deepened: {concept_id} curvature +0.1")
            else:
                # Failure   Understand WHY it failed (Latent Causality)
                # "           ,         "
                
                # Find related concept (if any exists in the field)
                related_concept = "understanding"  # Default target
                
                # Assess latent causality: WHY is this connection impossible?
                if hasattr(self.tension_field, 'assess_latent_causality'):
                    diagnosis = self.tension_field.assess_latent_causality(
                        concept_a=concept_id,
                        concept_b=related_concept
                    )
                    
                    logger.info(f"     Latent Causality Analysis:")
                    logger.info(f"      Possible: {diagnosis.get('possible', False)}")
                    logger.info(f"      Diagnosis: {diagnosis.get('diagnosis', 'Unknown')}")
                    logger.info(f"      Prescription: {diagnosis.get('prescription', 'Unknown')}")
                    
                    if diagnosis.get('energy_needed', 0) > 0:
                        logger.info(f"      Energy Needed: {diagnosis.get('energy_needed', 0):.2f}")
                    
                    if diagnosis.get('bridge_candidates'):
                        logger.info(f"      Bridge Concepts: {diagnosis.get('bridge_candidates', [])}")
                    
                    # Store the diagnosis for accumulation
                    growth.diagnosis = diagnosis
                
                # Increase charge for retry (tension accumulation)
                self.tension_field.charge_concept(concept_id, 0.3)
                logger.info(f"     Tension Charged: {concept_id} energy +0.3")
        
        # [SELF GOVERNANCE]                
        if self.governance:
            self.governance.adjust_after_result(
                action=action,
                success=result.success,
                learning=growth.learning
            )
            
            #            (10      )
            if self.cycle_count % 10 == 0:
                logger.info(self.governance.get_achievement_report())
        
        logger.info(f"     Growth: {growth.learning[:50]}...")
        logger.info(f"  Cycle #{self.cycle_count} complete")
        
        # [Phase 6] Predictive Verification
        if self.predictive_mind:
            #                
            if "LEARN" in action:
                concept = action.split(":")[-1]
                # 1.      
                hyp = self.predictive_mind.formulate_hypothesis(concept, ["Understanding", "Utility", "Connection"])
                if hyp:
                    # 2.       (          )
                    #                   ,           
                    verify_result = self.predictive_mind.verify_hypothesis(hyp, actual)
                    logger.info(f"     Predictive Verification: {verify_result}")
        
        return growth
    
    def get_total_growth(self) -> float:
        """        """
        return sum(g.growth_amount for g in self.growth_history)
    
    def get_status(self) -> Dict[str, Any]:
        """     """
        return {
            "cycle_count": self.cycle_count,
            "total_growth": self.get_total_growth(),
            "verification_success_rate": self._get_success_rate(),
            "transformation_count": len(self.transformation.transformation_log)
        }
    
    def _get_success_rate(self) -> float:
        """      """
        if not self.verification.history:
            return 0.0
        successes = sum(1 for r in self.verification.history if r.success)
        return successes / len(self.verification.history)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("="*60)
    print("  Life Cycle Demo")
    print("                         ")
    print("="*60)
    
    cycle = LifeCycle()
    
    # Cycle 1:       
    print("\n--- Cycle 1:    ---")
    cycle.begin_cycle()
    growth1 = cycle.complete_cycle(
        action="LEARN:Python",
        expected="Python knowledge increased",
        actual="Python knowledge increased"
    )
    
    # Cycle 2:       
    print("\n--- Cycle 2:         ---")
    cycle.begin_cycle()
    growth2 = cycle.complete_cycle(
        action="CONNECT:User",
        expected="User responded",
        actual="No response received"
    )
    
    # Cycle 3:         
    print("\n--- Cycle 3:          ---")
    cycle.begin_cycle()
    growth3 = cycle.complete_cycle(
        action="CONNECT:User:retry",
        expected="User engaged",
        actual="User engaged successfully"
    )
    
    #      
    print("\n" + "="*60)
    print("  Status:")
    status = cycle.get_status()
    print(f"   Cycles: {status['cycle_count']}")
    print(f"   Total Growth: {status['total_growth']:.2f}")
    print(f"   Success Rate: {status['verification_success_rate']:.1%}")
    print("="*60)
