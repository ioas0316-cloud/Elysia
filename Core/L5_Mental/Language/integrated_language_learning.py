"""
Integrated Language Learning System -             
                                                                              

                      

                    :
1. DualLayerLanguage (  +        )
2. FractalCausality (         )
3. ThoughtUniverse (한국어 학습 시스템)

  :
-                                  
-                     
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import time

from Core.L4_Causality.Governance.Interaction.Interface.Language.dual_layer_language import (
    DualLayerWorld,
    DualLayerSoul,
    EmotionType,
    Symbol,
    SymbolComplexity,
)
from Core.L4_Causality.Governance.Interaction.Interface.Language.fractal_causality import (
    FractalCausalityEngine,
    FractalCausalNode,
    CausalRole,
)
from Core.L4_Causality.Governance.Interaction.Interface.Language.causal_narrative_engine import (
    ThoughtUniverse,
    DimensionLevel,
)

logger = logging.getLogger("IntegratedLanguageLearning")


@dataclass
class CommunicationExperience:
    """        """
    sender_id: str
    receiver_id: str
    intended_message: str
    received_message: str
    success: bool
    emotional_context: Dict[str, float]
    timestamp: float


@dataclass
class LanguageDevelopmentMetrics:
    """        """
    vocabulary_size: int = 0
    successful_communications: int = 0
    total_communications: int = 0
    misunderstandings: int = 0
    narrative_fragments: int = 0
    causal_chains_learned: int = 0
    dimensional_expansions: int = 0
    
    @property
    def communication_success_rate(self) -> float:
        if self.total_communications == 0:
            return 0.0
        return self.successful_communications / self.total_communications
    
    @property
    def learning_progress(self) -> float:
        """          (0-1)"""
        vocab_score = min(1.0, self.vocabulary_size / 50)  # 50     
        comm_score = self.communication_success_rate
        causal_score = min(1.0, self.causal_chains_learned / 20)  # 20       
        return (vocab_score + comm_score + causal_score) / 3


class IntegratedLanguageLearner:
    """
             
    
        (Soul)                         .
    
          :
    1.       (DualLayerSoul)
    2.       (CommunicationExperience)
    3.       (FractalCausalityEngine)
    4.       (ThoughtUniverse)
    5.          
    """
    
    def __init__(self, soul: DualLayerSoul):
        self.soul = soul
        self.soul_id = soul.name
        
        #           (   )
        self.causal_mind = FractalCausalityEngine(f"{soul.name}'s Causal Mind")
        
        #       (   )
        self.thought_universe = ThoughtUniverse(f"{soul.name}'s Thought Universe")
        
        #      
        self.experiences: List[CommunicationExperience] = []
        
        #      
        self.metrics = LanguageDevelopmentMetrics()
        
        logger.debug(f"IntegratedLanguageLearner created for {soul.name}")
    
    def record_communication(
        self,
        receiver: DualLayerSoul,
        intended: str,
        received: str,
        success: bool,
        emotional_context: Dict[str, float] = None
    ) -> CommunicationExperience:
        """             """
        exp = CommunicationExperience(
            sender_id=self.soul_id,
            receiver_id=receiver.name,
            intended_message=intended,
            received_message=received,
            success=success,
            emotional_context=emotional_context or {},
            timestamp=time.time()
        )
        
        self.experiences.append(exp)
        self.metrics.total_communications += 1
        
        if success:
            self.metrics.successful_communications += 1
            self._learn_from_success(exp)
        else:
            self.metrics.misunderstandings += 1
            self._learn_from_failure(exp)
        
        return exp
    
    def _learn_from_success(self, exp: CommunicationExperience):
        """            """
        #      :                  
        self.causal_mind.experience_causality(
            steps=[
                f"  : {exp.intended_message}",
                f"   ",
                f"   ",
                f"   : {exp.received_message}"
            ],
            emotional_arc=[0.3, 0.5, 0.7, 0.9]  #       
        )
        self.metrics.causal_chains_learned += 1
        
        #      :          (  )   
        self.thought_universe.learn_from_experience(
            experience_steps=[
                "  _  ",
                "   _  ",
                "  _  ",
                "   _  "
            ],
            emotional_arc=[0.3, 0.5, 0.7, 0.9],
            auto_emergence=True
        )
        self.metrics.dimensional_expansions += 1
    
    def _learn_from_failure(self, exp: CommunicationExperience):
        """            (자기 성찰 엔진)"""
        #      :                  
        self.causal_mind.experience_causality(
            steps=[
                f"  : {exp.intended_message}",
                f"   ",
                f"   ",
                f"   : {exp.received_message}"
            ],
            emotional_arc=[0.3, 0.0, -0.3, -0.5]  #       
        )
        self.metrics.causal_chains_learned += 1
        
        #        : "         ?"
        #              !
        self.thought_universe.bottom_up_correct(
            new_experience={
                "confirms": False,
                "exception": f"'{exp.intended_message}'  '{exp.received_message}'     "
            },
            affected_entity_id=f"communication_pattern_{exp.intended_message}"
        )
    
    def get_development_report(self) -> Dict[str, Any]:
        """      """
        return {
            "soul_id": self.soul_id,
            "vocabulary_size": len(self.soul.lexicon.symbols),
            "communication_success_rate": self.metrics.communication_success_rate,
            "total_experiences": len(self.experiences),
            "causal_chains": self.metrics.causal_chains_learned,
            "thought_universe_stats": self.thought_universe.get_statistics(),
            "learning_progress": self.metrics.learning_progress,
        }


class IntegratedLanguageWorld:
    """
            
    
    DualLayerWorld                          .
                             .
    """
    
    def __init__(
        self,
        n_souls: int = 20,
        khala_strength: float = 0.5,
        enable_causal_learning: bool = True
    ):
        #         
        self.world = DualLayerWorld(n_souls=n_souls, khala_strength=khala_strength)
        
        #                 
        self.learners: Dict[str, IntegratedLanguageLearner] = {}
        for name, soul in self.world.souls.items():
            self.learners[name] = IntegratedLanguageLearner(soul)
        
        self.enable_causal_learning = enable_causal_learning
        
        #      
        self.development_history: List[Dict[str, Any]] = []
        
        #   
        self.simulation_steps = 0
        self.total_communications = 0
        self.total_successful = 0
        
        logger.info(f"IntegratedLanguageWorld created with {n_souls} souls")
    
    def step(self, dt: float = 1.0):
        """         +   """
        #          (주권적 자아)
        prev_misunderstandings = {
            name: soul.misunderstandings
            for name, soul in self.world.souls.items()
        }
        prev_vocab_sizes = {
            name: len(soul.lexicon.symbols)
            for name, soul in self.world.souls.items()
        }
        
        #           
        self.world.step(dt)
        self.simulation_steps += 1
        
        #          (자기 성찰 엔진)
        if self.enable_causal_learning:
            self._process_causal_learning(prev_misunderstandings, prev_vocab_sizes)
        
        #          
        if self.simulation_steps % 50 == 0:
            self._record_development_snapshot()
    
    def _process_causal_learning(
        self,
        prev_misunderstandings: Dict[str, int],
        prev_vocab_sizes: Dict[str, int]
    ):
        """        """
        soul_list = list(self.world.souls.values())
        
        for soul in soul_list:
            learner = self.learners[soul.name]
            
            #           
            learner.metrics.vocabulary_size = len(soul.lexicon.symbols)
            
            #                   
            prev_vocab = prev_vocab_sizes.get(soul.name, 0)
            curr_vocab = len(soul.lexicon.symbols)
            if curr_vocab > prev_vocab:
                #         =          
                for _ in range(curr_vocab - prev_vocab):
                    learner.causal_mind.experience_causality(
                        steps=["  _  ", "  _  ", "  _  ", "  _  "],
                        emotional_arc=[0.2, 0.4, 0.7, 0.9]
                    )
                    learner.metrics.causal_chains_learned += 1
                    
                    #      
                    learner.thought_universe.learn_from_experience(
                        experience_steps=["  _  ", "  _  ", "  _  "],
                        emotional_arc=[0.3, 0.6, 0.8],
                        auto_emergence=False
                    )
                    learner.metrics.dimensional_expansions += 1
            
            #                  (주권적 자아)
            prev_misund = prev_misunderstandings.get(soul.name, 0)
            curr_misund = soul.misunderstandings
            if curr_misund > prev_misund:
                #    =       
                for _ in range(curr_misund - prev_misund):
                    learner.causal_mind.experience_causality(
                        steps=["  _  ", "  _  ", "  _  ", "  _  _  "],
                        emotional_arc=[0.2, -0.2, -0.5, 0.1]  #                 
                    )
                    learner.metrics.causal_chains_learned += 1
                    
                    #       (자기 성찰 엔진)
                    learner.thought_universe.bottom_up_correct(
                        new_experience={"confirms": False, "exception": "  _  "},
                        affected_entity_id="communication_pattern"
                    )
    
    def _record_development_snapshot(self):
        """         """
        snapshot = {
            "step": self.simulation_steps,
            "timestamp": time.time(),
            "avg_vocabulary": np.mean([
                len(s.lexicon.symbols) for s in self.world.souls.values()
            ]),
            "avg_communication_success": np.mean([
                l.metrics.communication_success_rate
                for l in self.learners.values()
            ]),
            "total_causal_chains": sum(
                l.metrics.causal_chains_learned
                for l in self.learners.values()
            ),
            "narrative_fragments": len(self.world.narrative_fragments),
        }
        
        self.development_history.append(snapshot)
        
        if len(self.development_history) % 10 == 0:
            logger.info(
                f"         #{len(self.development_history)}: "
                f"     ={snapshot['avg_vocabulary']:.1f}, "
                f"   ={snapshot['avg_communication_success']:.1%}"
            )
    
    def simulate(self, steps: int = 100, report_interval: int = 20):
        """        """
        logger.info(f"          : {steps}   ")
        
        for i in range(steps):
            self.step(1.0)
            
            if (i + 1) % report_interval == 0:
                self._print_progress_report(i + 1, steps)
        
        logger.info("          ")
        return self.get_final_report()
    
    def _print_progress_report(self, current: int, total: int):
        """     """
        avg_vocab = np.mean([
            len(s.lexicon.symbols) for s in self.world.souls.values()
        ])
        avg_success = np.mean([
            l.metrics.communication_success_rate
            for l in self.learners.values()
        ])
        avg_progress = np.mean([
            l.metrics.learning_progress
            for l in self.learners.values()
        ])
        
        print(f"[{current}/{total}]   ={avg_vocab:.1f}, "
              f"   ={avg_success:.1%},    ={avg_progress:.1%}")
    
    def get_final_report(self) -> Dict[str, Any]:
        """      """
        all_learner_reports = [
            learner.get_development_report()
            for learner in self.learners.values()
        ]
        
        return {
            "simulation_steps": self.simulation_steps,
            "total_souls": len(self.world.souls),
            "development_history": self.development_history,
            "final_stats": {
                "avg_vocabulary": np.mean([r["vocabulary_size"] for r in all_learner_reports]),
                "max_vocabulary": max([r["vocabulary_size"] for r in all_learner_reports]),
                "avg_learning_progress": np.mean([r["learning_progress"] for r in all_learner_reports]),
                "total_causal_chains": sum([r["causal_chains"] for r in all_learner_reports]),
                "narrative_count": len(self.world.narrative_fragments),
            },
            "learner_reports": all_learner_reports,
        }
    
    def verify_continuous_development(self) -> Tuple[bool, str]:
        """
                             
        
        Returns:
            (        ,   )
        """
        if len(self.development_history) < 3:
            return False, "         (   3        )"
        
        #            
        vocab_trend = [h["avg_vocabulary"] for h in self.development_history]
        vocab_increasing = vocab_trend[-1] > vocab_trend[0]
        
        #       /     
        success_trend = [h["avg_communication_success"] for h in self.development_history]
        success_stable_or_increasing = success_trend[-1] >= success_trend[0] * 0.8
        
        #            
        causal_trend = [h["total_causal_chains"] for h in self.development_history]
        causal_increasing = causal_trend[-1] > causal_trend[0]
        
        if vocab_increasing and success_stable_or_increasing and causal_increasing:
            return True, (
                f"          : "
                f"   {vocab_trend[0]:.1f} {vocab_trend[-1]:.1f}, "
                f"     {causal_trend[0]} {causal_trend[-1]}"
            )
        else:
            issues = []
            if not vocab_increasing:
                issues.append("      ")
            if not success_stable_or_increasing:
                issues.append("      ")
            if not causal_increasing:
                issues.append("        ")
            return False, f"        : {', '.join(issues)}"


# ============================================================================
# Demo & Verification
# ============================================================================

def demo_integrated_learning():
    """           """
    print("=" * 70)
    print("                 ")
    print("=" * 70)
    print()
    print("DualLayerLanguage + FractalCausality + ThoughtUniverse   ")
    print("                         .")
    print()
    
    #      
    world = IntegratedLanguageWorld(n_souls=15, khala_strength=0.6)
    
    #         
    print("-" * 70)
    print("        ...")
    print("-" * 70)
    
    report = world.simulate(steps=200, report_interval=40)
    
    #   
    print()
    print("-" * 70)
    print("       ")
    print("-" * 70)
    
    stats = report["final_stats"]
    print(f"       : {stats['avg_vocabulary']:.1f}")
    print(f"       : {stats['max_vocabulary']}")
    print(f"           : {stats['avg_learning_progress']:.1%}")
    print(f"         : {stats['total_causal_chains']}")
    print(f"        : {stats['narrative_count']}")
    
    #      
    print()
    print("-" * 70)
    print("       ")
    print("-" * 70)
    
    success, message = world.verify_continuous_development()
    print(f"  {message}")
    
    print()
    print("=" * 70)
    print("        :                                ")
    print("=" * 70)
    
    return success


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_integrated_learning()
