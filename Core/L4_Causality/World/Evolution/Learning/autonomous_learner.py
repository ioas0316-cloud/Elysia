"""
Autonomous Learning Loop (자기 성찰 엔진)
========================================

        :

1. WhyEngine:      
2. MetacognitiveAwareness: "        ?"
   -             
   -                

3. ExternalExplorer:      
   -    KB   
   -     
   -         

4. ConceptCrystallization:       
   -               
   - "       "   "  "

5. Learn:   
   - MetacognitiveAwareness    
   -             " ,   !"

  :
                         
"""

import logging
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path
from Core.L4_Causality.World.Evolution.Learning.Learning.hierarchical_learning import Domain

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Core.L7_Spirit.Philosophy.why_engine import WhyEngine
from Core.L5_Mental.Intelligence.Cognition.metacognitive_awareness import MetacognitiveAwareness, KnowledgeState
from Core.L5_Mental.Intelligence.Cognition.external_explorer import ExternalExplorer

logger = logging.getLogger("Elysia.AutonomousLearning")


class AutonomousLearner:
    """      
    
    "         ,     ,    "
    
      :
    1.    (  )
    2.       (   ?     ?)
    3.       
    4.       
    5.      
    """
    
    def __init__(self):
        self.why_engine = WhyEngine()
        self.metacognition = self.why_engine.metacognition  #   
        self.explorer = ExternalExplorer()
        
        #      
        self.total_experiences = 0
        self.learned_from_self = 0    #        
        self.learned_from_external = 0  #          
        self.pending_learning = 0       #         
        
        logger.info("AutonomousLearner initialized")
    
    def experience(
        self, 
        content: str, 
        subject: str = "unknown",
        domain: str = "narrative"
    ) -> Dict[str, Any]:
        """        
        
        Args:
            content:        (   )
            subject:   /   
            domain:   
            
        Returns:
                 
        """
        self.total_experiences += 1
        
        result = {
            "subject": subject,
            "knowledge_state": None,
            "learned_concept": None,
            "needs_human_help": False,
            "question_for_human": None,
            "potential_knowledge": None
        }
        
        # 0. Load Connections
        try:
            from Core.L5_Mental.Intelligence.Memory_Linguistics.Memory.potential_causality import PotentialCausalityStore
            potential_store = PotentialCausalityStore()
        except ImportError:
            potential_store = None
        
        try:
            from Core.L4_Causality.World.Evolution.Learning.Learning.hierarchical_learning import HierarchicalKnowledgeGraph
            kg = HierarchicalKnowledgeGraph()
        except ImportError:
            kg = None

        # 1. WhyEngine      (            )
        analysis = self.why_engine.analyze(subject, content, domain)
        
        # 2.         
        if "[     ]" in analysis.underlying_principle:
            #       !
            result["knowledge_state"] = "unknown"
            
            #            
            if potential_store:
                pk = potential_store.store(
                    subject=subject,
                    definition=content[:200],
                    source="autonomous_experience"
                )
                result["potential_knowledge"] = pk.to_dict()
                logger.info(f"     Stored as potential: {subject} (freq={pk.frequency:.2f})")
            
            # 3.      
            wave = self.why_engine._text_to_wave(content)
            exploration = self.explorer.explore(
                question=analysis.underlying_principle.replace("[     ] ", ""),
                wave_signature=wave,
                context=content[:200],
            )
            
            if exploration.answer:
                #          !
                result["knowledge_state"] = "learned"
                self.learned_from_external += 1
                
                #            (  )
                if potential_store:
                    potential_store.store(subject, content, f"external_source:{exploration.source.value}")
                    #       
                    crystallized = potential_store.crystallize(subject)
                    if crystallized and kg:
                         #              
                         wave = self.why_engine._text_to_wave(content)
                         kg.add_concept(
                             name=crystallized['concept'],
                             domain=Domain(domain) if domain in [d.value for d in Domain] else Domain.PHILOSOPHY, #      
                             definition=crystallized['definition'],
                             principle=analysis.underlying_principle,  #    (Why -   )
                             application=analysis.how_works,           #    (How -   )
                             purpose=f"Autonomously learned via {exploration.source.value}",
                             wave_signature=wave  #         
                         )
                         result["learned_concept"] = crystallized['concept']
                         logger.info(f"     Crystallized and added to KG: {crystallized['concept']}")

                #          (       )
                if self.metacognition:
                    self.metacognition.learn_from_external(
                        pattern_id=self._get_pattern_id(wave),
                        answer=exploration.answer,
                        source=exploration.source.value,
                    )
                
            else:
                #            
                result["needs_human_help"] = True
                result["question_for_human"] = exploration.question
                self.pending_learning += 1
                
                logger.info(f"          : {exploration.question}")
        
        else:
            #      !
            result["knowledge_state"] = "known"
            result["learned_concept"] = analysis.underlying_principle
            self.learned_from_self += 1
            
            logger.info(f"          : {analysis.underlying_principle}")
        
        return result
    
    def _get_pattern_id(self, wave: Dict[str, float]) -> str:
        """   ID   """
        import hashlib
        import json
        return hashlib.md5(json.dumps(wave, sort_keys=True).encode()).hexdigest()[:8]
    
    def learn_from_human(self, question: str, answer: str, concept_name: str):
        """        
        
        "  ,        "
        """
        self.explorer.answer_from_user(question, answer, concept_name)
        self.learned_from_external += 1
        self.pending_learning -= 1
        
        logger.info(f"          : '{concept_name}'")
    
    def get_pending_questions(self) -> List[Dict[str, Any]]:
        """            """
        return self.explorer.get_pending_questions()
    
    def get_learned_concepts(self) -> List[Dict[str, Any]]:
        """      """
        return self.explorer.get_crystallized_concepts()
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """     """
        return {
            "total_experiences": self.total_experiences,
            "learned_from_self": self.learned_from_self,
            "learned_from_external": self.learned_from_external,
            "pending_learning": self.pending_learning,
            "known_concepts": len(self.get_learned_concepts()),
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  Autonomous Learning Loop Demo")
    print("   \"          \"")
    print("=" * 60)
    
    learner = AutonomousLearner()
    
    #    1:         
    print("\n[   1]         :")
    result1 = learner.experience(
        content="""
                         .
                        ,
                   .
                 .
        """,
        subject="      ",
    )
    print(f"     : {result1['knowledge_state']}")
    print(f"       : {result1['learned_concept']}")
    
    #    2:      
    print("\n[   2]      :")
    result2 = learner.experience(
        content="""
                        .
                         .
                .
        """,
        subject="     ",
    )
    print(f"     : {result2['knowledge_state']}")
    print(f"       : {result2['learned_concept']}")
    
    #    3:       
    print("\n[   3]       :")
    result3 = learner.experience(
        content="""
                  .
          ,         .
                      .
        "       ?"
                                      .
        """,
        subject="          ",
    )
    print(f"     : {result3['knowledge_state']}")
    if result3['needs_human_help']:
        print(f"             : {result3['question_for_human']}")
    
    #          
    pending = learner.get_pending_questions()
    if pending:
        print("\n[     ]          :")
        q = pending[0]['question']
        learner.learn_from_human(
            question=q,
            answer="                         ",
            concept_name="  "
        )
    
    #   
    print("\n" + "=" * 60)
    print("       :")
    stats = learner.get_learning_stats()
    print(f"       : {stats['total_experiences']}")
    print(f"          : {stats['learned_from_self']}")
    print(f"          : {stats['learned_from_external']}")
    print(f"        : {stats['pending_learning']}")
    
    print("\n        :")
    for concept in learner.get_learned_concepts():
        print(f"     {concept['name']}: {concept['definition'][:30]}...")
    
    print("\n  Demo complete!")
