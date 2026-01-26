"""
Language Bridge System (         )
=========================================

Soul         Elysia  MemeticField         .

  :
1. Soul             
2. MemeticField          
3.             Soul      
4.           

      :
- Soul (  )   MemeticField (  )
-                 
-               

"           ,            "
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("LanguageBridge")


# =============================================================================
# 1.        (Pattern Collector) - Soul        
# =============================================================================

@dataclass
class EmergentPattern:
    """
    Soul         
    """
    source_soul_id: int
    meaning_vector: np.ndarray  # 8D      
    symbol_type: str  # "entity", "action", "state", "relation"
    occurrence_count: int
    korean_projection: Optional[str] = None
    timestamp: float = 0.0


class PatternCollector:
    """
       Soul              
    """
    
    def __init__(self):
        self.patterns: List[EmergentPattern] = []
        self.pattern_clusters: Dict[str, List[EmergentPattern]] = defaultdict(list)
    
    def collect(self, pattern: EmergentPattern):
        """     """
        self.patterns.append(pattern)
        self.pattern_clusters[pattern.symbol_type].append(pattern)
    
    def get_common_patterns(self, min_occurrence: int = 3) -> List[EmergentPattern]:
        """           """
        return [p for p in self.patterns if p.occurrence_count >= min_occurrence]
    
    def cluster_similar_patterns(self, threshold: float = 0.8) -> List[List[EmergentPattern]]:
        """              """
        clusters = []
        used = set()
        
        for i, p1 in enumerate(self.patterns):
            if i in used:
                continue
            
            cluster = [p1]
            used.add(i)
            
            for j, p2 in enumerate(self.patterns):
                if j in used:
                    continue
                
                #       
                similarity = self._similarity(p1.meaning_vector, p2.meaning_vector)
                if similarity > threshold:
                    cluster.append(p2)
                    used.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    @staticmethod
    def _similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """      """
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0.0
        return dot / norm


# =============================================================================
# 2.      (Structurer) -            
# =============================================================================

@dataclass
class StructuredConcept:
    """
    MemeticField             
    """
    concept_id: str
    vector_64d: np.ndarray  # 64D    (8D   64D   )
    korean_word: str
    english_word: Optional[str] = None
    category: str = "emergent"  # "emergent", "core", "derived"
    source_patterns: List[int] = field(default_factory=list)  #           


class PatternStructurer:
    """
                       
    
    8D         64D      
    """
    
    def __init__(self):
        # 8D   64D       (     )
        self.expansion_matrix = self._init_expansion_matrix()
        
        #           
        self.registered_concepts: Dict[str, StructuredConcept] = {}
        
        #            
        self.category_bases = {
            "entity": np.array([1, 0, 0, 0, 0, 0, 0, 0]),
            "action": np.array([0, 1, 0, 0, 0, 0, 0, 0]),
            "state": np.array([0, 0, 1, 0, 0, 0, 0, 0]),
            "relation": np.array([0, 0, 0, 1, 0, 0, 0, 0]),
        }
    
    def _init_expansion_matrix(self) -> np.ndarray:
        """
        8D   64D          
        
              : 8D  8       64D
        """
        #   :          +      
        matrix = np.zeros((64, 8))
        
        for i in range(8):
            #   8D     8   64D       
            start = i * 8
            for j in range(8):
                #       (     )
                matrix[start + j, i] = 1.0 if j == i else 0.3
        
        return matrix
    
    def expand_to_64d(self, vector_8d: np.ndarray, symbol_type: str) -> np.ndarray:
        """
        8D     64D    
        
                        
        """
        #      
        expanded = self.expansion_matrix @ vector_8d
        
        #            
        if symbol_type in self.category_bases:
            base = self.category_bases[symbol_type]
            #    8            
            expanded[:8] += base * 0.5
        
        #    
        norm = np.linalg.norm(expanded)
        if norm > 0:
            expanded /= norm
        
        return expanded
    
    def structure_pattern(self, pattern: EmergentPattern) -> StructuredConcept:
        """
                        
        """
        # 64D      
        vector_64d = self.expand_to_64d(pattern.meaning_vector, pattern.symbol_type)
        
        #    ID   
        concept_id = f"em_{pattern.symbol_type}_{hash(tuple(pattern.meaning_vector)) % 10000}"
        
        #       (              )
        korean_word = pattern.korean_projection or self._generate_korean(pattern)
        
        return StructuredConcept(
            concept_id=concept_id,
            vector_64d=vector_64d,
            korean_word=korean_word,
            category="emergent",
            source_patterns=[id(pattern)]
        )
    
    def _generate_korean(self, pattern: EmergentPattern) -> str:
        """                """
        v = pattern.meaning_vector
        
        #            
        max_idx = np.argmax(np.abs(v))
        max_val = v[max_idx]
        
        #          
        dimension_words = {
            0: ("  ", "   "),  #   
            1: ("  ", "   "),   #   
            2: (" ", "  "),       #   
            3: ("  ", "  "),     #   
            4: ("   ", " "),     #    
            5: ("  ", "  "),     #   
            6: ("  ", "  "),     #  /  
            7: ("   ", "   "), #   
        }
        
        pos, neg = dimension_words.get(max_idx, (" ", " "))
        return pos if max_val > 0 else neg


# =============================================================================
# 3.         (Feedback Generator) - Soul      
# =============================================================================

@dataclass
class LanguageFeedback:
    """
    Soul           
    """
    concept_id: str
    korean_word: str
    category: str  # "word", "phrase", "sentence", "paragraph"
    structure_info: Dict[str, Any]  #          
    similar_concepts: List[str]  #        
    usage_examples: List[str]  #      


class FeedbackGenerator:
    """
    MemeticField      Soul           
    """
    
    def __init__(self):
        #          
        self.grammar_structures = {
            "entity": {
                "can_be_subject": True,
                "can_be_object": True,
                "particles": [" ", " ", " ", " ", " ", " "],
            },
            "action": {
                "can_be_predicate": True,
                "conjugations": [" ", "  ", "  "],
            },
            "state": {
                "can_be_predicate": True,
                "can_modify": True,
                "conjugations": [" ", " "],
            },
            "relation": {
                "connects": True,
                "particles": [" ", " ", "  ", "  ", "  "],
            },
        }
    
    def generate_feedback(self, concept: StructuredConcept, 
                         similar_concepts: List[str] = None) -> LanguageFeedback:
        """             """
        
        #         (   ID        )
        concept_type = "entity"
        for t in ["entity", "action", "state", "relation"]:
            if t in concept.concept_id:
                concept_type = t
                break
        
        #         
        structure_info = self.grammar_structures.get(concept_type, {})
        
        #         
        examples = self._generate_examples(concept.korean_word, concept_type)
        
        return LanguageFeedback(
            concept_id=concept.concept_id,
            korean_word=concept.korean_word,
            category="word",
            structure_info=structure_info,
            similar_concepts=similar_concepts or [],
            usage_examples=examples
        )
    
    def _generate_examples(self, word: str, concept_type: str) -> List[str]:
        """        """
        examples = []
        
        if concept_type == "entity":
            examples = [
                f"{word}    ",
                f"{word}    ",
                f"{word}    ",
            ]
        elif concept_type == "action":
            examples = [
                f"   {word}",
                f"    {word}",
            ]
        elif concept_type == "state":
            examples = [
                f"{word}     ",
                f"{word}   ",
            ]
        elif concept_type == "relation":
            examples = [
                f" {word}  ",
                f"  {word}   ",
            ]
        
        return examples


# =============================================================================
# 4.       (Language Bridge) -       
# =============================================================================

class LanguageBridge:
    """
    Soul   Elysia      
    
           :
    1. Soul        
    2.               
    3. MemeticField    
    4.          Soul     
    """
    
    def __init__(self, memetic_field=None):
        self.collector = PatternCollector()
        self.structurer = PatternStructurer()
        self.feedback_gen = FeedbackGenerator()
        
        # MemeticField    (   )
        self.memetic_field = memetic_field
        
        #   
        self.total_patterns_collected = 0
        self.total_concepts_registered = 0
        self.total_feedbacks_sent = 0
    
    def receive_from_soul(self, soul_id: int, meaning_vector: np.ndarray,
                         symbol_type: str, occurrence_count: int,
                         korean_projection: str = None) -> Optional[LanguageFeedback]:
        """
        Soul             
        
        Returns:     (   )
        """
        # 1.      
        pattern = EmergentPattern(
            source_soul_id=soul_id,
            meaning_vector=meaning_vector,
            symbol_type=symbol_type,
            occurrence_count=occurrence_count,
            korean_projection=korean_projection
        )
        self.collector.collect(pattern)
        self.total_patterns_collected += 1
        
        # 2.                 
        if occurrence_count >= 5:
            concept = self.structurer.structure_pattern(pattern)
            
            # 3. MemeticField     (   )
            if self.memetic_field is not None:
                self._register_to_memetic_field(concept)
            
            self.total_concepts_registered += 1
            
            # 4.       
            feedback = self.feedback_gen.generate_feedback(concept)
            self.total_feedbacks_sent += 1
            
            logger.info(f"      : Soul {soul_id}   {concept.korean_word}")
            
            return feedback
        
        return None
    
    def _register_to_memetic_field(self, concept: StructuredConcept):
        """MemeticField       """
        try:
            from Core.L6_Structure.Wave.infinite_hyperquaternion import InfiniteHyperQuaternion
            
            # 64D     InfiniteHyperQuaternion     
            vector = InfiniteHyperQuaternion(64, concept.vector_64d)
            
            # MemeticField    
            self.memetic_field.add_concept(concept.concept_id, vector)
            
            logger.info(f"MemeticField    : {concept.concept_id} ({concept.korean_word})")
            
        except ImportError:
            logger.warning("InfiniteHyperQuaternion       ")
        except Exception as e:
            logger.warning(f"MemeticField      : {e}")
    
    def process_batch(self) -> List[LanguageFeedback]:
        """
                      
        
                                    
        """
        feedbacks = []
        
        #      
        clusters = self.collector.cluster_similar_patterns(threshold=0.7)
        
        for cluster in clusters:
            #               
            avg_vector = np.mean([p.meaning_vector for p in cluster], axis=0)
            total_occurrences = sum(p.occurrence_count for p in cluster)
            
            #                
            projections = [p.korean_projection for p in cluster if p.korean_projection]
            best_projection = max(set(projections), key=projections.count) if projections else None
            
            #         
            unified = EmergentPattern(
                source_soul_id=-1,  #      
                meaning_vector=avg_vector,
                symbol_type=cluster[0].symbol_type,
                occurrence_count=total_occurrences,
                korean_projection=best_projection
            )
            
            #          
            concept = self.structurer.structure_pattern(unified)
            feedback = self.feedback_gen.generate_feedback(concept)
            feedbacks.append(feedback)
            
            logger.info(f"       : {len(cluster)}       {concept.korean_word}")
        
        return feedbacks
    
    def get_statistics(self) -> Dict[str, Any]:
        """  """
        return {
            "total_patterns": self.total_patterns_collected,
            "total_concepts": self.total_concepts_registered,
            "total_feedbacks": self.total_feedbacks_sent,
            "pattern_clusters": len(self.collector.cluster_similar_patterns()),
        }


# =============================================================================
# 5.   
# =============================================================================

def demo():
    """        """
    print("=" * 60)
    print("  Language Bridge Demo - Soul   Elysia")
    print("=" * 60)
    
    bridge = LanguageBridge()
    
    #      :    Soul        
    test_patterns = [
        #        (   Soul     )
        (0, np.array([0.8, 0.5, 0.1, 0.0, 0.3, 0.4, 0.6, 0.3]), "state", 10, "    "),
        (1, np.array([0.7, 0.4, 0.2, 0.1, 0.2, 0.3, 0.5, 0.2]), "state", 8, "   "),
        (2, np.array([0.9, 0.6, 0.0, 0.0, 0.4, 0.5, 0.7, 0.4]), "state", 12, "    "),
        
        #      
        (0, np.array([0.2, 0.3, 0.3, 0.2, 0.9, 0.3, 0.8, 0.5]), "entity", 15, "  "),
        (1, np.array([0.1, 0.2, 0.2, 0.1, 0.8, 0.2, 0.7, 0.4]), "entity", 10, "  "),
        
        #       
        (2, np.array([0.3, 0.4, 0.2, 0.9, 0.2, 0.8, 0.4, 0.9]), "action", 7, "   "),
        (0, np.array([0.2, 0.3, 0.1, 0.8, 0.1, 0.7, 0.3, 0.8]), "action", 5, "  "),
    ]
    
    print("\n         ...")
    for soul_id, vector, sym_type, count, proj in test_patterns:
        feedback = bridge.receive_from_soul(soul_id, vector, sym_type, count, proj)
        if feedback:
            print(f"       : {feedback.korean_word} ({feedback.category})")
            print(f"       : {feedback.usage_examples[0] if feedback.usage_examples else '-'}")
    
    print("\n        (     )...")
    batch_feedbacks = bridge.process_batch()
    for fb in batch_feedbacks:
        print(f"         : {fb.korean_word}")
    
    print("\n    :")
    stats = bridge.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)
    print("       ")
    print("=" * 60)


if __name__ == "__main__":
    demo()
