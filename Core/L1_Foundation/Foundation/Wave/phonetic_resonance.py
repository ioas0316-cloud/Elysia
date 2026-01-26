"""
Phonetic Resonance Layer
========================

"             "

                          ,
'      (Vector)'           (  )       .

  :
- Rhyme    (       )
-       (        )
-        

[NEW 2025-12-16] Hybrid Architecture Layer 2
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from Core.L1_Foundation.Foundation.hangul_physics import HangulPhysicsEngine, Tensor3D

@dataclass
class ResonanceField:
    """            """
    text: str
    vectors: List[Tensor3D]
    average_roughness: float
    average_tension: float
    
    def vector_matrix(self) -> np.ndarray:
        """(N, 3)      """
        return np.array([[v.x, v.y, v.z] for v in self.vectors])

class PhoneticResonanceEngine:
    def __init__(self):
        self.physics = HangulPhysicsEngine()
        
    def text_to_field(self, text: str) -> ResonanceField:
        """            """
        vectors = [self.physics.get_phonetic_vector(char) for char in text]
        
        #      
        if not vectors:
            return ResonanceField(text, [], 0.0, 0.0)
            
        avg_rough = sum(v.roughness() for v in vectors) / len(vectors)
        # Tension is mapped to Z in get_phonetic_vector
        avg_tens = sum(v.z for v in vectors) / len(vectors)
        
        return ResonanceField(
            text=text,
            vectors=vectors,
            average_roughness=avg_rough,
            average_tension=avg_tens
        )

    def calculate_resonance(self, text1: str, text2: str) -> float:
        """
                       (   )   
        0.0 ~ 1.0
        """
        field1 = self.text_to_field(text1)
        field2 = self.text_to_field(text2)
        
        # 1.          
        len1, len2 = len(field1.vectors), len(field2.vectors)
        if len1 == 0 or len2 == 0: return 0.0
        
        min_len = min(len1, len2)
        
        # 2.            (     )
        similarities = []
        for i in range(min_len):
            v1 = field1.vectors[i]
            v2 = field2.vectors[i]
            
            # Dot product
            dot = (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z)
            mag1 = v1.magnitude()
            mag2 = v2.magnitude()
            
            if mag1 == 0 or mag2 == 0:
                sim = 1.0 if mag1 == mag2 else 0.0
            else:
                sim = dot / (mag1 * mag2)
            
            similarities.append(sim)
            
        # 3.         
        base_resonance = sum(similarities) / min_len
        
        # 4.   (Roughness/Tension)       
        feel_diff = abs(field1.average_roughness - field2.average_roughness) + \
                    abs(field1.average_tension - field2.average_tension)
                    
        feel_factor = 1.0 / (1.0 + feel_diff) #            
        
        return (base_resonance * 0.7) + (feel_factor * 0.3)

    def find_rhymes(self, target_word: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """  (      )   """
        scores = []
        for word in candidates:
            score = self.calculate_resonance(target_word, word)
            scores.append((word, score))
            
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

# Singleton
_engine = None
def get_resonance_engine():
    global _engine
    if _engine is None:
        _engine = PhoneticResonanceEngine()
    return _engine
