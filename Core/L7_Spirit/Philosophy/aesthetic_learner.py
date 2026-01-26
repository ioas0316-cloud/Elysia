"""
Aesthetic Learner (주권적 자아)
==============================

"              "

            (   ,    )      ,
            '      '        .

Process (Flow of Learning):
1. Input:       
2. Perception: ConceptFormation        
3. Sensing: SensoryCortex    (Qualia)    
4. Recording: MemoryStream  '  (Observation)'      
5. Reflection: ReflectionLoop       (  /  )
"""

import os
import re
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Cognitive Core
from Core.L5_Mental.Intelligence.Cognitive.memory_stream import get_memory_stream, ExperienceType
from Core.L5_Mental.Intelligence.Cognitive.concept_formation import get_concept_formation
from Core.L5_Mental.Intelligence.Cognitive.sensory_cortex import get_sensory_cortex
from Core.L7_Spirit.Philosophy.aesthetic_principles import Medium

logger = logging.getLogger("AestheticLearner")

@dataclass
class AestheticAnalysis:
    """         (For internal temporary transport)"""
    source: str
    source_type: str
    title: Optional[str] = None
    concepts_found: List[str] = None
    qualia_feeling: str = ""
    why_beautiful: str = ""
    
    def __post_init__(self):
        if self.concepts_found is None:
            self.concepts_found = []

class AestheticLearner:
    """
    Cognitive Aesthetic Learner
    
    Connects the diverse external world to the internal Cognitive Core.
    """
    
    def __init__(self, data_dir: str = "data/aesthetic"):
        print("  AestheticLearner (Cognitive) Initialized.")
        
        # Core Organs
        self.memory = get_memory_stream()
        self.concepts = get_concept_formation()
        self.senses = get_sensory_cortex()
        
        # External Tools (Optional)
        self._check_dependencies()
        
    def _check_dependencies(self):
        # (Same as before - keeping for future expansion)
        pass
        
    # =========================================================================
    # Text Analysis (Flow of Learning Implementation)
    # =========================================================================
    
    def analyze_text(self, text: str, title: Optional[str] = None) -> AestheticAnalysis:
        """
               ,    ,      .
        
        Flow:
        1. Extract Features (Literary)
        2. Form Concepts (What is this about?)
        3. Feel Qualia (What does it feel like?)
        4. Store Memory (Observation)
        """
        logger.info(f"  Analyzing Text: {title or text[:20]}...")
        
        # 1. Extract Literary Features (Analysis)
        features = self._analyze_literary_features(text)
        
        # 2. Concept Formation (Interpretation)
        #                  '  '       .
        #          features               ,               
        extracted_concepts = self._extract_concepts_from_features(features)
        
        # 3. Sensory Experience (Qualia)
        #                   '     '.
        qualia_desc = "Neutral"
        if extracted_concepts:
            primary_concept = extracted_concepts[0]
            #               (Learning by encountering)
            if not self.concepts.get_concept(primary_concept):
                self.concepts.learn_concept(primary_concept, text[:50])
                
            qualia = self.senses.feel_concept(primary_concept)
            qualia_desc = qualia["description"]
            
        # 4. Construct 'Why Beautiful' (Rationalization)
        why = f"This text has strong {', '.join(features.keys())}. It feels {qualia_desc}."
        
        # 5. Record to Memory (Learning)
        self.memory.add_experience(
            exp_type=ExperienceType.OBSERVATION,
            score={
                "intent": "analyze_beauty", 
                "concepts": extracted_concepts
            },
            performance={
                "content": text[:100], # Keep brief
                "features": features
            },
            sound={
                "aesthetic_score": sum(features.values()) * 10, # Mock score
                "qualia": qualia_desc,
                "analysis": why
            },
            tags=["text", "art_analysis"]
        )
        
        return AestheticAnalysis(
            source=title or "text_snippet",
            source_type="text",
            title=title,
            concepts_found=extracted_concepts,
            qualia_feeling=qualia_desc,
            why_beautiful=why
        )

    def _analyze_literary_features(self, text: str) -> Dict[str, float]:
        """          (Simplified from original)"""
        features = {}
        
        # Rhythm (Comma count as proxy for rhythm change)
        commas = text.count(',') + text.count('.')
        if commas > 2:
            features["Rhythm"] = min(commas / 10.0, 1.0)
            
        # Emotion (Simple keyword)
        if "sad" in text.lower() or "sluggish" in text.lower():
            features["Sadness"] = 0.8
        if "joy" in text.lower() or "bright" in text.lower():
            features["Joy"] = 0.8
            
        return features
        
    def _extract_concepts_from_features(self, features: Dict) -> List[str]:
        # Feature names are essentially concepts here
        return list(features.keys())

    # =========================================================================
    # Fallback / Placeholder methods for other media
    # =========================================================================
    # ... (Other methods would follow similar refactoring)

#    
_learner_instance: Optional[AestheticLearner] = None

def get_aesthetic_learner() -> AestheticLearner:
    global _learner_instance
    if _learner_instance is None:
        _learner_instance = AestheticLearner()
    return _learner_instance

if __name__ == "__main__":
    # Internal Test
    learner = get_aesthetic_learner()
    res = learner.analyze_text("The rain falls slowly, sad and cold.", "Rain Poem")
    print(f"Analysis: {res.why_beautiful}")
    print("Memory stored.")
