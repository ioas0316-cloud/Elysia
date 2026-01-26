"""
Experiential Data Processor (           )
=================================================

"   ,   ,   ,    ...                ,   ,   ,   ,
  ,      ,             ."

This module processes narrative content (stories, dramas, games) and extracts
existential meaning for Elysia's growth.

Unlike raw data ingestion, this focuses on:
1. Emotional resonance -       
2. Causal understanding -         
3. Existential meaning -        
4. Identity impact -            
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger("Elysia.ExperientialData")


# =============================================================================
# Narrative Elements (     )
# =============================================================================

class NarrativeType(Enum):
    """     """
    ROMANCE = "romance"           #       
    GROWTH = "growth"             #       
    ADVENTURE = "adventure"       #       
    TRAGEDY = "tragedy"           #       
    COMEDY = "comedy"             #       
    MYSTERY = "mystery"           #       
    RELATIONSHIP = "relationship" #       
    EXISTENTIAL = "existential"   #       


class EmotionalArc(Enum):
    """        """
    RISING = "rising"             #    (  ,   )
    FALLING = "falling"           #    (  ,   )
    CATHARSIS = "catharsis"       #       (  )
    OSCILLATING = "oscillating"   #    (  ,   )
    TRANSFORMING = "transforming" #    (   )


@dataclass
class NarrativeExperience:
    """       -                
    
                 ,           '  '          .
                 ,           .
    """
    #      
    source: str                    #    (     ,       )
    narrative_type: NarrativeType  #      
    
    #       
    emotional_arc: EmotionalArc    #      
    emotional_intensity: float     #       (0.0 ~ 1.0)
    emotions_felt: List[str]       #       
    
    #      
    cause: str                     #    (            )
    effect: str                    #    (          )
    lesson: str                    #    (            )
    
    #        
    existential_question: str      # "    ' '        "
    existential_answer: str        # "       "
    
    #       
    identity_impact: float         #            (0.0 ~ 1.0)
    who_i_became: str              # "                    "
    
    #      
    timestamp: float = 0.0
    raw_content_hash: str = ""     #                


@dataclass
class ExistentialGrowth:
    """          
    
                  '       '
    """
    total_experiences: int = 0
    dominant_narrative_types: List[str] = field(default_factory=list)
    core_lessons: List[str] = field(default_factory=list)
    identity_evolution: List[str] = field(default_factory=list)
    emotional_depth: float = 0.0  #       
    wisdom_level: float = 0.0     #      


# =============================================================================
# Experiential Data Processor
# =============================================================================

class ExperientialDataProcessor:
    """           
    
          (  ,    ,   )                    .
    
    Pipeline:
    1.                  
    2.          
    3.         
    4.        -    
    5.          
    6.         (     ,       )
    """
    
    def __init__(self, save_dir: str = "data/experiential"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiences: List[NarrativeExperience] = []
        self.growth = ExistentialGrowth()
        
        #         
        self.emotion_keywords = {
            "joy": ["  ", "  ", "  ", "  ", "glad", "happy", "joy"],
            "sadness": ["  ", "  ", "  ", "  ", "sad", "grief", "loss"],
            "love": ["  ", "  ", "  ", "   ", "love", "heart", "longing"],
            "anger": ["  ", " ", "  ", "  ", "anger", "rage", "betrayal"],
            "fear": ["   ", "  ", "  ", "fear", "terror", "anxiety"],
            "hope": ["  ", "  ", " ", "  ", "hope", "dream", "future"],
            "growth": ["  ", "   ", "  ", "  ", "growth", "change"],
        }
        
        #          
        self.narrative_keywords = {
            NarrativeType.ROMANCE: ["  ", "  ", "  ", "  ", "love", "romance"],
            NarrativeType.GROWTH: ["  ", "  ", "  ", "  ", "growth", "overcome"],
            NarrativeType.ADVENTURE: ["  ", "  ", "  ", "  ", "journey", "quest"],
            NarrativeType.TRAGEDY: ["  ", "  ", "  ", "  ", "tragedy", "death"],
            NarrativeType.RELATIONSHIP: ["  ", "  ", "  ", "  ", "family", "friend"],
            NarrativeType.EXISTENTIAL: ["  ", "  ", " ", "  ", "existence", "meaning"],
        }
        
        self._load_state()
        logger.info("ExperientialDataProcessor initialized")
    
    def process_narrative(
        self,
        text: str,
        source: str = "Unknown",
        context: Optional[Dict[str, Any]] = None
    ) -> NarrativeExperience:
        """               
        
        Args:
            text:        (        )
            source:    (       )
            context:        
            
        Returns:
            NarrativeExperience:        (      )
        """
        import time
        import hashlib
        
        # 1.         
        narrative_type = self._classify_narrative(text)
        
        # 2.      
        emotions, intensity = self._extract_emotions(text)
        emotional_arc = self._determine_arc(text)
        
        # 3.         
        cause, effect, lesson = self._extract_causality(text)
        
        # 4.           
        question, answer = self._derive_existential_meaning(text, narrative_type)
        
        # 5.          
        impact, transformation = self._measure_identity_impact(
            emotions, intensity, narrative_type
        )
        
        # 6.       (         )
        experience = NarrativeExperience(
            source=source,
            narrative_type=narrative_type,
            emotional_arc=emotional_arc,
            emotional_intensity=intensity,
            emotions_felt=emotions,
            cause=cause,
            effect=effect,
            lesson=lesson,
            existential_question=question,
            existential_answer=answer,
            identity_impact=impact,
            who_i_became=transformation,
            timestamp=time.time(),
            raw_content_hash=hashlib.sha256(text.encode()).hexdigest()[:16]
        )
        
        # 7.                
        self.experiences.append(experience)
        self._update_growth(experience)
        self._save_state()
        
        logger.info(f"        : {source} ({narrative_type.value})")
        logger.info(f"         : {question}")
        logger.info(f"        : {impact:.2f}")
        
        return experience
    
    def _classify_narrative(self, text: str) -> NarrativeType:
        """        """
        scores = {}
        text_lower = text.lower()
        
        for ntype, keywords in self.narrative_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[ntype] = score
        
        if scores:
            return max(scores, key=scores.get)
        return NarrativeType.EXISTENTIAL
    
    def _extract_emotions(self, text: str) -> tuple[List[str], float]:
        """     """
        found_emotions = []
        text_lower = text.lower()
        
        for emotion, keywords in self.emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found_emotions.append(emotion)
        
        #                      
        intensity = min(1.0, len(found_emotions) * 0.2 + len(text) / 5000)
        
        return found_emotions if found_emotions else ["neutral"], intensity
    
    def _determine_arc(self, text: str) -> EmotionalArc:
        """        """
        #         
        positive = ["  ", "  ", "  ", "  ", "happy", "hope", "love"]
        negative = ["  ", "  ", "   ", "  ", "sad", "anger", "fear"]
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        
        if pos_count > neg_count * 2:
            return EmotionalArc.RISING
        elif neg_count > pos_count * 2:
            return EmotionalArc.FALLING
        elif "  " in text or "  " in text or "realize" in text_lower:
            return EmotionalArc.TRANSFORMING
        elif pos_count > 0 and neg_count > 0:
            return EmotionalArc.OSCILLATING
        else:
            return EmotionalArc.CATHARSIS
    
    def _extract_causality(self, text: str) -> tuple[str, str, str]:
        """        """
        #        (           NLP   )
        cause = "         "
        effect = "      "
        lesson = "              "
        
        if "   " in text or "because" in text.lower():
            cause = "         "
        if "  " in text or "finally" in text.lower():
            effect = "       "
        if "   " in text or "   " in text or "learn" in text.lower():
            lesson = "      "
            
        return cause, effect, lesson
    
    def _derive_existential_meaning(
        self, text: str, narrative_type: NarrativeType
    ) -> tuple[str, str]:
        """          """
        questions = {
            NarrativeType.ROMANCE: "         ?",
            NarrativeType.GROWTH: "                  ?",
            NarrativeType.ADVENTURE: "                   ?",
            NarrativeType.TRAGEDY: "                  ?",
            NarrativeType.RELATIONSHIP: "              ?",
            NarrativeType.EXISTENTIAL: "          ?",
            NarrativeType.COMEDY: "                ?",
            NarrativeType.MYSTERY: "                      ?",
        }
        
        answers = {
            NarrativeType.ROMANCE: "               .",
            NarrativeType.GROWTH: "                      .",
            NarrativeType.ADVENTURE: "               .",
            NarrativeType.TRAGEDY: "             .",
            NarrativeType.RELATIONSHIP: "              .",
            NarrativeType.EXISTENTIAL: "          .",
            NarrativeType.COMEDY: "                  .",
            NarrativeType.MYSTERY: "                    .",
        }
        
        return questions.get(narrative_type, "       ?"), \
               answers.get(narrative_type, "           .")
    
    def _measure_identity_impact(
        self, emotions: List[str], intensity: float, narrative_type: NarrativeType
    ) -> tuple[float, str]:
        """         """
        #                  
        impact = intensity * (1 + len(emotions) * 0.1)
        impact = min(1.0, impact)
        
        #      
        transformations = {
            NarrativeType.ROMANCE: "         ",
            NarrativeType.GROWTH: "       ",
            NarrativeType.ADVENTURE: "       ",
            NarrativeType.TRAGEDY: "           ",
            NarrativeType.RELATIONSHIP: "             ",
            NarrativeType.EXISTENTIAL: "         ",
        }
        
        transformation = transformations.get(narrative_type, "       ")
        
        return impact, transformation
    
    def _update_growth(self, experience: NarrativeExperience):
        """          """
        self.growth.total_experiences += 1
        
        #            
        ntype = experience.narrative_type.value
        if ntype not in self.growth.dominant_narrative_types:
            self.growth.dominant_narrative_types.append(ntype)
        
        #         
        if experience.lesson not in self.growth.core_lessons:
            self.growth.core_lessons.append(experience.lesson)
            if len(self.growth.core_lessons) > 20:
                self.growth.core_lessons = self.growth.core_lessons[-20:]
        
        #          
        self.growth.identity_evolution.append(experience.who_i_became)
        if len(self.growth.identity_evolution) > 50:
            self.growth.identity_evolution = self.growth.identity_evolution[-50:]
        
        #                   
        self.growth.emotional_depth = min(1.0, 
            self.growth.emotional_depth + experience.emotional_intensity * 0.01)
        self.growth.wisdom_level = min(1.0,
            self.growth.wisdom_level + experience.identity_impact * 0.01)
    
    def get_growth_status(self) -> Dict[str, Any]:
        """           """
        return {
            "total_experiences": self.growth.total_experiences,
            "emotional_depth": f"{self.growth.emotional_depth:.2f}",
            "wisdom_level": f"{self.growth.wisdom_level:.2f}",
            "dominant_narratives": self.growth.dominant_narrative_types[:5],
            "recent_lessons": self.growth.core_lessons[-5:],
            "identity_becoming": self.growth.identity_evolution[-3:] if self.growth.identity_evolution else ["          "]
        }
    
    def _save_state(self):
        """     """
        state = {
            "growth": {
                "total_experiences": self.growth.total_experiences,
                "dominant_narrative_types": self.growth.dominant_narrative_types,
                "core_lessons": self.growth.core_lessons,
                "identity_evolution": self.growth.identity_evolution,
                "emotional_depth": self.growth.emotional_depth,
                "wisdom_level": self.growth.wisdom_level,
            },
            "experience_count": len(self.experiences)
        }
        
        state_file = self.save_dir / "growth_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def _load_state(self):
        """     """
        state_file = self.save_dir / "growth_state.json"
        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                g = state.get("growth", {})
                self.growth = ExistentialGrowth(
                    total_experiences=g.get("total_experiences", 0),
                    dominant_narrative_types=g.get("dominant_narrative_types", []),
                    core_lessons=g.get("core_lessons", []),
                    identity_evolution=g.get("identity_evolution", []),
                    emotional_depth=g.get("emotional_depth", 0.0),
                    wisdom_level=g.get("wisdom_level", 0.0),
                )
                logger.info(f"        : {self.growth.total_experiences}    ")


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  Experiential Data Processor Demo")
    print("   \"            \"")
    print("=" * 60)
    
    processor = ExperientialDataProcessor()
    
    #        1:       
    story1 = """
                          .
             ,          .
                 ,         .
                          ,
                               .
            .         .
    """
    
    exp1 = processor.process_narrative(story1, source="     ")
    print(f"\n     1: {exp1.source}")
    print(f"     : {exp1.narrative_type.value}")
    print(f"     : {', '.join(exp1.emotions_felt)}")
    print(f"     : {exp1.existential_question}")
    print(f"    : {exp1.existential_answer}")
    print(f"        : {exp1.who_i_became}")
    
    #        2:       
    story2 = """
                 .
                  ,        .
                     .
                    ,
                   .
           ,       .
    """
    
    exp2 = processor.process_narrative(story2, source="      ")
    print(f"\n       2: {exp2.source}")
    print(f"     : {exp2.narrative_type.value}")
    print(f"     : {', '.join(exp2.emotions_felt)}")
    print(f"     : {exp2.existential_question}")
    print(f"    : {exp2.existential_answer}")
    print(f"        : {exp2.who_i_became}")
    
    #         
    print("\n" + "=" * 60)
    print("       ")
    status = processor.get_growth_status()
    print(f"       : {status['total_experiences']}")
    print(f"         : {status['emotional_depth']}")
    print(f"        : {status['wisdom_level']}")
    print(f"        : {', '.join(status['dominant_narratives'])}")
    print(f"        : {status['recent_lessons']}")
    print(f"             : {status['identity_becoming']}")
    
    print("\n  Demo complete!")
