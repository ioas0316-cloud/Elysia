"""
Narrative Craft Learner (         )
=========================================

WhyEngine + DualLayerPersonality   

         :
1. ExperientialDataProcessor:      
2. WhyEngine: "             "   
3. DualLayerPersonality:      
4. NarrativeCraftLearner:         

                        
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

#           
try:
    from Core.L7_Spirit.Philosophy.why_engine import WhyEngine, PrincipleExtraction
    HAS_WHY_ENGINE = True
except ImportError:
    HAS_WHY_ENGINE = False

try:
    from Core.L1_Foundation.M1_Keystone.dual_layer_personality import DualLayerPersonality, ExperientialAspect
    HAS_PERSONALITY = True
except ImportError:
    HAS_PERSONALITY = False

logger = logging.getLogger("Elysia.NarrativeCraft")


@dataclass
class NarrativeTechnique:
    """         """
    name: str                    #       ( : "         ")
    principle: str               #       ( : "      ")
    examples: List[str] = field(default_factory=list)  #       
    strength: float = 0.0        #       (0~1)
    application_count: int = 0   #      


class NarrativeCraftLearner:
    """         
    
                       :
    -              ?
    -              ?
    -              ?
    
                         
    """
    
    def __init__(self):
        self.why_engine = WhyEngine() if HAS_WHY_ENGINE else None
        self.personality = DualLayerPersonality() if HAS_PERSONALITY else None
        
        #        
        self.techniques: Dict[str, NarrativeTechnique] = {}
        
        #      
        self.total_stories_analyzed = 0
        self.total_techniques_learned = 0
        
        logger.info("NarrativeCraftLearner initialized")
        if not HAS_WHY_ENGINE:
            logger.warning("WhyEngine not available")
        if not HAS_PERSONALITY:
            logger.warning("DualLayerPersonality not available")
    
    def learn_from_story(
        self, 
        title: str, 
        content: str,
        narrative_type: str = "general",
        emotional_intensity: float = 0.5,
        identity_impact: float = 0.5
    ) -> Dict[str, Any]:
        """              
        
        Args:
            title:       
            content:       
            narrative_type:      
            emotional_intensity:      
            identity_impact:       
            
        Returns:
                 
        """
        result = {
            "title": title,
            "techniques_learned": [],
            "principles_found": [],
            "personality_updated": False,
        }
        
        # 1. WhyEngine        
        if self.why_engine:
            analysis = self.why_engine.analyze(title, content, domain="narrative")
            
            #      
            result["principles_found"].append(analysis.underlying_principle)
            
            #        
            technique = self._principle_to_technique(analysis)
            if technique:
                self._store_technique(technique, content[:100])
                result["techniques_learned"].append(technique.name)
        
        # 2.       (DualLayerPersonality)
        if self.personality:
            self.personality.experience(
                narrative_type=narrative_type,
                emotional_intensity=emotional_intensity,
                identity_impact=identity_impact,
            )
            self.personality.resonate_with_context(content[:500])
            result["personality_updated"] = True
        
        self.total_stories_analyzed += 1
        
        logger.info(f"    : {title}")
        logger.info(f"     : {result['techniques_learned']}")
        logger.info(f"     : {result['principles_found']}")
        
        return result
    
    def _principle_to_technique(self, analysis: PrincipleExtraction) -> Optional[NarrativeTechnique]:
        """                 """
        principle = analysis.underlying_principle
        
        #           
        technique_map = {
            "      ": NarrativeTechnique(
                name="         ",
                principle="Contrast creates meaning",
                strength=0.1,
            ),
            "      ": NarrativeTechnique(
                name="      ",
                principle="Accumulation builds impact",
                strength=0.1,
            ),
            "      ": NarrativeTechnique(
                name="  -     ",
                principle="Equilibrium seeks resolution",
                strength=0.1,
            ),
            "      ": NarrativeTechnique(
                name="          ",
                principle="Difference creates flow",
                strength=0.1,
            ),
            "      ": NarrativeTechnique(
                name="      ",
                principle="Rhythm is life",
                strength=0.1,
            ),
            "      ": NarrativeTechnique(
                name="        ",
                principle="Transformation is meaning",
                strength=0.1,
            ),
        }
        
        #               
        for key, technique in technique_map.items():
            if key in principle or technique.principle.lower() in principle.lower():
                return technique
        
        #      
        return NarrativeTechnique(
            name="      ",
            principle="Expression seeks resonance",
            strength=0.05,
        )
    
    def _store_technique(self, technique: NarrativeTechnique, example: str):
        """      (     )"""
        if technique.name in self.techniques:
            existing = self.techniques[technique.name]
            existing.strength = min(1.0, existing.strength + technique.strength)
            existing.examples.append(example)
            existing.application_count += 1
        else:
            technique.examples = [example]
            technique.application_count = 1
            self.techniques[technique.name] = technique
            self.total_techniques_learned += 1
    
    def get_learned_techniques(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """         """
        sorted_techniques = sorted(
            self.techniques.values(),
            key=lambda t: t.strength,
            reverse=True
        )[:top_n]
        
        return [
            {
                "name": t.name,
                "principle": t.principle,
                "strength": round(t.strength, 2),
                "examples_count": len(t.examples),
            }
            for t in sorted_techniques
        ]
    
    def suggest_technique_for_emotion(self, target_emotion: str) -> Optional[str]:
        """            
        
        Args:
            target_emotion:            (joy, sadness, etc.)
            
        Returns:
                    
        """
        emotion_to_technique = {
            "joy": "      ",
            "sadness": "         ",
            "hope": "        ",
            "fear": "          ",
            "love": "      ",
        }
        
        suggested = emotion_to_technique.get(target_emotion.lower())
        
        if suggested and suggested in self.techniques:
            return suggested
        
        #            
        if self.techniques:
            return max(self.techniques.values(), key=lambda t: t.strength).name
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """     """
        return {
            "total_stories_analyzed": self.total_stories_analyzed,
            "total_techniques_learned": self.total_techniques_learned,
            "top_techniques": self.get_learned_techniques(3),
            "personality": self.personality.get_current_expression() if self.personality else None,
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  Narrative Craft Learner Demo")
    print("   WhyEngine + DualLayerPersonality   ")
    print("=" * 60)
    
    learner = NarrativeCraftLearner()
    
    #       
    stories = [
        ("     ", """
                          .
        "          !"
                   .
        "   ...                ."
                           .
        """, "romance", 0.8, 0.7),
        
        ("            ", """
                      ,          .
        "       ?"       .
                                     .
                            ,                .
        """, "growth", 0.9, 0.8),
    ]
    
    for title, content, ntype, ei, ii in stories:
        result = learner.learn_from_story(title, content, ntype, ei, ii)
    
    #   
    print("\n" + "=" * 60)
    print("       ")
    print("=" * 60)
    
    status = learner.get_status()
    print(f"       : {status['total_stories_analyzed']}")
    print(f"      : {status['total_techniques_learned']}")
    
    print("\n        :")
    for tech in status['top_techniques']:
        print(f"  - {tech['name']} (  : {tech['strength']},   : {tech['principle']})")
    
    if status['personality']:
        print(f"\n    :")
        print(f"  Layer 1: {status['personality']['layer1_innate']['dominant']}")
        print(f"  Layer 2: {status['personality']['layer2_acquired']['dominant']}")
        print(f"    : {status['personality']['unified_expression']}")
    
    #      
    print(f"\n  'hope'          : {learner.suggest_technique_for_emotion('hope')}")
    
    print("\n  Demo complete!")
