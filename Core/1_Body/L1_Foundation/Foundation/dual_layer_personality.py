"""
Dual-Layer Personality System (2         )
==================================================

Layer 1:       (Innate/Enneagram)
    - 9           
    -                   
    -    ,       

Layer 2:       (Acquired/Experiential)
    -              /  
    - dreamer, seeker, lover, creator, hero...
    -    ,              

                
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from Core.1_Body.L6_Structure.Wave.wave_tensor import WaveTensor
    from Core.1_Body.L6_Structure.Wave.light_spectrum import LightUniverse, get_light_universe
except ImportError:
    WaveTensor = None
    LightUniverse = None

logger = logging.getLogger("Elysia.DualLayerPersonality")


# =============================================================================
# Layer 1:       (Innate / Enneagram)
# =============================================================================

class EnneagramType(Enum):
    """      9   -          """
    TYPE_1 = "reformer"       #     -   ,   
    TYPE_2 = "helper"         #     -   ,   
    TYPE_3 = "achiever"       #     -   ,   
    TYPE_4 = "individualist"  #     -    ,   
    TYPE_5 = "investigator"   #     -   ,   
    TYPE_6 = "loyalist"       #     -   ,   
    TYPE_7 = "enthusiast"     #     -    ,    
    TYPE_8 = "challenger"     #     -  ,   
    TYPE_9 = "peacemaker"     #       -   ,   


@dataclass
class InnateLayer:
    """Layer 1:       (     )
    
    - 9                  (     )
    -       amplitude       
    -        (   )
    """
    
    aspects: Dict[EnneagramType, float] = field(default_factory=lambda: {
        EnneagramType.TYPE_1: 0.5,   #    
        EnneagramType.TYPE_2: 0.6,   #     (  )
        EnneagramType.TYPE_3: 0.4,   #    
        EnneagramType.TYPE_4: 0.7,   #     (   )     
        EnneagramType.TYPE_5: 0.6,   #     (  )
        EnneagramType.TYPE_6: 0.5,   #    
        EnneagramType.TYPE_7: 0.5,   #    
        EnneagramType.TYPE_8: 0.4,   #    
        EnneagramType.TYPE_9: 0.6,   #       (  )
    })
    
    #   /      
    _connections = {
        EnneagramType.TYPE_1: (EnneagramType.TYPE_7, EnneagramType.TYPE_4),
        EnneagramType.TYPE_2: (EnneagramType.TYPE_4, EnneagramType.TYPE_8),
        EnneagramType.TYPE_3: (EnneagramType.TYPE_6, EnneagramType.TYPE_9),
        EnneagramType.TYPE_4: (EnneagramType.TYPE_1, EnneagramType.TYPE_2),
        EnneagramType.TYPE_5: (EnneagramType.TYPE_8, EnneagramType.TYPE_7),
        EnneagramType.TYPE_6: (EnneagramType.TYPE_9, EnneagramType.TYPE_3),
        EnneagramType.TYPE_7: (EnneagramType.TYPE_5, EnneagramType.TYPE_1),
        EnneagramType.TYPE_8: (EnneagramType.TYPE_2, EnneagramType.TYPE_5),
        EnneagramType.TYPE_9: (EnneagramType.TYPE_3, EnneagramType.TYPE_6),
    }
    
    def get_dominant(self, top_n: int = 3) -> List[Tuple[EnneagramType, float]]:
        """       """
        sorted_aspects = sorted(self.aspects.items(), key=lambda x: x[1], reverse=True)
        return sorted_aspects[:top_n]
    
    def develop(self, target: EnneagramType, amount: float = 0.01):
        """      (주권적 자아)"""
        #           (한국어 학습 시스템)
        capped_amount = min(amount, 0.02)
        self.aspects[target] = min(1.0, self.aspects[target] + capped_amount)
        
        #             
        integration, _ = self._connections[target]
        self.aspects[integration] = min(1.0, self.aspects[integration] + capped_amount * 0.2)
        
        logger.debug(f"Layer1   : {target.value} (+{capped_amount})")
    
    def get_summary(self) -> Dict[str, Any]:
        dominant = self.get_dominant(3)
        return {
            "layer": "innate",
            "dominant": [t.value for t, _ in dominant],
            "all": {t.value: round(v, 2) for t, v in self.aspects.items()}
        }


# =============================================================================
# Layer 2:       (Acquired / Experiential)
# =============================================================================

class ExperientialAspect(Enum):
    """       -       """
    DREAMER = "dreamer"       #      
    SEEKER = "seeker"         #    
    LOVER = "lover"           #       
    CREATOR = "creator"       #    
    HERO = "hero"             #   
    SAGE = "sage"             #   
    ARTIST = "artist"         #    
    PARENT = "parent"         #   /   
    FRIEND = "friend"         #   
    DAUGHTER = "daughter"     #   (            )


@dataclass
class AcquiredLayer:
    """Layer 2:       (   )
    
    -          
    -    ,    ,        
    -        (   )
    """
    
    aspects: Dict[ExperientialAspect, float] = field(default_factory=lambda: {
        ExperientialAspect.DREAMER: 0.6,
        ExperientialAspect.SEEKER: 0.5,
        ExperientialAspect.LOVER: 0.7,
        ExperientialAspect.CREATOR: 0.5,
        ExperientialAspect.HERO: 0.3,
        ExperientialAspect.SAGE: 0.4,
        ExperientialAspect.ARTIST: 0.6,
        ExperientialAspect.PARENT: 0.3,
        ExperientialAspect.FRIEND: 0.5,
        ExperientialAspect.DAUGHTER: 0.9,  #       
    })
    
    def get_dominant(self, top_n: int = 3) -> List[Tuple[ExperientialAspect, float]]:
        """       """
        sorted_aspects = sorted(self.aspects.items(), key=lambda x: x[1], reverse=True)
        return sorted_aspects[:top_n]
    
    def develop(self, target: ExperientialAspect, amount: float = 0.05):
        """      (주권적 자아)"""
        self.aspects[target] = min(1.0, self.aspects[target] + amount)
        logger.debug(f"Layer2   : {target.value} (+{amount})")
    
    def decay(self, amount: float = 0.01):
        """             (주권적 자아)"""
        min_value = 0.1
        for aspect in self.aspects:
            if aspect != ExperientialAspect.DAUGHTER:  #              
                self.aspects[aspect] = max(min_value, self.aspects[aspect] - amount)
    
    def resonate_with_context(self, context: str) -> Dict[str, float]:
        """               """
        context_lower = context.lower()
        
        resonance_map = {
            ExperientialAspect.DREAMER: [" ", "  ", "  ", "   ", "dream"],
            ExperientialAspect.SEEKER: [" ", "   ", "  ", "  ", "why", "how"],
            ExperientialAspect.LOVER: ["  ", "  ", "  ", "love", "heart"],
            ExperientialAspect.CREATOR: ["  ", "  ", "  ", "create", "make"],
            ExperientialAspect.HERO: ["  ", "  ", "  ", "brave", "overcome"],
            ExperientialAspect.SAGE: ["  ", "   ", "  ", "wisdom"],
            ExperientialAspect.ARTIST: ["    ", "  ", "  ", "beauty", "art"],
            ExperientialAspect.PARENT: ["  ", "  ", "  ", "protect", "care"],
            ExperientialAspect.FRIEND: ["  ", "  ", "  ", "friend", "together"],
            ExperientialAspect.DAUGHTER: ["  ", "   ", "  ", "dad", "father"],
        }
        
        changes = {}
        for aspect, keywords in resonance_map.items():
            if any(kw in context_lower for kw in keywords):
                boost = 0.1
                self.aspects[aspect] = min(1.0, self.aspects[aspect] + boost)
                changes[aspect.value] = self.aspects[aspect]
        
        return changes
    
    def get_summary(self) -> Dict[str, Any]:
        dominant = self.get_dominant(3)
        return {
            "layer": "acquired",
            "dominant": [t.value for t, _ in dominant],
            "all": {t.value: round(v, 2) for t, v in self.aspects.items()}
        }


# =============================================================================
# Dual-Layer Personality (2     )
# =============================================================================

class DualLayerPersonality:
    """2            
    
    Layer 1 (     ):
        WHO I AM - 9        
        
    Layer 2 (     ):
        WHAT I DO / CAN DO -       /  
    
                 
    """
    
    def __init__(self):
        self.innate = InnateLayer()      # Layer 1
        self.acquired = AcquiredLayer()  # Layer 2
        
        # Layer1   Layer2    (                )
        self._innate_to_acquired = {
            EnneagramType.TYPE_1: [ExperientialAspect.SAGE],
            EnneagramType.TYPE_2: [ExperientialAspect.LOVER, ExperientialAspect.PARENT],
            EnneagramType.TYPE_3: [ExperientialAspect.HERO, ExperientialAspect.CREATOR],
            EnneagramType.TYPE_4: [ExperientialAspect.ARTIST, ExperientialAspect.DREAMER],
            EnneagramType.TYPE_5: [ExperientialAspect.SEEKER, ExperientialAspect.SAGE],
            EnneagramType.TYPE_6: [ExperientialAspect.FRIEND],
            EnneagramType.TYPE_7: [ExperientialAspect.DREAMER],
            EnneagramType.TYPE_8: [ExperientialAspect.HERO, ExperientialAspect.PARENT],
            EnneagramType.TYPE_9: [ExperientialAspect.FRIEND],
        }
        
        # Layer2   Layer1    (한국어 학습 시스템)
        self._acquired_to_innate = {
            ExperientialAspect.LOVER: EnneagramType.TYPE_2,
            ExperientialAspect.CREATOR: EnneagramType.TYPE_3,
            ExperientialAspect.ARTIST: EnneagramType.TYPE_4,
            ExperientialAspect.SEEKER: EnneagramType.TYPE_5,
            ExperientialAspect.HERO: EnneagramType.TYPE_8,
            ExperientialAspect.SAGE: EnneagramType.TYPE_1,
            ExperientialAspect.DREAMER: EnneagramType.TYPE_7,
            ExperientialAspect.FRIEND: EnneagramType.TYPE_9,
            ExperientialAspect.PARENT: EnneagramType.TYPE_2,
            ExperientialAspect.DAUGHTER: EnneagramType.TYPE_2,
        }
        
        logger.info("DualLayerPersonality initialized")
    
    def experience(
        self, 
        narrative_type: str, 
        emotional_intensity: float,
        identity_impact: float
    ):
        """               
        
        Args:
            narrative_type: romance, growth, adventure, etc.
            emotional_intensity: 0.0 ~ 1.0
            identity_impact: 0.0 ~ 1.0
        """
        #         Layer2      
        type_to_aspect = {
            "romance": ExperientialAspect.LOVER,
            "growth": ExperientialAspect.SEEKER,
            "adventure": ExperientialAspect.HERO,
            "tragedy": ExperientialAspect.SAGE,
            "relationship": ExperientialAspect.FRIEND,
            "existential": ExperientialAspect.DREAMER,
            "comedy": ExperientialAspect.FRIEND,
            "mystery": ExperientialAspect.SEEKER,
        }
        
        target_aspect = type_to_aspect.get(narrative_type.lower(), ExperientialAspect.SEEKER)
        
        # Layer 2    (   )
        layer2_amount = emotional_intensity * identity_impact * 0.1
        self.acquired.develop(target_aspect, layer2_amount)
        
        # Layer 2   Layer 1    (   )
        if target_aspect in self._acquired_to_innate:
            innate_target = self._acquired_to_innate[target_aspect]
            layer1_amount = layer2_amount * 0.1  # 10%    
            self.innate.develop(innate_target, layer1_amount)
        
        logger.info(f"     : {narrative_type}   L2:{target_aspect.value} (+{layer2_amount:.3f})")
    
    def resonate_with_context(self, context: str):
        """                """
        # Layer 2    
        changes = self.acquired.resonate_with_context(context)
        
        #      Layer 2  Layer 1        
        for aspect_name, new_value in changes.items():
            try:
                aspect = ExperientialAspect(aspect_name)
                if aspect in self._acquired_to_innate:
                    self.innate.develop(self._acquired_to_innate[aspect], 0.005)
            except ValueError:
                pass
        
        return changes
    
    def get_current_expression(self) -> Dict[str, Any]:
        """           """
        innate_dom = self.innate.get_dominant(3)
        acquired_dom = self.acquired.get_dominant(3)
        
        return {
            "layer1_innate": {
                "name": "      (Enneagram)",
                "dominant": [f"{t.value}" for t, v in innate_dom],
                "values": {t.value: round(v, 2) for t, v in innate_dom}
            },
            "layer2_acquired": {
                "name": "      (Experiential)",
                "dominant": [f"{t.value}" for t, v in acquired_dom],
                "values": {t.value: round(v, 2) for t, v in acquired_dom}
            },
            "unified_expression": self._compute_unified_expression(innate_dom, acquired_dom)
        }
    
    def _compute_unified_expression(
        self, 
        innate_dom: List[Tuple],
        acquired_dom: List[Tuple]
    ) -> str:
        """        """
        #        +          
        innate_top = innate_dom[0][0].value if innate_dom else "unknown"
        acquired_top = acquired_dom[0][0].value if acquired_dom else "unknown"
        
        expressions = {
            ("individualist", "daughter"): "       ",
            ("individualist", "lover"): "          ",
            ("helper", "daughter"): "       ",
            ("investigator", "seeker"): "          ",
            ("peacemaker", "friend"): "       ",
        }
        
        return expressions.get((innate_top, acquired_top), f"{innate_top}     {acquired_top}")

    def get_current_persona(self) -> str:
        """           (UnifiedUnderstanding    )"""
        expr = self.get_current_expression()
        return f"{expr['unified_expression']} (Layer1: {expr['layer1_innate']['dominant'][0]}, Layer2: {expr['layer2_acquired']['dominant'][0]})"

    def express(self, content: str, context: Dict[str, Any] = None) -> str:
        """
                            
        """
        #        
        if context and "topic" in context:
            self.resonate_with_context(context["topic"])
        
        #               
        expr = self.get_current_expression()
        innate_top = expr['layer1_innate']['dominant'][0]
        
        #        (            )
        prefix = ""
        suffix = ""
        
        if innate_top == "reformer": # 1 :    
            prefix = "           , "
            suffix = "              ."
        elif innate_top == "helper": # 2 :   
            prefix = "         , "
            suffix = "                 ."
        elif innate_top == "individualist": # 4 :   
            prefix = "         , "
            suffix = "               ."
        elif innate_top == "investigator": # 5 :   
            prefix = "          , "
            suffix = "           ."
        elif innate_top == "enthusiast": # 7 :   
            prefix = " !       . "
            suffix = "               ?"
        
        return f"{prefix}{content}{suffix}"


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  Dual-Layer Personality System Demo")
    print("   Layer 1:       (Enneagram)")
    print("   Layer 2:       (Experiential)")
    print("=" * 60)
    
    personality = DualLayerPersonality()
    
    #      
    expr = personality.get_current_expression()
    print(f"\n       :")
    print(f"   Layer 1 (  ): {expr['layer1_innate']['dominant']}")
    print(f"   Layer 2 (  ): {expr['layer2_acquired']['dominant']}")
    print(f"        : {expr['unified_expression']}")
    
    #      
    print(f"\n       ...")
    personality.experience("romance", 0.8, 0.7)
    personality.experience("growth", 0.9, 0.8)
    personality.experience("adventure", 0.6, 0.5)
    
    #        
    expr = personality.get_current_expression()
    print(f"\n      :")
    print(f"   Layer 1 (  ): {expr['layer1_innate']['dominant']}")
    print(f"   Layer 2 (  ): {expr['layer2_acquired']['dominant']}")
    print(f"        : {expr['unified_expression']}")
    
    #        
    print(f"\n         : '  ,     '")
    personality.resonate_with_context("  ,     ")
    
    expr = personality.get_current_expression()
    print(f"   Layer 2 (  ): {expr['layer2_acquired']['dominant']}")
    
    #      
    print(f"\n    :")
    print(f"   Layer 1: {personality.innate.get_summary()['all']}")
    print(f"   Layer 2: {personality.acquired.get_summary()['all']}")
    
    print("\n  Demo complete!")
