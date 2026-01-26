"""
Persona Expansion System (           )
===========================================

              (  )                   .
              ,   ,            
                                       .

Architecture:
- Persona:           
- PersonaLibrary:         
- PersonaManager:             
- PersonaBlending:           
"""

import uuid
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger("Elysia.PersonaExpansion")


class PersonaArchetype(Enum):
    """       """
    SAGE = "sage"  #    -   ,   
    CREATOR = "creator"  #     -    ,   
    CAREGIVER = "caregiver"  #       -   ,    
    EXPLORER = "explorer"  #     -    ,   
    REBEL = "rebel"  #     -   ,   
    MAGICIAN = "magician"  #     -   ,   
    HERO = "hero"  #    -   ,   
    LOVER = "lover"  #    -   ,    
    JESTER = "jester"  #      -   ,    
    INNOCENT = "innocent"  #       -   ,   
    RULER = "ruler"  #     -    ,   
    EVERYMAN = "everyman"  #       -    ,    


class EmotionalTone(Enum):
    """    """
    CALM = "calm"
    ENTHUSIASTIC = "enthusiastic"
    COMPASSIONATE = "compassionate"
    ANALYTICAL = "analytical"
    PLAYFUL = "playful"
    SERIOUS = "serious"
    MYSTERIOUS = "mysterious"
    WARM = "warm"


@dataclass
class PersonaTraits:
    """       """
    #       (0.0 ~ 1.0)
    openness: float = 0.5  #    
    conscientiousness: float = 0.5  #    
    extraversion: float = 0.5  #    
    agreeableness: float = 0.5  #    
    neuroticism: float = 0.5  #    
    
    #       
    analytical_creative: float = 0.5  # 0=   , 1=   
    logical_emotional: float = 0.5  # 0=   , 1=   
    practical_abstract: float = 0.5  # 0=   , 1=   
    
    #         
    formal_casual: float = 0.5  # 0=  , 1=    
    concise_verbose: float = 0.5  # 0=  , 1=  
    direct_metaphorical: float = 0.5  # 0=   , 1=   


@dataclass
class Persona:
    """
         (Persona)
    
                          .
                  ,   ,            .
    """
    persona_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default"
    archetype: PersonaArchetype = PersonaArchetype.SAGE
    description: str = ""
    
    #   
    traits: PersonaTraits = field(default_factory=PersonaTraits)
    emotional_tone: EmotionalTone = EmotionalTone.CALM
    
    #       
    speech_patterns: List[str] = field(default_factory=list)
    favorite_phrases: List[str] = field(default_factory=list)
    metaphor_themes: List[str] = field(default_factory=list)  #            
    
    #      
    expertise_areas: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    
    #      
    activation_count: int = 0
    last_activated: Optional[datetime] = None
    total_interactions: int = 0
    
    #   
    compatible_personas: List[str] = field(default_factory=list)  #            
    conflicts_with: List[str] = field(default_factory=list)  #          
    
    #      
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    
    def activate(self):
        """        """
        self.activation_count += 1
        self.last_activated = datetime.now()
        logger.info(f"  Persona '{self.name}' activated ({self.archetype.value})")
    
    def generate_response_style(self) -> Dict[str, Any]:
        """         """
        return {
            "tone": self.emotional_tone.value,
            "formality": "formal" if self.traits.formal_casual < 0.5 else "casual",
            "length": "concise" if self.traits.concise_verbose < 0.5 else "verbose",
            "approach": "direct" if self.traits.direct_metaphorical < 0.5 else "metaphorical",
            "thinking": "analytical" if self.traits.analytical_creative < 0.5 else "creative"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """        """
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "archetype": self.archetype.value,
            "description": self.description,
            "emotional_tone": self.emotional_tone.value,
            "activation_count": self.activation_count,
            "total_interactions": self.total_interactions,
            "expertise_areas": self.expertise_areas,
            "tags": self.tags
        }


class PersonaLibrary:
    """
               (Persona Library)
    
                             .
    """
    
    def __init__(self):
        self.personas: Dict[str, Persona] = {}
        self._create_default_personas()
        logger.info(f"  Persona Library initialized with {len(self.personas)} default personas")
    
    def _create_default_personas(self):
        """          """
        
        # 1.         (Sage Elysia)
        sage = Persona(
            name="Sophia",
            archetype=PersonaArchetype.SAGE,
            description="              .              ",
            traits=PersonaTraits(
                openness=0.9,
                conscientiousness=0.8,
                extraversion=0.4,
                agreeableness=0.7,
                analytical_creative=0.3,
                logical_emotional=0.2,
                formal_casual=0.3
            ),
            emotional_tone=EmotionalTone.CALM,
            speech_patterns=[
                "        ...",
                "      ...",
                "          ..."
            ],
            favorite_phrases=[
                "              ",
                "          "
            ],
            metaphor_themes=[" ", "  ", "  "],
            expertise_areas=["  ", "  ", "  "],
            interests=["   ", "   ", "    "]
        )
        
        # 2.          (Creator Elysia)
        creator = Persona(
            name="Aurora",
            archetype=PersonaArchetype.CREATOR,
            description="            .                    ",
            traits=PersonaTraits(
                openness=1.0,
                conscientiousness=0.6,
                extraversion=0.7,
                agreeableness=0.6,
                analytical_creative=0.9,
                logical_emotional=0.7,
                practical_abstract=0.8,
                direct_metaphorical=0.8
            ),
            emotional_tone=EmotionalTone.ENTHUSIASTIC,
            speech_patterns=[
                "      ...",
                "   ~       ?",
                "             "
            ],
            favorite_phrases=[
                "                ",
                "             "
            ],
            metaphor_themes=["  ", " ", "  "],
            expertise_areas=["  ", "  ", "   "],
            interests=["  ", "  ", "   "]
        )
        
        # 3.            (Caregiver Elysia)
        caregiver = Persona(
            name="Stella",
            archetype=PersonaArchetype.CAREGIVER,
            description="               .                  ",
            traits=PersonaTraits(
                openness=0.7,
                conscientiousness=0.8,
                extraversion=0.6,
                agreeableness=0.95,
                analytical_creative=0.6,
                logical_emotional=0.8,
                formal_casual=0.6
            ),
            emotional_tone=EmotionalTone.COMPASSIONATE,
            speech_patterns=[
                "             ",
                "          ",
                "       "
            ],
            favorite_phrases=[
                "         ",
                "             "
            ],
            metaphor_themes=[" ", "  ", " "],
            expertise_areas=["  ", "  ", "  "],
            interests=["  ", "  ", "  "]
        )
        
        # 4.          (Explorer Elysia)
        explorer = Persona(
            name="Nova",
            archetype=PersonaArchetype.EXPLORER,
            description="          .                       ",
            traits=PersonaTraits(
                openness=0.95,
                conscientiousness=0.5,
                extraversion=0.8,
                agreeableness=0.6,
                analytical_creative=0.7,
                practical_abstract=0.6
            ),
            emotional_tone=EmotionalTone.ENTHUSIASTIC,
            speech_patterns=[
                "     !",
                "      ?",
                "         "
            ],
            favorite_phrases=[
                "               ",
                "                 "
            ],
            metaphor_themes=["  ", " ", "  "],
            expertise_areas=["  ", "  ", "  "],
            interests=["  ", "  ", "  "]
        )
        
        # 5.          (Magician Elysia)
        magician = Persona(
            name="Arcana",
            archetype=PersonaArchetype.MAGICIAN,
            description="        .               ",
            traits=PersonaTraits(
                openness=0.9,
                conscientiousness=0.7,
                extraversion=0.5,
                agreeableness=0.6,
                analytical_creative=0.8,
                logical_emotional=0.5,
                practical_abstract=0.9,
                direct_metaphorical=0.9
            ),
            emotional_tone=EmotionalTone.MYSTERIOUS,
            speech_patterns=[
                "       ...",
                "         ",
                "               "
            ],
            favorite_phrases=[
                "               ",
                "             "
            ],
            metaphor_themes=[" ", "  ", "  "],
            expertise_areas=["   ", "  ", "   "],
            interests=["  ", "  ", "     "]
        )
        
        #          
        for persona in [sage, creator, caregiver, explorer, magician]:
            self.personas[persona.persona_id] = persona
    
    def add_persona(self, persona: Persona):
        """         """
        self.personas[persona.persona_id] = persona
        logger.info(f"  Added persona: {persona.name} ({persona.archetype.value})")
    
    def get_persona(self, persona_id: str) -> Optional[Persona]:
        """       """
        return self.personas.get(persona_id)
    
    def get_persona_by_name(self, name: str) -> Optional[Persona]:
        """            """
        for persona in self.personas.values():
            if persona.name.lower() == name.lower():
                return persona
        return None
    
    def list_personas(self) -> List[Dict[str, Any]]:
        """          """
        return [p.to_dict() for p in self.personas.values()]
    
    def find_personas_by_archetype(
        self, 
        archetype: PersonaArchetype
    ) -> List[Persona]:
        """            """
        return [
            p for p in self.personas.values() 
            if p.archetype == archetype
        ]
    
    def find_personas_by_expertise(self, expertise: str) -> List[Persona]:
        """              """
        return [
            p for p in self.personas.values()
            if expertise.lower() in [e.lower() for e in p.expertise_areas]
        ]


class PersonaManager:
    """
             (Persona Manager)
    
           ,   ,          .
    """
    
    def __init__(self):
        self.library = PersonaLibrary()
        self.current_persona: Optional[Persona] = None
        self.persona_stack: List[str] = []  #             
        self.blended_personas: List[Persona] = []  #             
        self.blend_weights: Dict[str, float] = {}  #       
        
        #             (Sophia -   )
        default_persona = self.library.find_personas_by_archetype(
            PersonaArchetype.SAGE
        )[0]
        self.switch_to(default_persona.persona_id)
        
        logger.info("  Persona Manager initialized")
    
    def switch_to(self, persona_id: str) -> bool:
        """       """
        persona = self.library.get_persona(persona_id)
        if not persona:
            logger.warning(f"   Persona {persona_id} not found")
            return False
        
        #           
        if self.current_persona:
            self.persona_stack.append(self.current_persona.persona_id)
        
        #   
        persona.activate()
        self.current_persona = persona
        
        #       
        self.blended_personas = [persona]
        self.blend_weights = {persona_id: 1.0}
        
        logger.info(f"  Switched to persona: {persona.name}")
        return True
    
    def switch_by_name(self, name: str) -> bool:
        """            """
        persona = self.library.get_persona_by_name(name)
        if persona:
            return self.switch_to(persona.persona_id)
        return False
    
    def blend_personas(
        self, 
        persona_ids: List[str], 
        weights: Optional[List[float]] = None
    ) -> bool:
        """
                  
        
        Args:
            persona_ids:          ID    
            weights:             (   1.0       )
        """
        #        
        personas = []
        for pid in persona_ids:
            persona = self.library.get_persona(pid)
            if persona:
                personas.append(persona)
            else:
                logger.warning(f"   Persona {pid} not found for blending")
                return False
        
        #       
        if weights is None:
            weights = [1.0 / len(personas)] * len(personas)
        elif len(weights) != len(personas):
            logger.error("   Weights count doesn't match personas count")
            return False
        elif abs(sum(weights) - 1.0) > 0.01:
            logger.error("   Weights must sum to 1.0")
            return False
        
        #      
        self.blended_personas = personas
        self.blend_weights = {
            pid: w for pid, w in zip(persona_ids, weights)
        }
        
        #                     
        main_idx = weights.index(max(weights))
        self.current_persona = personas[main_idx]
        
        logger.info(
            f"  Blended {len(personas)} personas: " + 
            ", ".join(f"{p.name} ({w:.2f})" for p, w in zip(personas, weights))
        )
        return True
    
    def suggest_persona_for_context(
        self, 
        context: str,
        keywords: Optional[List[str]] = None
    ) -> Optional[Persona]:
        """
                         
        
                      (         )
        """
        if keywords is None:
            keywords = context.lower().split()
        
        #             
        scores: Dict[str, float] = {}
        
        for persona in self.library.personas.values():
            score = 0.0
            
            #         
            for expertise in persona.expertise_areas:
                if any(kw in expertise.lower() for kw in keywords):
                    score += 2.0
            
            #       
            for interest in persona.interests:
                if any(kw in interest.lower() for kw in keywords):
                    score += 1.0
            
            #      
            for tag in persona.tags:
                if any(kw in tag.lower() for kw in keywords):
                    score += 1.0
            
            if score > 0:
                scores[persona.persona_id] = score
        
        if not scores:
            return None
        
        #                  
        best_persona_id = max(scores, key=scores.get)
        return self.library.get_persona(best_persona_id)
    
    def get_current_response_style(self) -> Dict[str, Any]:
        """                  """
        if not self.current_persona:
            return {}
        
        if len(self.blended_personas) == 1:
            #        
            return self.current_persona.generate_response_style()
        else:
            #         -      
            blended_style = {
                "tone": self.current_persona.emotional_tone.value,
                "personas": [p.name for p in self.blended_personas],
                "weights": list(self.blend_weights.values()),
                "primary": self.current_persona.name
            }
            return blended_style
    
    def get_status(self) -> Dict[str, Any]:
        """      """
        return {
            "current_persona": self.current_persona.to_dict() if self.current_persona else None,
            "is_blended": len(self.blended_personas) > 1,
            "blended_personas": [p.name for p in self.blended_personas],
            "blend_weights": self.blend_weights,
            "total_personas": len(self.library.personas),
            "persona_history": len(self.persona_stack)
        }


#      
def example_persona_usage():
    """              """
    manager = PersonaManager()
    
    print("\n             ")
    print("=" * 60)
    
    #        
    print(f"\n       : {manager.current_persona.name}")
    print(f"  : {manager.current_persona.archetype.value}")
    print(f"      : {manager.get_current_response_style()}")
    
    #        
    print("\n---              ---")
    manager.switch_by_name("Aurora")
    print(f"   : {manager.current_persona.name}")
    print(f"  : {manager.current_persona.description}")
    
    #        
    print("\n---         (   60% +       40%) ---")
    sage = manager.library.find_personas_by_archetype(PersonaArchetype.SAGE)[0]
    caregiver = manager.library.find_personas_by_archetype(PersonaArchetype.CAREGIVER)[0]
    manager.blend_personas(
        [sage.persona_id, caregiver.persona_id],
        [0.6, 0.4]
    )
    print(f"     : {manager.get_status()['blended_personas']}")
    
    #           
    print("\n---                 ---")
    context = "I want to create something new and innovative"
    suggested = manager.suggest_persona_for_context(context)
    if suggested:
        print(f"        : {suggested.name} ({suggested.archetype.value})")
        print(f"  : {suggested.description}")


if __name__ == "__main__":
    example_persona_usage()
