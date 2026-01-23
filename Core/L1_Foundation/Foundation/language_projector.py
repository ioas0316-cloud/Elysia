# -*- coding: utf-8 -*-
"""
Language Projector
==================

Projects abstract concepts and energy flows into natural language (Korean/English).
Adapted from Legacy/Language/emergent_language.py.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SymbolType(Enum):
    ENTITY = "entity"      #    ( ,  ,   )
    ACTION = "action"      #    (  ,   ,   )
    STATE = "state"        #    (  ,    ,   )
    RELATION = "relation"  #    ( ,   ,   )
    QUANTITY = "quantity"  #   (  ,   ,   )
    TIME = "time"          #    (  ,   ,   )
    SPACE = "space"        #    (  ,   ,  )
    EMOTION = "emotion"    #    (  ,   ,   )
    UNKNOWN = "unknown"

class LanguageProjector:
    """
    Projects abstract concepts into natural language.
    Handles Korean particles (Josa) and English word order based on energy flow.
    """
    
    def __init__(self):
        #            (     )
        # This will be supplemented by the dynamic Concept definitions
        self.korean_lexicon = {
            #   
            "SELF": " ", "OTHER": " ", "IT": "  ", "WE": "  ",
            "PARENT": "  ", "CHILD": "  ", "FRIEND": "  ",
            
            #   
            "EXIST": "  ", "MOVE": "  ", "EAT": "  ", "SPEAK": "   ",
            "SEE": "  ", "HEAR": "  ", "FEEL": "   ", "THINK": "    ",
            "LOVE": "    ", "HATE": "    ", "WANT": "   ",
            "GIVE": "  ", "TAKE": "  ", "MAKE": "   ",
            "CREATES": "   ", "CAUSES": "    ", "ENABLES": "       ",
            
            #   
            "GOOD": "  ", "BAD": "   ", "BIG": "  ", "SMALL": "  ",
            "HAPPY": "   ", "SAD": "   ", "ANGRY": "   ",
            "WARM": "    ", "COLD": "   ", "BRIGHT": "  ", "DARK": "   ",
            
            #   
            "WITH": " ", "TO": "  ", "FROM": "  ", "IN": "  ",
            "AND": "   ", "BUT": "   ", "BECAUSE": "    ",
            
            #   
            "NOW": "  ", "BEFORE": "  ", "AFTER": "  ", "ALWAYS": "  ",
            
            #   
            "HERE": "  ", "THERE": "  ", "UP": " ", "DOWN": "  ",
            
            #   
            "JOY": "  ", "SORROW": "  ", "FEAR": "   ", "LOVE_N": "  ",
        }
        
        #      
        self.english_lexicon = {
            "SELF": "I", "OTHER": "you", "IT": "it", "WE": "we",
            "EXIST": "exist", "MOVE": "go", "EAT": "eat", "SPEAK": "speak",
            "GOOD": "good", "BAD": "bad", "HAPPY": "happy", "SAD": "sad",
            "NOW": "now", "HERE": "here", "WITH": "with", "TO": "to",
            "CREATES": "creates", "CAUSES": "causes", "ENABLES": "enables",
        }

    def get_korean_name(self, concept_name: str) -> str:
        """Get Korean name for a concept, defaulting to lowercased name"""
        return self.korean_lexicon.get(concept_name.upper(), concept_name)

    def get_english_name(self, concept_name: str) -> str:
        """Get English name for a concept"""
        return self.english_lexicon.get(concept_name.upper(), concept_name)

    def project_to_korean(self, source: str, action: str, target: str, passive: bool = False) -> str:
        """
        Project an energy flow (Source -> Action -> Target) to Korean (SOV).
        Active: Source( / ) Target( / ) Action(  ).
        Passive: Source( / ) Target(    ) Action(  /  ).
        """
        s_name = self.get_korean_name(source)
        t_name = self.get_korean_name(target)
        a_name = self.get_korean_name(action)
        
        if passive:
            # Passive: Source is the Subject (originally Target), Target is the Agent (originally Source)
            # But in our StarSystem, 'star' is passed as 'source' here because it's the subject.
            # So: Star(Subject) ... Planet(Agent) ...
            # Let's clarify: The caller (StarSystem) aligns Star as Subject.
            # If Passive, Star was the Object of the action.
            # "Bonds(Star) are created by Love(Planet)"
            # Korean: "Bonds( / ) Love(    )      "
            
            # Map action to passive form if possible
            passive_map = {
                "   ": "     ",
                "    ": "    ", # or     
                "       ": "     ",
                "    ": "    ",
                "  ": "   ",
                "  ": "   ",
            }
            a_name = passive_map.get(a_name, a_name + "    /  ") # Fallback
            
            sentence = f"{s_name} /  {t_name}     {a_name}"
        else:
            # Active
            sentence = f"{s_name} /  {t_name} /  {a_name}"
            if not a_name.endswith(" "):
                sentence += "  "
            
        return sentence

    def project_to_english(self, source: str, action: str, target: str, passive: bool = False) -> str:
        """
        Project an energy flow to English (SVO).
        Active: Source Action Target.
        Passive: Source is Actioned by Target.
        """
        s_name = self.get_english_name(source)
        t_name = self.get_english_name(target)
        a_name = self.get_english_name(action)
        
        if passive:
            # Passive: Source is Actioned by Target
            # "Bonds are created by Love"
            
            # Simple conjugation
            if a_name.endswith("e"):
                past_participle = a_name + "d"
            else:
                past_participle = a_name + "ed"
                
            # Handle irregulars if needed (creates -> created is fine)
            
            return f"{s_name} is {past_participle} by {t_name}"
        else:
            return f"{s_name} {a_name} {t_name}"