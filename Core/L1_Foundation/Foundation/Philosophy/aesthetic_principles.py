"""
Aesthetic Principles (     )
================================

"       ?" - Why is it beautiful?

                         .
           ,   ,                     .

8 Universal Principles:
1. Harmony (  ) - Elements working together
2. Contrast (  ) - Differences create emphasis
3. Balance (  ) - Visual/emotional stability
4. Rhythm (  ) - Repetition and variation
5. Tension-Release (  -  ) - Emotional waves
6. Proportion (  ) - Golden ratio, rule of thirds
7. Unity (   ) - All elements serve one theme
8. Flow (  ) - Natural movement of eye/emotion
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class Medium(Enum):
    """     """
    VISUAL = "visual"       #       (  ,   )
    LITERARY = "literary"   #    ( ,   )
    TEMPORAL = "temporal"   #       (  ,   )
    UNIVERSAL = "universal" #      


@dataclass
class AestheticVector:
    """
           - 4              
    
    w: intensity (  ) -                    
    x: visual (   ) -            
    y: literary (   ) -         
    z: temporal (   ) -   /        
    """
    w: float = 0.0  # intensity
    x: float = 0.0  # visual
    y: float = 0.0  # literary
    z: float = 0.0  # temporal
    
    def magnitude(self) -> float:
        """       (       )"""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'AestheticVector':
        """   """
        mag = self.magnitude()
        if mag == 0:
            return AestheticVector(0, 0, 0, 0)
        return AestheticVector(self.w/mag, self.x/mag, self.y/mag, self.z/mag)
    
    def __add__(self, other: 'AestheticVector') -> 'AestheticVector':
        return AestheticVector(
            self.w + other.w, self.x + other.x,
            self.y + other.y, self.z + other.z
        )
    
    def __mul__(self, scalar: float) -> 'AestheticVector':
        return AestheticVector(
            self.w * scalar, self.x * scalar,
            self.y * scalar, self.z * scalar
        )
    
    def dot(self, other: 'AestheticVector') -> float:
        """             """
        return self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z


@dataclass
class AestheticPrinciple:
    """
             
    
                           ,
             "        "   .
    """
    name: str
    korean_name: str
    description: str
    vector: AestheticVector
    
    #               
    visual_expression: str = ""      #           ?
    literary_expression: str = ""    #         ?
    temporal_expression: str = ""    #   /        ?
    
    #          (          )
    opposite: Optional[str] = None
    
    def apply_to_medium(self, medium: Medium) -> float:
        """                   """
        if medium == Medium.VISUAL:
            return self.vector.x * self.vector.w
        elif medium == Medium.LITERARY:
            return self.vector.y * self.vector.w
        elif medium == Medium.TEMPORAL:
            return self.vector.z * self.vector.w
        else:
            return self.vector.magnitude()


@dataclass
class AestheticField:
    """
              
    
                       "       ( )"
       ResonanceField               ,
    AestheticField                 .
    """
    principles: Dict[str, float] = field(default_factory=dict)  #     ->   
    dominant_principle: Optional[str] = None
    medium: Medium = Medium.UNIVERSAL
    
    def add_principle(self, name: str, intensity: float):
        """           """
        self.principles[name] = self.principles.get(name, 0) + intensity
        
        #            
        if self.principles:
            self.dominant_principle = max(self.principles, key=self.principles.get)
    
    def calculate_beauty_score(self) -> float:
        """
                  
        
                 ,                 .
                             ,
                                  .
        """
        if not self.principles:
            return 0.0
        
        values = list(self.principles.values())
        total = sum(values)
        
        if total == 0:
            return 0.0
        
        #             (   )
        entropy = 0.0
        for v in values:
            if v > 0:
                p = v / total
                entropy -= p * math.log(p + 1e-10)
        
        #         (         )
        max_entropy = math.log(len(values))
        
        #     (0~1)
        harmony = entropy / (max_entropy + 1e-10) if max_entropy > 0 else 0
        
        #    (     )
        intensity = min(total / 10.0, 1.0)  #    
        
        #      :           
        return (intensity * 0.6 + harmony * 0.4) * 100
    
    def analyze_why_beautiful(self) -> str:
        """
        "       ?"      
        
                  -                      ,
                             .
        """
        if not self.principles:
            return "               ."
        
        #    3       
        sorted_principles = sorted(
            self.principles.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        explanations = []
        for name, intensity in sorted_principles:
            if intensity > 0.5:
                explanations.append(f"'{name}'             (  : {intensity:.1f})")
        
        score = self.calculate_beauty_score()
        harmony_level = "  " if score > 70 else "  " if score > 40 else "  "
        
        result = f"[     ]\n"
        result += f"       : {score:.1f}/100\n"
        result += f"     : {harmony_level}\n"
        result += f"      : {self.dominant_principle or '  '}\n\n"
        result += "     :\n"
        for exp in explanations:
            result += f"    {exp}\n"
        
        return result


class AestheticWisdom:
    """
           (Aesthetic Wisdom)
    
          "             "     ,
                                      .
    """
    
    def __init__(self):
        print("  AestheticWisdom    :                ...")
        self.principles = self._init_universal_principles()
        self.learned_patterns: List[AestheticField] = []
        
    def _init_universal_principles(self) -> Dict[str, AestheticPrinciple]:
        """8               """
        
        return {
            "harmony": AestheticPrinciple(
                name="Harmony",
                korean_name="  ",
                description="                      ",
                vector=AestheticVector(1.0, 0.9, 0.8, 0.85),
                visual_expression="     ,      ",
                literary_expression="         ,       ",
                temporal_expression="           ",
                opposite="discord"
            ),
            "contrast": AestheticPrinciple(
                name="Contrast",
                korean_name="  ",
                description="           ,         ",
                vector=AestheticVector(1.0, 0.95, 0.7, 0.9),
                visual_expression="     ,      ,      ",
                literary_expression="           ,       ",
                temporal_expression="          ,       ",
                opposite="monotony"
            ),
            "balance": AestheticPrinciple(
                name="Balance",
                korean_name="  ",
                description="   /          ",
                vector=AestheticVector(0.9, 1.0, 0.75, 0.8),
                visual_expression="  /      ,      ",
                literary_expression="       ,      ",
                temporal_expression="         ,       ",
                opposite="imbalance"
            ),
            "rhythm": AestheticPrinciple(
                name="Rhythm",
                korean_name="  ",
                description="              ",
                vector=AestheticVector(0.95, 0.7, 0.9, 1.0),
                visual_expression="     ,       ",
                literary_expression="        ,   ",
                temporal_expression="     ,   ",
                opposite="chaos"
            ),
            "tension_release": AestheticPrinciple(
                name="Tension-Release",
                korean_name="  -  ",
                description="      ,       ",
                vector=AestheticVector(1.0, 0.75, 0.95, 0.9),
                visual_expression="       ,      ",
                literary_expression="      ,     ",
                temporal_expression="         ,   ",
                opposite="flatness"
            ),
            "proportion": AestheticPrinciple(
                name="Proportion",
                korean_name="  ",
                description="   ,               ",
                vector=AestheticVector(0.85, 1.0, 0.6, 0.7),
                visual_expression="      ,       ",
                literary_expression="3    ,      ",
                temporal_expression="         ,      ",
                opposite="disproportion"
            ),
            "unity": AestheticPrinciple(
                name="Unity",
                korean_name="   ",
                description="                 ",
                vector=AestheticVector(0.9, 0.85, 0.9, 0.85),
                visual_expression="     ,        ",
                literary_expression="     ,       ",
                temporal_expression="      ,       ",
                opposite="fragmentation"
            ),
            "flow": AestheticPrinciple(
                name="Flow",
                korean_name="  ",
                description="        /      ",
                vector=AestheticVector(0.9, 0.8, 0.85, 0.95),
                visual_expression="      ,      ",
                literary_expression="     ,      ",
                temporal_expression="    ,     ",
                opposite="stagnation"
            )
        }
    
    def analyze(self, content_description: str, medium: Medium = Medium.UNIVERSAL) -> AestheticField:
        """
                   
        
        Args:
            content_description:        (         )
            medium:      
            
        Returns:
                AestheticField
        """
        field = AestheticField(medium=medium)
        desc_lower = content_description.lower()
        
        #              (   ML          )
        keyword_map = {
            "harmony": ["  ", "  ", "   ", "harmony", "balanced colors", "complementary"],
            "contrast": ["  ", "  ", "contrast", "bold", "striking", "difference"],
            "balance": ["  ", "  ", "symmetry", "balanced", "centered"],
            "rhythm": ["  ", "  ", "  ", "rhythm", "pattern", "repetition"],
            "tension_release": ["  ", "  ", "    ", "climax", "tension", "release"],
            "proportion": ["  ", "   ", "   ", "golden ratio", "rule of thirds"],
            "unity": ["  ", "  ", "  ", "unified", "cohesive", "consistent"],
            "flow": ["  ", "  ", "     ", "flow", "leading", "movement"]
        }
        
        for principle_name, keywords in keyword_map.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    #         
                    intensity = 1.0 + desc_lower.count(keyword) * 0.5
                    field.add_principle(principle_name, intensity)
        
        return field
    
    def learn_from_example(self, field: AestheticField):
        """
               
        
                 AestheticField      
                 .
        """
        self.learned_patterns.append(field)
        print(f"          : {field.dominant_principle} (  {len(self.learned_patterns)} )")
    
    def get_principle(self, name: str) -> Optional[AestheticPrinciple]:
        """          """
        return self.principles.get(name)
    
    def explain_principle(self, name: str, medium: Medium = Medium.UNIVERSAL) -> str:
        """      (          )"""
        principle = self.get_principle(name)
        if not principle:
            return f"   '{name}'           ."
        
        explanation = f"[{principle.korean_name} ({principle.name})]\n"
        explanation += f"{principle.description}\n\n"
        
        if medium in [Medium.VISUAL, Medium.UNIVERSAL]:
            explanation += f"     : {principle.visual_expression}\n"
        if medium in [Medium.LITERARY, Medium.UNIVERSAL]:
            explanation += f"  : {principle.literary_expression}\n"
        if medium in [Medium.TEMPORAL, Medium.UNIVERSAL]:
            explanation += f"  /  : {principle.temporal_expression}\n"
        
        return explanation
    
    def suggest_for_creation(self, concept: str, medium: Medium) -> Dict[str, float]:
        """
                    
        
                              
                    .
        """
        suggestions = {}
        
        for name, principle in self.principles.items():
            #               
            strength = principle.apply_to_medium(medium)
            
            #          (        )
            if concept:
                concept_lower = concept.lower()
                #         tension_release   
                if any(w in concept_lower for w in ["  ", "emotion", "drama", "  ", "  "]):
                    if name == "tension_release":
                        strength *= 1.5
                #          harmony   
                if any(w in concept_lower for w in ["  ", "  ", "peace", "calm"]):
                    if name == "harmony":
                        strength *= 1.5
            
            suggestions[name] = strength
        
        return suggestions


#         
_wisdom_instance: Optional[AestheticWisdom] = None

def get_aesthetic_principles() -> AestheticWisdom:
    """AestheticWisdom         """
    global _wisdom_instance
    if _wisdom_instance is None:
        _wisdom_instance = AestheticWisdom()
    return _wisdom_instance


#       
if __name__ == "__main__":
    wisdom = get_aesthetic_principles()
    
    #          
    print(wisdom.explain_principle("harmony"))
    print("\n" + "="*50 + "\n")
    
    #       
    analysis = wisdom.analyze(
        "                                      . "
        "                                 .",
        Medium.VISUAL
    )
    print(analysis.analyze_why_beautiful())
    
    #          
    print("\n[     : '     '   ]")
    suggestions = wisdom.suggest_for_creation("     ", Medium.LITERARY)
    for name, strength in sorted(suggestions.items(), key=lambda x: -x[1]):
        print(f"  {name}: {strength:.2f}")