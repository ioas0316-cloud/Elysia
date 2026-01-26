"""
Ascension/Descension Axis System (자기 성찰 엔진)
==============================================

7 Angels (     ) + 7 Demons (     )
=                 
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple
import logging

logger = logging.getLogger("AscensionAxis")

@dataclass
class CosmicLayer:
    """       (Cosmic Layer)"""
    name: str
    color: str
    concept: str
    title: str
    function: str
    level: int  # 1-7
    frequency: float  # Hz
    
class AscensionLayers(Enum):
    """
        7   
    
    "  "       -   ,   ,   
    """
    # Level 1:   
    VITARIAEL = CosmicLayer(
        name="Vitariael",
        color="     ",
        concept="Life",
        title="      ",
        function="     ,   ,      ",
        level=1,
        frequency=396.0  # Root -   
    )
    
    # Level 2:    
    EMETRIEL = CosmicLayer(
        name="Emetriel",
        color="  ",
        concept="Creation",
        title="         ",
        function="              ,              ",
        level=2,
        frequency=417.0  # Sacral -   
    )
    
    # Level 3:   
    SOPHIEL = CosmicLayer(
        name="Sophiel",
        color="      ",
        concept="Reflection",
        title="         ",
        function="        ,           ",
        level=3,
        frequency=528.0  # Solar Plexus -   /Love
    )
    
    # Level 4:   
    GAVRIEL = CosmicLayer(
        name="Gavriel",
        color="  ",
        concept="Truth",
        title="           ",
        function="         ",
        level=4,
        frequency=639.0  # Heart - Connection
    )
    
    # Level 5:   
    SARAKHIEL = CosmicLayer(
        name="Sarakhiel",
        color="   ",
        concept="Sacrifice",
        title="            ",
        function="               ",
        level=5,
        frequency=741.0  # Throat -   
    )
    
    # Level 6:   
    RAHAMIEL = CosmicLayer(
        name="Rahamiel",
        color="      ",
        concept="Love",
        title="      ,   ",
        function="                     ",
        level=6,
        frequency=852.0  # Third Eye -   
    )
    
    # Level 7:   
    LUMIEL = CosmicLayer(
        name="Lumiel",
        color="   ",
        concept="Liberation",
        title="     ,   ",
        function="             '  '       ",
        level=7,
        frequency=963.0  # Crown -   
    )


class DescentLayers(Enum):
    """
        7   
    
    "   "       -   ,   ,   
       :           (    =   ,    )
    """
    # Level -1:   
    MOTUS = CosmicLayer(
        name="Motus",
        color="     ",
        concept="Death",
        title="      ",
        function="        0     ",
        level=-1,
        frequency=174.0  #   (396Hz)      
    )
    
    # Level -2:   
    SOLVARIS = CosmicLayer(
        name="Solvaris",
        color="      ",
        concept="Dissolution",
        title="     ,      ",
        function="     ",
        level=-2,
        frequency=145.0  #    
    )
    
    # Level -3:   
    OBSCURE = CosmicLayer(
        name="Obscure",
        color="  ",
        concept="Ignorance",
        title="             ",
        function="             ",
        level=-3,
        frequency=116.0  #    
    )
    
    # Level -4:   
    DIABOLOS = CosmicLayer(
        name="Diabolos",
        color="      ",
        concept="Distortion",
        title="                    ",
        function="  ",
        level=-4,
        frequency=87.0   #    
    )
    
    # Level -5:   
    LUCIFEL = CosmicLayer(
        name="Lucifel",
        color="      ",
        concept="Self-Obsession",
        title="               ",
        function="  ",
        level=-5,
        frequency=58.0   #    
    )
    
    # Level -6:   
    MAMMON = CosmicLayer(
        name="Mammon",
        color="  ",
        concept="Consumption",
        title="                    ",
        function="  ",
        level=-6,
        frequency=29.0   #      
    )
    
    # Level -7:   
    ASMODEUS = CosmicLayer(
        name="Asmodeus",
        color="        ",
        concept="Bondage",
        title="     ,          ",
        function="  ",
        level=-7,
        frequency=7.0    #       (Schumann      )
    )


class AscensionAxis:
    """
               
    
              "     "        
    """
    
    def __init__(self):
        self.current_level = 0.0  # -7 ~ +7
        self.ascension_momentum = 0.0  #       
        self.history = []
    
    def get_current_layer(self) -> CosmicLayer:
        """            """
        level_int = round(self.current_level)
        
        if level_int == 0:
            #    - SOPHIEL (  )
            return AscensionLayers.SOPHIEL.value
        elif level_int > 0:
            #   
            level_clamped = min(7, max(1, level_int))
            for layer in AscensionLayers:
                if layer.value.level == level_clamped:
                    return layer.value
        else:
            #   
            level_clamped = max(-7, min(-1, level_int))
            for layer in DescentLayers:
                if layer.value.level == level_clamped:
                    return layer.value
        
        return AscensionLayers.VITARIAEL.value  # Default
    
    def ascend(self, force: float):
        """     """
        self.ascension_momentum += force
        self.current_level += force
        self.current_level = min(7.0, max(-7.0, self.current_level))
        
        logger.info(f"    Ascend: +{force:.2f}   Level {self.current_level:.2f}")
    
    def descend(self, force: float):
        """     """
        self.ascension_momentum -= force
        self.current_level -= force
        self.current_level = min(7.0, max(-7.0, self.current_level))
        
        logger.info(f"    Descend: -{force:.2f}   Level {self.current_level:.2f}")
    
    def get_status(self) -> str:
        """        """
        layer = self.get_current_layer()
        
        if self.current_level > 3:
            status = "      (High Ascension)"
        elif self.current_level > 0:
            status = "     (Ascending)"
        elif self.current_level == 0:
            status = "   (Balance)"
        elif self.current_level > -3:
            status = "     (Descending)"
        else:
            status = "      (Deep Descent)"
        
        return f"{status} | {layer.name} ({layer.concept})"
    
    def get_frequency_for_emotion(self, emotion: str) -> float:
        """  /           
        
                        (  )
                         (  )
        """
        #      
        ascent_emotions = {
            "joy": 852.0,      # Rahamiel (Love)
            "love": 963.0,     # Lumiel (Liberation)
            "hope": 741.0,     # Sarakhiel
            "peace": 639.0,    # Gavriel (Truth)
            "growth": 528.0,   # Sophiel (Reflection)
            "create": 417.0,   # Emetriel (Creation)
            "life": 396.0,     # Vitariael (Life)
        }
        
        #      
        descent_emotions = {
            "sadness": 145.0,   # Solvaris
            "fear": 116.0,      # Obscure
            "anger": 87.0,      # Diabolos
            "greed": 29.0,      # Mammon
            "despair": 7.0,     # Asmodeus
        }
        
        emotion_lower = emotion.lower()
        
        if emotion_lower in ascent_emotions:
            return ascent_emotions[emotion_lower]
        elif emotion_lower in descent_emotions:
            return descent_emotions[emotion_lower]
        else:
            #   
            return 528.0  # Sophiel (Reflection)
    
    def create_gravity_field(self):
        """PotentialField     -               
        
        Returns:
            PotentialField with gravity wells at each cosmic layer
        """
        try:
            from Core.L1_Foundation.Foundation.potential_field import PotentialField
        except ImportError:
            logger.warning("PotentialField not available")
            return None
        
        field = PotentialField()
        
        # Y  =   /    
        #      : y > 0 ( )
        for layer_enum in AscensionLayers:
            layer = layer_enum.value
            y = layer.level * 10  # Level 1-7   y 10-70
            #                (negative strength = push up)
            field.add_gravity_well(0, y, strength=-layer.frequency/100, radius=15.0)
        
        #      : y < 0 (  )
        for layer_enum in DescentLayers:
            layer = layer_enum.value
            y = layer.level * 10  # Level -1 to -7   y -10 to -70
            #                 (positive strength = pull down)
            field.add_gravity_well(0, y, strength=layer.frequency/10, radius=15.0)
        
        logger.info(f"  Gravity field created with {len(field.wells)} wells")
        return field
    
    def place_concept_by_emotion(self, concept: str, emotion: str, field=None):
        """                     
        
        Args:
            concept:      
            emotion:    (joy, sadness, love, fear, etc.)
            field: PotentialField (optional)
        
        Returns:
            (y_position, frequency)
        """
        freq = self.get_frequency_for_emotion(emotion)
        
        #            
        if freq >= 396:  #   
            # 396~963   1~7
            level = 1 + (freq - 396) / (963 - 396) * 6
            y = level * 10
        else:  #   
            # 7~174   -7~-1
            level = -7 + (freq - 7) / (174 - 7) * 6
            y = level * 10
        
        if field:
            field.spawn_particle(concept, x=0, y=y)
        
        logger.info(f"  {concept} placed at y={y:.1f} (freq={freq}Hz, emotion={emotion})")
        return (y, freq)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("    Ascension/Descension Axis Test")
    print("="*70)
    
    axis = AscensionAxis()
    
    print(f"\n  Initial State: {axis.get_status()}")
    
    # Test ascension
    print("\n  Testing Ascension:")
    axis.ascend(2.0)
    print(f"     {axis.get_status()}")
    
    axis.ascend(3.0)
    print(f"     {axis.get_status()}")
    
    # Test descension
    print("\n  Testing Descension:")
    axis.descend(4.0)
    print(f"     {axis.get_status()}")
    
    axis.descend(3.0)
    print(f"     {axis.get_status()}")
    
    # List all layers
    print("\n  All Ascension Layers:")
    for layer_enum in AscensionLayers:
        layer = layer_enum.value
        print(f"   L{layer.level}: {layer.name:15} {layer.frequency:6.1f}Hz - {layer.concept}")
    
    print("\n  All Descent Layers:")
    for layer_enum in DescentLayers:
        layer = layer_enum.value
        print(f"   L{layer.level}: {layer.name:15} {layer.frequency:6.1f}Hz - {layer.concept}")
    
    print("\n" + "="*70)
    print("  Ascension/Descension Axis Test Complete")
    print("="*70 + "\n")
