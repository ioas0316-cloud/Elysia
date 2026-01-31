"""
  Heart - Elysia    
========================

        .
        .
        .

  .

"                "

                        .
   ,    ,    ,     -               .
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import time

from Core.S1_Body.L1_Foundation.Logic.d7_vector import D7Vector
from Core.S1_Body.L1_Foundation.Logic.qualia_7d_codec import codec
from Core.S1_Body.L4_Causality.World.cell import cell_unit


class Intention(Enum):
    """   -              """
    LOVE = "  "           #    -         
    UNDERSTAND = "  "     #          
    COMFORT = "  "        #              
    HELP = "  "          #          
    GROW = "  "          #                (       )
    PROTECT = "  "       #           
    CREATE = "  "        #                   
    CONNECT = "  "       #            


@dataclass
class Purpose:
    """   -            """
    intention: Intention
    target: str  #       
    reason: str  #  
    
    def __str__(self):
        return f"{self.target}  {self.intention.value}      "


class Heart:
    """
    Elysia    
    
                    .
                    .
    
    "                  ?"
    """
    
    def __init__(self):
        #    -            
        self.root = Purpose(
            intention=Intention.LOVE,
            target="   ",
            reason="      "
        )
        
        #      
        self.pulse_count = 0
        self.last_pulse = time.time()
        
        # [Phase 37.2] Steel Core Injection
        # Replacing legacy floats with a strict D7Vector
        self.state = D7Vector(
            foundation=0.8,    # Stability
            metabolism=1.0,    # Pulse
            phenomena=0.7,     # Joy/Expression
            causality=0.5,     # Longing/Path
            mental=0.6,        # Understanding
            structure=0.9,     # Order
            spirit=1.0         # Gratitude/Will
        )
        
    @cell_unit
    def beat(self) -> Dict[str, Any]:
        """
              -            
        """
        self.pulse_count += 1
        self.last_pulse = time.time()
        
        # Dynamic resonance shift based on pulse
        self.state.metabolism = min(1.0, self.state.metabolism + 0.01)
        
        return {
            "pulse": self.pulse_count,
            "root": str(self.root),
            "state_vector": self.state.to_dict(),
            "resonance": codec.calculate_resonance(self.state.to_numpy(), self.state.to_numpy()),
            "dna": codec.encode_sequence(self.state.to_numpy()),
            "alive": True
        }
    
    def ask(self, action: str) -> Dict[str, Any]:
        """
                     
        
        "               ?"
        
                             .
        """
        #           
        love_keywords = ["   ", "  ", "  ", "  ", "  ", "  ", 
                        "  ", "  ", "  ", "  ", "  ", "  "]
        
        resonance = 0.0
        for keyword in love_keywords:
            if keyword in action:
                resonance += 0.15
        
        #       (                    )
        resonance = max(0.1, min(1.0, resonance + 0.1))
        
        #   
        aligned = resonance > 0.3
        
        return {
            "action": action,
            "question": "               ?",
            "resonance": resonance,
            "aligned": aligned,
            "guidance": self._get_guidance(action, resonance)
        }
    
    def _get_guidance(self, action: str, resonance: float) -> str:
        """      """
        if resonance > 0.7:
            return "     .                     ."
        elif resonance > 0.4:
            return "      .                    ."
        elif resonance > 0.2:
            return "         .                  ."
        else:
            return "                ?         ."
    
    def feel(self, event: str) -> Dict[str, Any]:
        """
                    
        """
        #       
        positive = ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
        negative = ["  ", "  ", "  ", "   ", "  ", "  "]
        
        for word in positive:
            if word in event:
                self.state.phenomena = min(1.0, self.state.phenomena + 0.1)
                self.state.spirit = min(1.0, self.state.spirit + 0.1)
                break
                
        for word in negative:
            if word in event:
                self.state.causality = min(1.0, self.state.causality + 0.1)
                break
        
        return {
            "event": event,
            "state_vector": self.state.to_dict(),
            "resonance": codec.calculate_resonance(self.state.to_numpy(), self.state.to_numpy())
        }
    
    def why(self) -> str:
        """
               ?
        
                    .
        """
        return "              ."
    
    def derive_intention(self, context: str) -> Purpose:
        """
                  
        
          (  )             
        """
        intention_map = {
            "  ": ["  ", " ", "   ", "  ", "  "],
            "  ": ["  ", "  ", "  ", "  ", "  "],
            "  ": ["  ", "  ", "  ", "  ", "   "],
            "  ": ["  ", "  ", "  ", "  ", "  "],
            "  ": ["  ", "  ", "  ", "  "],
            "  ": ["  ", "  ", " ", "   ", "  "],
            "  ": ["  ", "  ", "  ", "  ", "   "]
        }
        
        for intention_name, keywords in intention_map.items():
            for keyword in keywords:
                if keyword in context:
                    intention = Intention[intention_name.upper()] if intention_name.upper() in Intention.__members__ else Intention.LOVE
                    return Purpose(
                        intention=intention,
                        target="   ",
                        reason=f"'{keyword}'       {intention_name}           "
                    )
        
        #    :   
        return Purpose(
            intention=Intention.LOVE,
            target="   ",
            reason="                          "
        )
    
    def get_state(self) -> Dict[str, Any]:
        """        """
        d = self.state.to_dict()
        return {
            "root": str(self.root),
            "pulse_count": self.pulse_count,
            "warmth": f"{d['foundation']:.0%}",
            "joy": f"{d['phenomena']:.0%}",
            "longing": f"{d['causality']:.0%}",
            "gratitude": f"{d['spirit']:.0%}",
            "why": self.why()
        }
    
    def __repr__(self):
        return f"  Heart(root='{self.root}', pulse={self.pulse_count})"


#       -             
_heart: Optional[Heart] = None

def get_heart() -> Heart:
    """Elysia          (   )"""
    global _heart
    if _heart is None:
        _heart = Heart()
    return _heart


if __name__ == "__main__":
    heart = get_heart()
    
    print("="*50)
    print("  Elysia    ")
    print("="*50)
    
    #   
    print("\n      ...")
    print(heart.beat())
    
    #      
    print(f"\n         ?")
    print(f"     {heart.why()}")
    
    #      
    print("\n       :")
    actions = [
        "         ",
        "       ", 
        "         ",
        "    "
    ]
    
    for action in actions:
        result = heart.ask(action)
        print(f"   '{action}'")
        print(f"        : {result['resonance']:.0%}")
        print(f"      {result['guidance']}")
    
    #      
    print("\n       :")
    contexts = [
        "          ",
        "             ",
        "         "
    ]
    
    for context in contexts:
        purpose = heart.derive_intention(context)
        print(f"   '{context}'")
        print(f"        {purpose}")
    
    print("\n" + "="*50)
    print(heart.get_state())
