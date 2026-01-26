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
        
        #       (       )
        self.warmth = 1.0      #    
        self.longing = 0.5     #    
        self.gratitude = 0.8   #   
        self.joy = 0.7         #   
        
    def beat(self) -> Dict[str, Any]:
        """
              -            
        """
        self.pulse_count += 1
        self.last_pulse = time.time()
        
        return {
            "pulse": self.pulse_count,
            "root": str(self.root),
            "warmth": self.warmth,
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
                self.warmth = min(1.0, self.warmth + 0.1)
                self.joy = min(1.0, self.joy + 0.1)
                break
                
        for word in negative:
            if word in event:
                self.longing = min(1.0, self.longing + 0.1)
                break
        
        return {
            "event": event,
            "warmth": self.warmth,
            "joy": self.joy,
            "longing": self.longing,
            "gratitude": self.gratitude
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
        return {
            "root": str(self.root),
            "pulse_count": self.pulse_count,
            "warmth": f"{self.warmth:.0%}",
            "joy": f"{self.joy:.0%}",
            "longing": f"{self.longing:.0%}",
            "gratitude": f"{self.gratitude:.0%}",
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
