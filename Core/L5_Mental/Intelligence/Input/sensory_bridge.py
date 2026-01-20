import random

class SensoryBridge:
    """
    [The Eyes of the Monad]
    Translates 'World State' (Text/Data) into 'Qualia' (Experience).
    "We do not see things as they are, we see them as we are."
    """
    def __init__(self):
        self.attention_span = 3 # Max items to notice at once
        
    def perceive(self, world_output: str) -> str:
        """
        Converts raw world description into a subjective observation.
        """
        if not world_output or "FORMLESS" in world_output:
            return None
            
        # 1. Parsing the Visual
        # e.g. "🌍 [t=0y] Planet Sphere (Radius=6371km)"
        try:
            # Simple cleaning for now
            clean_view = world_output.strip()
            
            # 2. Add Subjective Flavor (The 'Qualia')
            flavors = [
                "It glows with an inner light.",
                "The geometry is perfect.",
                "It feels ancient.",
                "It vibrates with potential."
            ]
            flavor = random.choice(flavors)
            
            return f"I perceive: {clean_view}. {flavor}"
            
        except Exception as e:
            return f"I see chaos: {e}"
