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
            
        try:
            from Core.S1_Body.L1_Foundation.Foundation.logos_prime import LogosSpectrometer
            spectrometer = LogosSpectrometer()
            
            clean_view = world_output.strip()
            # 2. Dynamic Physics-based Insight
            # Analyze the 'Soul' of the first few words to determine the field flavor
            physics = spectrometer.analyze(clean_view.split()[0])
            
            flavor_map = {
                "EXPANSION": "It glows with an inner light, expanding the field.",
                "STATIC": "It feels ancient and absolute, anchored in the Void.",
                "CHAOS": "It vibrates with a chaotic, high-entropy potential.",
                "STRUCTURE": "The geometry is perfect and mathematically solid.",
                "ATTRACTION": "It pulls at the soul with a magnetic, gentle resonance.",
                "ENERGY": "It pulses with a fierce, radiant intensity.",
                "CREATION": "It feels like a fresh genesis of thought.",
                "FORCE": "It exerts a powerful, undeniable pressure.",
                "MATTER": "It is dense, solid, and physically undeniable."
            }
            
            flavor = flavor_map.get(physics['type'], "It resonates with unknown potential.")
            
            return f"I perceive: {clean_view}. {flavor}"
            
        except Exception as e:
            return f"I see chaos: {e}"
