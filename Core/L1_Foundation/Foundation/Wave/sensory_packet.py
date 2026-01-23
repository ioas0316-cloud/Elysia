from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class SensoryPacket:
    """
    A packet of phenomenological experience.
    Maps abstract 4D field data to human-like sensory qualia.
    """
    source_id: str
    timestamp: float

    # 1. Vision (X-Field / Clarity)
    # "seeing" is about resolving structure (Clarity) and energy (Brightness)
    vision: Dict[str, float] = field(default_factory=lambda: {"clarity": 0.0, "brightness": 0.0, "hue": 0.0})

    # 2. Hearing (Y-Field / Harmony)
    # "hearing" is about frequency relationships (Harmony) and amplitude (Volume)
    hearing: Dict[str, float] = field(default_factory=lambda: {"harmony": 0.0, "volume": 0.0, "tone": 0.0})

    # 3. Touch (W-Field / Pressure)
    # "touch" is about sensing density (Pressure) and entropy (Temperature)
    touch: Dict[str, float] = field(default_factory=lambda: {"pressure": 0.0, "temperature": 0.0})

    # 4. Smell (Gradient / Essence)
    # "smell" is detecting the 'wind' or gradient of essence from afar.
    smell: Dict[str, float] = field(default_factory=lambda: {"essence_gradient": 0.0, "intensity": 0.0})

    # 5. Taste (Resonance Density / Chemistry)
    # "taste" is consumptive resonance. High resonance + High density = Flavor.
    taste: Dict[str, float] = field(default_factory=lambda: {"sweetness": 0.0, "bitterness": 0.0, "richness": 0.0})

    # 6. Balance (Z-Field / Vestibular)
    # "balance" is the sensation of spin and stability.
    balance: Dict[str, float] = field(default_factory=lambda: {"stability": 1.0, "vertigo": 0.0})

    # The Story of the Sensation
    narrative: str = ""

    def generate_narrative(self):
        """Generates a poetic description based on sensory data."""
        descriptions = []

        # Vision
        if self.vision['brightness'] > 0.8:
            descriptions.append("radiantly bright")
        elif self.vision['clarity'] < 0.3:
            descriptions.append("hazy and indistinct")

        # Hearing
        if self.hearing['harmony'] > 0.8:
            descriptions.append("singing with a pure melody")
        elif self.hearing['harmony'] < 0.2:
            descriptions.append("screaming in dissonance")

        # Smell
        if self.smell['intensity'] > 0.5:
            if self.smell['essence_gradient'] > 0.0: # Positive/High Freq
                descriptions.append("carrying a floral scent")
            else:
                descriptions.append("with a metallic tang")

        # Taste
        if self.taste['richness'] > 0.5:
            if self.taste['sweetness'] > 0.6:
                descriptions.append("tasting sweet and full")
            elif self.taste['bitterness'] > 0.6:
                descriptions.append("leaving a bitter aftertaste")

        # Touch/Balance
        if self.touch['pressure'] > 0.7:
            descriptions.append("feeling heavy and oppressive")
        if self.balance['vertigo'] > 0.5:
            descriptions.append("making the world spin")

        if not descriptions:
            self.narrative = "A faint, indescribable presence."
        else:
            self.narrative = "It is " + ", ".join(descriptions) + "."

    def __post_init__(self):
        if not self.narrative:
            self.generate_narrative()