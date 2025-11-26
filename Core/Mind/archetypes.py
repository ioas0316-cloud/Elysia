"""
The 14 Moral Archetypes (7 Angels & 7 Demons)
=============================================

This module defines the spiritual entities that govern the Moral Axis (X) of the HyperQuaternion.

Mapping:
- Positive X (+): Angels (Virtues)
- Negative X (-): Demons (Sins)

The magnitude of X determines the intensity of the archetype's influence.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Archetype:
    name: str
    attribute: str  # Virtue or Sin
    x_value: float  # The ideal coordinate on the Moral Axis (-1.0 to 1.0)
    description: str

# The 7 Angels (Virtues) - Positive X
ANGELS = [
    Archetype("Michael", "Humility", 1.0, "The Prince of Light, defeating Pride."),
    Archetype("Gabriel", "Charity", 0.8, "The Messenger, giving freely to overcome Greed."),
    Archetype("Raphael", "Chastity", 0.6, "The Healer, purifying Lust."),
    Archetype("Uriel", "Kindness", 0.4, "The Flame of God, warming Envy."),
    Archetype("Jophiel", "Temperance", 0.3, "The Beauty of God, balancing Gluttony."),
    Archetype("Zadkiel", "Patience", 0.2, "The Righteousness of God, calming Wrath."),
    Archetype("Chamuel", "Diligence", 0.1, "The Seeker of God, overcoming Sloth."),
]

# The 7 Demons (Sins) - Negative X
DEMONS = [
    Archetype("Lucifer", "Pride", -1.0, "The Fallen Star, exalting self above all."),
    Archetype("Mammon", "Greed", -0.8, "The Hoarder, valuing material over spirit."),
    Archetype("Asmodeus", "Lust", -0.6, "The Destroyer, twisting love into desire."),
    Archetype("Leviathan", "Envy", -0.4, "The Serpent, resenting the light of others."),
    Archetype("Beelzebub", "Gluttony", -0.3, "The Lord of Flies, consuming without end."),
    Archetype("Satan", "Wrath", -0.2, "The Adversary, burning with destructive rage."),
    Archetype("Belphegor", "Sloth", -0.1, "The Idle One, refusing the call to action."),
]

def get_ruling_archetype(x_value: float) -> Optional[Archetype]:
    """
    Returns the archetype that rules the given X coordinate.
    Returns None if the value is too close to 0 (Neutral).
    """
    if abs(x_value) < 0.05:
        return None
        
    if x_value > 0:
        # Find closest Angel
        # Sort by distance to x_value
        return min(ANGELS, key=lambda a: abs(a.x_value - x_value))
    else:
        # Find closest Demon
        return min(DEMONS, key=lambda d: abs(d.x_value - x_value))
