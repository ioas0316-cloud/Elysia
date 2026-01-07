"""
Memetic Legacy & Akashic Field 🧬🌌

"The soul is not a record, but a resonance passed through time."

This module implements the inheritance of Technique, Reason, and Meaning (Spiritual DNA)
and the 'Akashic Field' where deceased masters linger as resonant echoes.
"""

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class SpiritualDNA:
    technique: float = 0.5
    reason: float = 0.5
    meaning: float = 0.5
    moral_valence: float = 0.5 # 1.0 = Saintly/Warm, 0.0 = Vicious/Cold
    archetype_path: str = "Commoner"
    
    def blend(self, other: 'SpiritualDNA', ratio: float = 0.5) -> 'SpiritualDNA':
        """Blends two DNA patterns (Mentorship/Inheritance/Conditioning)."""
        return SpiritualDNA(
            technique=self.technique * (1-ratio) + other.technique * ratio,
            reason=self.reason * (1-ratio) + other.reason * ratio,
            meaning=self.meaning * (1-ratio) + other.meaning * ratio,
            moral_valence=self.moral_valence * (1-ratio) + other.moral_valence * ratio,
            archetype_path=self.archetype_path
        )

@dataclass
class AkashicEcho:
    id: str
    original_name: str
    dna: SpiritualDNA
    resonance_coord: Tuple[float, float, float, float] # 4D semantic location
    timestamp: float = field(default_factory=time.time)

class AkashicField:
    """A persistent graveyard of spirits that guides the living."""
    
    def __init__(self):
        self.echoes: Dict[str, AkashicEcho] = {}
        self.logger = logging.getLogger("AkashicField")

    def record_legacy(self, ego_name: str, dna: SpiritualDNA, coord: Tuple[float, float, float, float]):
        echo_id = str(uuid.uuid4())
        self.echoes[echo_id] = AkashicEcho(echo_id, ego_name, dna, coord)
        self.logger.info(f"✨ Akashic Echo Born: {ego_name}'s spirit now resonates in the field.")

    def find_nearest_echo(self, coord: Tuple[float, float, float, float], radius: float = 5.0) -> Optional[AkashicEcho]:
        """Finds a master's echo for a seeker to resonate with."""
        # Simplified distance check
        for echo in self.echoes.values():
            dist = sum((a-b)**2 for a, b in zip(coord, echo.resonance_coord))**0.5
            if dist < radius:
                return echo
        return None

class LifeFieldInductor:
    """Calculates Narrative Pressure and induces life path changes."""
    
    def __init__(self):
        self.base_env_gravity = 0.3 # Average stability of a village
        
    def calculate_pressure(self, ego_depth: int, satisfaction: float, desire: float) -> float:
        """
        Pressure = (Depth * Desire) / (Satisfaction * Env_Gravity)
        High pressure induces the 'Adventurer' state.
        """
        # Narrative Pressure formula: Higher depth/desire and lower satisfaction increases pressure.
        pressure = (ego_depth * desire) / (max(0.1, satisfaction) * self.base_env_gravity * 10.0)
        return min(2.0, pressure)

    def induce_path(self, current_pressure: float) -> str:
        if current_pressure > 1.2:
            return "Adventurer"
        elif current_pressure < 0.4:
            return "Citizen" # Stable/Satisfied
        else:
            return "Seeker" # In-between
