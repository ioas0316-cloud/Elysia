"""
Global Attractor Dynamics Architecture
=====================================
This module defines macro-scale world forces (Demon Lord Awakening, Celestial War, Cataclysm, Mana Density Shifts)
that exert global attractor pressure across all entities, factions, and regions.

Core Philosophy:
1. Macro-scale events are Global Attractors (Attractor Fields).
2. When pressure breaches threshold, entities' lowest-level survival / emergency protocol is elevated to top priority.
3. Forces state transitions: enemies are forced into survival alliances, or Opportunistic Betrayal triggers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
from core.causal_world.protocol import Entity
from core.causal_world.factions import Faction, FactionType


class AttractorType(Enum):
    DEMON_LORD_AWAKENING = "Demon Lord Awakening"  # Catastrophic Dark Pressure
    CELESTIAL_WAR = "Celestial War"                # Absolute Order vs Chaos Grid
    NATURAL_CATACLYSM = "Natural Cataclysm"        # Total Infrastructure Destruction
    MANA_DENSITY_SHIFT = "Mana Density Shift"      # Magic Volatility Surge


@dataclass
class GlobalAttractorState:
    attractor_type: AttractorType
    title: str
    description: str
    current_pressure: float = 0.0          # 0.0 to 100.0
    threshold_pressure: float = 80.0       # Trigger threshold for macro overrides
    is_active: bool = False
    affected_regions: List[str] = field(default_factory=list)


class GlobalAttractorEngine:
    """
    Manages global attractor forces and applies emergency survival overrides to factions and entities.
    """
    def __init__(self, factions: Dict[str, Faction]):
        self.factions = factions
        self.active_attractors: Dict[str, GlobalAttractorState] = {}
        # Track alliance states between factions: (faction1_id, faction2_id) -> is_allied
        self.faction_alliances: Dict[Tuple[str, str], bool] = {}

    def register_attractor(self, attractor: GlobalAttractorState):
        self.active_attractors[attractor.attractor_type.value] = attractor

    def accumulate_world_pressure(self, attractor_type: AttractorType, delta_pressure: float) -> bool:
        """
        Increases global attractor pressure (e.g., as global destruction/voids accumulate).
        Returns True if threshold is breached and trigger occurs.
        """
        attractor_key = attractor_type.value
        attractor = self.active_attractors.get(attractor_key)
        if not attractor:
            return False

        attractor.current_pressure = min(100.0, attractor.current_pressure + delta_pressure)
        if attractor.current_pressure >= attractor.threshold_pressure and not attractor.is_active:
            attractor.is_active = True
            self._trigger_macro_state_override(attractor)
            return True

        return attractor.is_active

    def _trigger_macro_state_override(self, attractor: GlobalAttractorState):
        """
        When macro threshold is breached, force emergency alliance state transitions.
        Even hostile factions (e.g., Merchant Syndicate & Underground Outlaws) enter emergency survival alliances,
        or extreme betrayal branches occur.
        """
        faction_ids = list(self.factions.keys())
        for i in range(len(faction_ids)):
            for j in range(i + 1, len(faction_ids)):
                f1_id = faction_ids[i]
                f2_id = faction_ids[j]

                # Under Demon Lord Awakening, force survival alliance across all non-demon factions
                if attractor.attractor_type == AttractorType.DEMON_LORD_AWAKENING:
                    self.faction_alliances[(f1_id, f2_id)] = True
                    self.faction_alliances[(f2_id, f1_id)] = True

                # Also elevate STR and VIT Voids in all factions due to existential threat
                f1 = self.factions[f1_id]
                f2 = self.factions[f2_id]
                f1.update_void_pressure({"STR": 50.0, "VIT": 50.0})
                f2.update_void_pressure({"STR": 50.0, "VIT": 50.0})

    def is_faction_allied(self, faction1_id: str, faction2_id: str) -> bool:
        return self.faction_alliances.get((faction1_id, faction2_id), False)
