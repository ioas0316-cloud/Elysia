"""
Faction Void Gradients & Dynamic Mission Emergence Architecture
==============================================================
This module defines Factions (Merchants, Mercenary Guilds, Noble Houses, Underground Outlaws, Magic Towers)
and solves their resource/capability Voids (Deficiencies).

Core Philosophy:
1. Factions experience dynamic resource/capability Voids (STR, VIT, INT, AGI, SPR, Wealth, Defense).
2. Missions are NOT hardcoded scripts. Missions emerge dynamically as drive-shaft trajectories
   connecting entity QWER capabilities to faction Voids when pressure gradients breach thresholds.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
from core.causal_world.protocol import Entity, Stats, SkillNode, SlotType


class FactionType(Enum):
    MERCHANT_SYNDICATE = "Merchant Syndicate" # High Wealth, Low Security (STR/AGI Void)
    MERCENARY_GUILD = "Mercenary Guild"       # High Strength, Low Funds/Legitimacy (Wealth/SPR Void)
    NOBLE_HOUSE = "Noble House"               # High Legitimacy, Lack of Friction Solvers (INT/STR Void)
    UNDERGROUND_OUTLAWS = "Underground Outlaws"# High Stealth/Aggro, Isolated from Law (Legitimacy Void)
    MAGIC_TOWER = "Magic Tower"               # High INT/Magic, Low Physical Defense/Raw Resources (VIT/STR Void)


@dataclass
class VoidState:
    """
    Defines the current resource/capability deficiency levels of a Faction.
    Value range: 0.0 (No deficiency / Saturated) to 100.0 (Extreme Void / Desperate Need).
    """
    str_void: float = 0.0     # Need for raw force / physical protection
    vit_void: float = 0.0     # Need for endurance / defense / raw materials
    int_void: float = 0.0     # Need for information / intelligence / spellcraft
    agi_void: float = 0.0     # Need for mobility / swift transportation / escorts
    spr_void: float = 0.0     # Need for legitimacy / morale / diplomacy
    wealth_void: float = 0.0  # Need for gold / financial solvency

    def get_highest_void(self) -> Tuple[str, float]:
        voids = {
            "STR": self.str_void,
            "VIT": self.vit_void,
            "INT": self.int_void,
            "AGI": self.agi_void,
            "SPR": self.spr_void,
            "WEALTH": self.wealth_void
        }
        highest_stat = max(voids, key=voids.get)
        return highest_stat, voids[highest_stat]


@dataclass
class DynamicMission:
    mission_id: str
    title: str
    issuing_faction_id: str
    target_void_type: str
    gradient_pressure: float
    description: str
    reward_wealth: float
    reward_reputation: float
    required_stat_type: str
    required_min_capability: float
    is_active: bool = True
    is_completed: bool = False
    assigned_entity_id: Optional[str] = None


class Faction:
    """
    Represents a major political, economic, or physical power in the world.
    Each faction consumes energy and suffers Voids (Deficiencies) that drive mission generation.
    """
    def __init__(
        self,
        faction_id: str,
        name: str,
        faction_type: FactionType,
        base_wealth: float = 1000.0,
        initial_voids: Optional[VoidState] = None
    ):
        self.faction_id = faction_id
        self.name = name
        self.faction_type = faction_type
        self.wealth = base_wealth
        self.void_state = initial_voids if initial_voids else VoidState()
        self.active_missions: Dict[str, DynamicMission] = {}
        self.member_ids: List[str] = []

    def update_void_pressure(self, delta_void: Dict[str, float]):
        """Adjust void levels in response to world friction, disasters, or economic consumption."""
        self.void_state.str_void = max(0.0, min(100.0, self.void_state.str_void + delta_void.get("STR", 0.0)))
        self.void_state.vit_void = max(0.0, min(100.0, self.void_state.vit_void + delta_void.get("VIT", 0.0)))
        self.void_state.int_void = max(0.0, min(100.0, self.void_state.int_void + delta_void.get("INT", 0.0)))
        self.void_state.agi_void = max(0.0, min(100.0, self.void_state.agi_void + delta_void.get("AGI", 0.0)))
        self.void_state.spr_void = max(0.0, min(100.0, self.void_state.spr_void + delta_void.get("SPR", 0.0)))
        self.void_state.wealth_void = max(0.0, min(100.0, self.void_state.wealth_void + delta_void.get("WEALTH", 0.0)))

    def evaluate_and_generate_missions(self, threshold_pressure: float = 30.0) -> List[DynamicMission]:
        """
        Calculates Void Gradients. When deficiency pressure > threshold,
        a new mission is dynamically generated.
        """
        generated_missions = []
        void_map = {
            "STR": (self.void_state.str_void, "Caravan Escort / Force Suppression", "Escort trade routes or clear dangerous monster nests."),
            "VIT": (self.void_state.vit_void, "Defense & Infrastructure Supply", "Supply durable goods and reinforce fortifications."),
            "INT": (self.void_state.int_void, "Arcane Intelligence & Recon", "Gather intelligence or resolve arcane anomalies."),
            "AGI": (self.void_state.agi_void, "Swift Dispatch & Smuggling", "Deliver urgent dispatches or navigate dangerous routes."),
            "SPR": (self.void_state.spr_void, "Diplomatic Mediation & Public Order", "Restore public trust or negotiate alliance treaties."),
            "WEALTH": (self.void_state.wealth_void, "Financial Insolvency Relief", "Provide emergency capital loans or loot recovery.")
        }

        for stat, (pressure, title_template, desc) in void_map.items():
            if pressure >= threshold_pressure:
                mission_id = f"mission_{self.faction_id}_{stat}_{int(pressure)}"
                if mission_id not in self.active_missions:
                    mission = DynamicMission(
                        mission_id=mission_id,
                        title=f"[{self.name}] {title_template}",
                        issuing_faction_id=self.faction_id,
                        target_void_type=stat,
                        gradient_pressure=pressure,
                        description=desc,
                        reward_wealth=pressure * 10.0,
                        reward_reputation=10.0 if self.faction_type != FactionType.UNDERGROUND_OUTLAWS else -15.0,
                        required_stat_type=stat,
                        required_min_capability=pressure * 0.3
                    )
                    self.active_missions[mission_id] = mission
                    generated_missions.append(mission)

        return generated_missions


class VoidGradientSolver:
    """
    Connects Entity QWER capabilities to Faction Void Gradients.
    Acts as the 'Drive Shaft' mapping entity stats/skills to dynamic missions.
    """
    def __init__(self, factions: Dict[str, Faction]):
        self.factions = factions

    def match_entity_to_missions(self, entity: Entity) -> List[Tuple[DynamicMission, float]]:
        """
        Evaluates how well an entity's capability matches active faction missions.
        Returns sorted list of (Mission, MatchScore).
        """
        matches = []
        entity_stats = entity.stats.to_dict()

        for faction in self.factions.values():
            for mission in faction.active_missions.values():
                if not mission.is_active or mission.is_completed:
                    continue

                req_stat = mission.required_stat_type
                if req_stat in entity_stats:
                    entity_cap = entity_stats[req_stat]
                    if entity_cap >= mission.required_min_capability:
                        # Match score scales with void pressure and entity capability
                        match_score = (entity_cap / max(1.0, mission.required_min_capability)) * mission.gradient_pressure
                        matches.append((mission, match_score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def fulfill_mission(self, entity: Entity, mission: DynamicMission) -> bool:
        """
        Fulfills a dynamic mission.
        Transfers wealth reward to entity, updates reputation, and relieves Faction Void.
        """
        faction = self.factions.get(mission.issuing_faction_id)
        if not faction or not mission.is_active or mission.is_completed:
            return False

        # Relief Void
        void_relief = {mission.target_void_type: -mission.gradient_pressure * 0.8}
        faction.update_void_pressure(void_relief)

        # Reward Entity
        entity.wealth += mission.reward_wealth
        entity.update_reputation(mission.reward_reputation)

        # Complete Mission
        mission.is_completed = True
        mission.is_active = False
        mission.assigned_entity_id = entity.entity_id

        return True
