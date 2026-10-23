"""
logistics_engine.py
===================
Tiered Equipment Production Logistics Chain & Tech Tree Simulation Engine.

Implements:
1. 4-Tier Supply Chain & Tech Tree:
   - Tier 1: Common (Iron Ore, Wood -> Steel/Weapon -> Common Gear)
   - Tier 2: Advanced (Coal, Tanned Leather -> Steel Ingot, Leather -> Advanced Gear)
   - Tier 3: Rare & Magic (Mana Crystals, Fine Wine -> Mana Essence, Precision Gear -> Magic Gear)
   - Tier 4: Artifact & Legendary (Grand Mage Tower, Contraband, Luxury Wine/Silk -> Artifact/Legendary Gear)
2. Graph Connectivity Supply Chain:
   - Nodes: Raw Material Sources, Processing Facilities, Consumer Hubs
   - Edges: Logistics Routes (connected/broken/blocked)
3. Equipment Finite State Machine (FSM):
   - States: ACTIVE, PARALYZED, SEALED
   - Transitions driven by supply availability (Mana Essence, Luxury Wine/Silk)
4. Strategic Loop Cycle Tracking:
   - Early -> Mid -> Late game strategic loop evaluation
"""

import enum
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field


class ResourceTier(enum.Enum):
    TIER_1_COMMON = 1      # Iron Ore, Wood
    TIER_2_ADVANCED = 2    # Coal, Steel Ingot, Processed Leather
    TIER_3_MAGIC = 3       # Mana Crystal, Mana Essence, Fine Wine
    TIER_4_ARTIFACT = 4    # Luxury Silk, Contraband, Artifact Materials


class EquipmentState(enum.Enum):
    ACTIVE = "ACTIVE"          # Fully operational
    PARALYZED = "PARALYZED"    # Magic Aura paralyzed due to Mana Crystal deficit
    SEALED = "SEALED"          # Artifact Powers sealed due to Luxury supply deficit


@dataclass
class FacilityNode:
    id: str
    name: str
    node_type: str  # "SOURCE", "FACTORY", "HUB", "CONSUMER"
    tier_level: int = 1
    input_resources: Dict[str, float] = field(default_factory=dict)
    output_resources: Dict[str, float] = field(default_factory=dict)
    is_operational: bool = True


@dataclass
class LogisticsRoute:
    source_id: str
    target_id: str
    capacity: float = 100.0
    is_blocked: bool = False
    disruption_cause: Optional[str] = None


class SupplyChainGraph:
    """Graph structure representing supply nodes and logistics routes."""
    def __init__(self):
        self.nodes: Dict[str, FacilityNode] = {}
        self.routes: List[LogisticsRoute] = []

    def add_node(self, node: FacilityNode):
        self.nodes[node.id] = node

    def add_route(self, source_id: str, target_id: str, capacity: float = 100.0):
        self.routes.append(LogisticsRoute(source_id=source_id, target_id=target_id, capacity=capacity))

    def set_route_blocked(self, source_id: str, target_id: str, blocked: bool = True, cause: str = "Siege Blockade"):
        for r in self.routes:
            if r.source_id == source_id and r.target_id == target_id:
                r.is_blocked = blocked
                r.disruption_cause = cause if blocked else None

    def is_path_connected(self, start_id: str, end_id: str) -> bool:
        """BFS path connectivity check considering blocked routes."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return False

        visited = set()
        queue = [start_id]

        while queue:
            curr = queue.pop(0)
            if curr == end_id:
                return True

            visited.add(curr)
            for r in self.routes:
                if r.source_id == curr and not r.is_blocked:
                    neighbor = r.target_id
                    if neighbor not in visited and neighbor not in queue:
                        queue.append(neighbor)
        return False


@dataclass
class TieredEquipment:
    id: str
    name: str
    tier: ResourceTier
    state: EquipmentState = EquipmentState.ACTIVE
    base_atk_bonus: float = 10.0
    magic_aura_shield: float = 0.0
    domain_power_enabled: bool = True

    def update_state(self, mana_available: bool, luxury_available: bool) -> EquipmentState:
        """
        Equipment FSM State Transitions:
        - Magic Gear: requires mana_available. If False -> PARALYZED (magic aura = 0)
        - Artifact/Legendary Gear: requires luxury_available. If False -> SEALED (domain power disabled)
        """
        if self.tier == ResourceTier.TIER_3_MAGIC:
            if not mana_available:
                self.state = EquipmentState.PARALYZED
                self.magic_aura_shield = 0.0
            else:
                self.state = EquipmentState.ACTIVE
                self.magic_aura_shield = 50.0

        elif self.tier == ResourceTier.TIER_4_ARTIFACT:
            if not luxury_available:
                self.state = EquipmentState.SEALED
                self.domain_power_enabled = False
            else:
                self.state = EquipmentState.ACTIVE
                self.domain_power_enabled = True

        else:
            self.state = EquipmentState.ACTIVE

        return self.state


class LogisticsEngine:
    """
    Logistics Engine managing production, consumption, graph routes,
    and equipment maintenance states.
    """
    def __init__(self):
        self.graph = SupplyChainGraph()
        self.inventory: Dict[str, float] = {
            "iron_ore": 50.0,
            "wood": 50.0,
            "coal": 20.0,
            "leather": 20.0,
            "mana_crystal": 10.0,
            "fine_wine": 5.0,
            "luxury_silk": 5.0,
            "steel_ingot": 10.0,
            "mana_essence": 5.0,
            "common_gear": 10.0,
            "advanced_gear": 5.0,
            "magic_gear": 2.0,
            "artifact_gear": 1.0
        }
        self.equipment_registry: Dict[str, TieredEquipment] = {}
        self._init_default_supply_graph()

    def _init_default_supply_graph(self):
        # 1-Star Raw Sources
        self.graph.add_node(FacilityNode("mine_iron", "1-Star Iron Mine", "SOURCE", tier_level=1))
        self.graph.add_node(FacilityNode("lumber_mill", "1-Star Lumber Mill", "SOURCE", tier_level=1))
        self.graph.add_node(FacilityNode("mana_refinery", "3-Star Mana Refinery", "SOURCE", tier_level=3))

        # 2-Star Processing
        self.graph.add_node(FacilityNode("forge_1star", "1-Star Forge", "FACTORY", tier_level=1))
        self.graph.add_node(FacilityNode("workshop_2star", "2-Star Workshop", "FACTORY", tier_level=2))
        self.graph.add_node(FacilityNode("mage_tower_3star", "3-Star Mage Tower", "FACTORY", tier_level=3))
        self.graph.add_node(FacilityNode("black_market_4star", "4-Star Subterranean Black Market", "FACTORY", tier_level=4))

        # Consumers
        self.graph.add_node(FacilityNode("east_gate_armory", "East Gate Armory", "CONSUMER", tier_level=1))
        self.graph.add_node(FacilityNode("hero_quarter", "Hero Quarters", "CONSUMER", tier_level=3))

        # Routes
        self.graph.add_route("mine_iron", "forge_1star")
        self.graph.add_route("lumber_mill", "forge_1star")
        self.graph.add_route("forge_1star", "workshop_2star")
        self.graph.add_route("mana_refinery", "mage_tower_3star")
        self.graph.add_route("workshop_2star", "east_gate_armory")
        self.graph.add_route("mage_tower_3star", "hero_quarter")
        self.graph.add_route("black_market_4star", "hero_quarter")

    def register_equipment(self, gear: TieredEquipment):
        self.equipment_registry[gear.id] = gear

    def tick_logistics_step(self) -> Dict[str, Any]:
        """
        Executes 1 tick (turn/month) of raw material production, processing,
        consumption, and equipment FSM updates.
        """
        # 1. Raw Materials Production if connected to factories
        if self.graph.is_path_connected("mine_iron", "forge_1star"):
            self.inventory["iron_ore"] += 10.0
            self.inventory["wood"] += 10.0
        else:
            # Path broken!
            pass

        if self.graph.is_path_connected("mana_refinery", "mage_tower_3star"):
            self.inventory["mana_crystal"] += 5.0
            self.inventory["mana_essence"] += 3.0

        # 2. Consumption for Equipment Maintenance
        mana_supplied = False
        if self.inventory["mana_crystal"] >= 2.0:
            self.inventory["mana_crystal"] -= 2.0
            mana_supplied = True

        luxury_supplied = False
        if self.inventory["fine_wine"] >= 1.0 and self.inventory["luxury_silk"] >= 1.0:
            self.inventory["fine_wine"] -= 1.0
            self.inventory["luxury_silk"] -= 1.0
            luxury_supplied = True

        # 3. Update Equipment FSM States
        fsm_statuses = {}
        for gear_id, gear in self.equipment_registry.items():
            new_state = gear.update_state(mana_available=mana_supplied, luxury_available=luxury_supplied)
            fsm_statuses[gear_id] = {
                "name": gear.name,
                "tier": gear.tier.name,
                "state": new_state.value,
                "magic_aura": gear.magic_aura_shield,
                "domain_power_enabled": gear.domain_power_enabled
            }

        return {
            "inventory": dict(self.inventory),
            "mana_supplied": mana_supplied,
            "luxury_supplied": luxury_supplied,
            "fsm_statuses": fsm_statuses,
            "strategic_phase": self.get_strategic_phase()
        }

    def get_strategic_phase(self) -> str:
        """Determines current strategic loop phase based on logistics progression."""
        if self.inventory["magic_gear"] > 0 or self.inventory["mana_crystal"] > 10.0:
            return "LATE_GAME_MAGE_TOWER_HEGEMONY"
        elif self.inventory["advanced_gear"] > 5.0:
            return "MID_GAME_HERO_ORDEAL_SUBLIMATION"
        else:
            return "EARLY_GAME_MINES_AND_FORGE_DEFENSE"
