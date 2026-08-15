"""
Rumor Propagation & Noise Dynamics Architecture
==============================================
This module defines the spatial, carrier-based rumor propagation network across NPCs and regions.

Core Philosophy:
1. Rumors are NOT instant global state updates. Rumors are wave-packets carried physically by NPCs.
2. Distance, INT/SPR traits of carriers, and time introduce Noise (Distortion / Exaggeration).
3. Rumors trigger Hero Path vs. Outlaw Path state transitions in regions (Guard Aggro, Black Market Unlocks).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
import random
from core.causal_world.protocol import Entity


@dataclass
class RumorPacket:
    rumor_id: str
    originator_id: str
    target_entity_id: str          # Entity the rumor is about (e.g. Player)
    raw_action_type: str           # e.g., "PIRACY_ATTACK", "CARAVAN_RESCUE"
    reputation_delta: float        # Original reputation impact (-: outlaw, +: hero)
    aggressiveness: float         # Original aggressiveness
    current_location: Tuple[float, float]
    carrier_id: str
    timestamp: float
    distortion_count: int = 0
    perceived_severity: float = 1.0 # Multiplier due to exaggeration / noise


class SpatialRegion:
    """
    Represents a village, town, kingdom capital, or underground territory.
    Maintains guards, law compliance level, and local NPC perception of entities.
    """
    def __init__(self, region_id: str, name: str, center_pos: Tuple[float, float], radius: float = 50.0):
        self.region_id = region_id
        self.name = name
        self.center_pos = center_pos
        self.radius = radius

        # Local Perception Map: target_entity_id -> perceived_reputation
        self.entity_perceptions: Dict[str, float] = {}

        # Local Guard & Black Market state
        self.guard_hostility: Dict[str, bool] = {}       # target_id -> is_hostile
        self.black_market_unlocked: Dict[str, bool] = {}  # target_id -> is_unlocked
        self.honored_hero_quests: Dict[str, bool] = {}    # target_id -> possesses_hero_access

    def is_pos_in_region(self, pos: Tuple[float, float]) -> bool:
        dx = pos[0] - self.center_pos[0]
        dy = pos[1] - self.center_pos[1]
        return math.sqrt(dx*dx + dy*dy) <= self.radius

    def receive_rumor_packet(self, packet: RumorPacket):
        """
        Processes an arriving rumor packet. Updates local perception and state transitions.
        """
        target_id = packet.target_entity_id
        current_rep = self.entity_perceptions.get(target_id, 0.0)

        # Apply distorted reputation delta
        effective_delta = packet.reputation_delta * packet.perceived_severity
        new_rep = current_rep + effective_delta
        self.entity_perceptions[target_id] = new_rep

        # State Transition Evaluation
        if new_rep <= -40.0:
            # Outlaw Path Transition: Guards hostile, Black Market unlocked
            self.guard_hostility[target_id] = True
            self.black_market_unlocked[target_id] = True
            self.honored_hero_quests[target_id] = False
        elif new_rep >= 40.0:
            # Hero Path Transition: Guards honor, Hero quests unlocked
            self.guard_hostility[target_id] = False
            self.black_market_unlocked[target_id] = False
            self.honored_hero_quests[target_id] = True
        else:
            # Neutral State
            self.guard_hostility[target_id] = False
            self.black_market_unlocked[target_id] = False
            self.honored_hero_quests[target_id] = False


class RumorPropagationNetwork:
    """
    Simulates the physical propagation and distortion of rumors across NPC carriers and regions.
    """
    def __init__(self, regions: List[SpatialRegion]):
        self.regions = {r.region_id: r for r in regions}
        self.active_rumors: List[RumorPacket] = []

    def create_rumor_from_action(
        self,
        originator_id: str,
        target_entity_id: str,
        action_type: str,
        reputation_delta: float,
        aggressiveness: float,
        origin_pos: Tuple[float, float],
        timestamp: float
    ) -> RumorPacket:
        """Instantiates a fresh rumor packet at the location of an action."""
        rumor = RumorPacket(
            rumor_id=f"rumor_{timestamp}_{originator_id}_{target_entity_id}",
            originator_id=originator_id,
            target_entity_id=target_entity_id,
            raw_action_type=action_type,
            reputation_delta=reputation_delta,
            aggressiveness=aggressiveness,
            current_location=origin_pos,
            carrier_id=originator_id,
            timestamp=timestamp
        )
        self.active_rumors.append(rumor)

        # Immediate impact on local region
        for region in self.regions.values():
            if region.is_pos_in_region(origin_pos):
                region.receive_rumor_packet(rumor)

        return rumor

    def propagate_and_distort(
        self,
        carrier_npc: Entity,
        destination_pos: Tuple[float, float],
        current_time: float,
        rng_seed: Optional[int] = None
    ):
        """
        Moves active rumors carried by carrier_npc toward destination_pos.
        Applies Noise/Distortion based on carrier's INT (Intelligence) and distance traveled.
        """
        if rng_seed is not None:
            random.seed(rng_seed)

        for rumor in self.active_rumors:
            if rumor.carrier_id == carrier_npc.entity_id or carrier_npc.is_pos_in_range(rumor.current_location, range_dist=100.0):
                # Calculate travel distance noise
                dx = destination_pos[0] - rumor.current_location[0]
                dy = destination_pos[1] - rumor.current_location[1]
                dist = math.sqrt(dx*dx + dy*dy)

                # Low INT carriers cause higher exaggeration / noise
                int_stat = carrier_npc.stats.INT
                distortion_factor = (1.0 + (dist / 100.0)) * (1.5 - (int_stat / 30.0))

                # Update rumor state
                rumor.current_location = destination_pos
                rumor.carrier_id = carrier_npc.entity_id
                rumor.distortion_count += 1
                rumor.perceived_severity *= max(0.5, min(3.0, distortion_factor))

                # Deliver to new region if entered
                for region in self.regions.values():
                    if region.is_pos_in_region(destination_pos):
                        region.receive_rumor_packet(rumor)


# Helper method addition for Entity position checks
def is_pos_in_range(self, other_pos: Tuple[float, float], range_dist: float) -> bool:
    dx = self.position[0] - other_pos[0]
    dy = self.position[1] - other_pos[1]
    return math.sqrt(dx*dx + dy*dy) <= range_dist

Entity.is_pos_in_range = is_pos_in_range
