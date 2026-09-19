"""
test_alignment_and_npc_constellations.py
=========================================
Unit and Integration Test Suite for Elysia Alignment Tensor Field,
NPC Constellations, Revelation Auctions, Causal Interference, and Divine Diplomacy.
"""

import pytest
import math
from modules.causal_game_engine.alignment_field import (
    AlignmentVector,
    AlignmentType,
    RelationState,
    ConstellationNode,
    HeroAlignmentState,
    AlignmentTensorField
)
from modules.causal_game_engine.npc_constellation_engine import (
    ConstellationAgent,
    ConstellationArchetype,
    NPCConstellationManager
)
from modules.causal_game_engine.revelation_auction import (
    BidProposal,
    RevelationAuctionHouse
)
from modules.causal_game_engine.causal_interference import (
    CausalInterferenceEngine,
    HereticalArtifact
)
from modules.causal_game_engine.divine_diplomacy import (
    DivineDiplomacyEngine,
    TradeContract
)


def test_alignment_vector_distance_and_clamp():
    v1 = AlignmentVector(x=0.5, y=0.5)
    v2 = AlignmentVector(x=-0.5, y=-0.5)
    dist = v1.distance_to(v2)
    assert abs(dist - math.sqrt(2.0)) < 1e-4

    v_overflow = AlignmentVector(x=2.5, y=-3.0)
    v_overflow.clamp()
    assert v_overflow.x == 1.0
    assert v_overflow.y == -1.0


def test_constellation_affinity_and_diplomatic_states():
    tensor_field = AlignmentTensorField()

    c_lg = ConstellationNode(
        constellation_id="lg",
        name="Lawful Good",
        alignment=AlignmentVector(x=0.8, y=0.8),
        is_player=True
    )
    c_cg = ConstellationNode(
        constellation_id="cg",
        name="Chaotic Good",
        alignment=AlignmentVector(x=-0.8, y=0.7)
    )
    c_ce = ConstellationNode(
        constellation_id="ce",
        name="Chaotic Evil",
        alignment=AlignmentVector(x=-0.9, y=-0.8)
    )

    tensor_field.register_constellation(c_lg)
    tensor_field.register_constellation(c_cg)
    tensor_field.register_constellation(c_ce)

    # LG vs CG
    aff_cg, state_cg = tensor_field.calculate_constellation_affinity("lg", "cg")
    assert state_cg in [RelationState.ENTENTE, RelationState.FRICTION]

    # LG vs CE (Extreme Opposite -> Schism)
    aff_ce, state_ce = tensor_field.calculate_constellation_affinity("lg", "ce")
    assert state_ce == RelationState.SCHISM
    assert aff_ce < -0.40


def test_hero_alignment_drift_equation():
    tensor_field = AlignmentTensorField()
    manager = NPCConstellationManager(tensor_field)

    hero = HeroAlignmentState(
        hero_id="hero_1",
        name="Sir Guard A",
        current_alignment=AlignmentVector(x=0.0, y=0.0),
        base_anchor_alignment=AlignmentVector(x=0.0, y=0.0),
        spi_stat=50.0
    )
    tensor_field.register_hero(hero)

    # Tick 1: Drift under 4-faction field
    res = tensor_field.update_hero_alignment_drift("hero_1")
    assert "dh_vector" in res
    assert len(hero.trajectory_history) == 2  # initial + 1 update


def test_revelation_auction_and_apostasy():
    tensor_field = AlignmentTensorField()
    manager = NPCConstellationManager(tensor_field)
    auction_house = RevelationAuctionHouse(tensor_field)

    hero = HeroAlignmentState(
        hero_id="hero_guard",
        name="Guard Captain B",
        current_alignment=AlignmentVector(x=0.7, y=0.7),
        base_anchor_alignment=AlignmentVector(x=0.7, y=0.7),
        spi_stat=30.0,  # Low SPI -> Easy Temptation
        bound_constellation_id="const_player_lg"
    )
    tensor_field.register_hero(hero)

    bids = [
        BidProposal(
            constellation_id="const_player_lg",
            constellation_name="Player LG",
            causal_power_bid=10.0,
            revelation_text="Hold wall",
            offered_artifact_name="Old Shield",
            alignment_resonance=0.9
        ),
        BidProposal(
            constellation_id="const_npc_le",
            constellation_name="Contract LE",
            causal_power_bid=80.0,  # High power bid
            revelation_text="Sign dark pact",
            offered_artifact_name="Shadow Dagger",
            alignment_resonance=0.4
        )
    ]

    auction_res = auction_house.trigger_ordeal_auction(
        hero=hero,
        ordeal_type="CRITICAL_FAMINE",
        bids=bids
    )

    assert auction_res["auction_status"] == "COMPLETED"
    assert auction_res["is_apostasy"] is True
    assert hero.bound_constellation_id == "const_npc_le"


def test_causal_interference_and_heretical_artifact():
    tensor_field = AlignmentTensorField()
    manager = NPCConstellationManager(tensor_field)
    interference_engine = CausalInterferenceEngine(tensor_field)

    hero = HeroAlignmentState(
        hero_id="hero_rogue",
        name="Rogue C",
        current_alignment=AlignmentVector(x=0.0, y=0.0),
        base_anchor_alignment=AlignmentVector(x=0.0, y=0.0)
    )
    tensor_field.register_hero(hero)

    eval_res = interference_engine.evaluate_multi_observation_focus(
        hero=hero,
        focusing_constellation_ids=["const_player_lg", "const_npc_ce"]
    )

    assert eval_res["interference_active"] is True
    assert eval_res["amplified_gravity"] > 1.0
    assert eval_res["phase_distortion"] > 0.3
    assert isinstance(eval_res["heretical_artifact"], HereticalArtifact)


def test_divine_diplomacy_and_trade():
    tensor_field = AlignmentTensorField()
    manager = NPCConstellationManager(tensor_field)
    diplomacy_engine = DivineDiplomacyEngine(tensor_field)

    player_agent = manager.npc_agents["const_player_lg"]
    le_agent = manager.npc_agents["const_npc_le"]

    initial_player_power = player_agent.causal_power_pool
    initial_le_power = le_agent.causal_power_pool

    trade_res = diplomacy_engine.execute_causal_trade(
        contract_id="contract_001",
        initiator=player_agent,
        target=le_agent,
        amount=20.0,
        resource_desc="Protect Subterranean Black Market"
    )

    assert trade_res["success"] is True
    assert player_agent.causal_power_pool == initial_player_power - 20.0
    assert le_agent.causal_power_pool == initial_le_power + 20.0


def test_deicide_awakening_event():
    tensor_field = AlignmentTensorField()
    manager = NPCConstellationManager(tensor_field)

    hero_neutral = HeroAlignmentState(
        hero_id="hero_slayer",
        name="Anonymus Slayer",
        current_alignment=AlignmentVector(x=0.01, y=-0.01),
        base_anchor_alignment=AlignmentVector(x=0.0, y=0.0),
        spi_stat=95.0  # High SPI + Near Center (0,0)
    )
    tensor_field.register_hero(hero_neutral)

    drift_res = tensor_field.update_hero_alignment_drift("hero_slayer")
    assert hero_neutral.is_deicide is True
    assert drift_res["triggered_event"]["event_type"] == "DEICIDE_AWAKENING"
