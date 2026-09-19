"""
demo_npc_constellation_polytheism_cli.py
========================================
CLI Simulator for Elysia NPC Constellations & Polytheistic Chessboard.
Simulates 10 turns of 4-Faction Constellation Alignment Field, Revelation Auctions,
Hero Alignment Drift, Apostasy, Deicide Awakening, and Divine Diplomacy.
"""

import time
import json
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
    DivineDiplomacyEngine
)


def run_polytheistic_chessboard_cli():
    print("=" * 85)
    print(" Elysia Causal Engine: NPC Constellations & Polytheistic Chessboard Simulation ")
    print(" System: 2D Alignment Tensor Field × Revelation Auction × Divine Diplomacy ")
    print("=" * 85)

    # 1. Initialize Engines
    tensor_field = AlignmentTensorField()
    npc_manager = NPCConstellationManager(tensor_field)
    auction_house = RevelationAuctionHouse(tensor_field)
    interference_engine = CausalInterferenceEngine(tensor_field)
    diplomacy_engine = DivineDiplomacyEngine(tensor_field)

    # 2. Register Initial Heroes
    hero_a = HeroAlignmentState(
        hero_id="hero_sir_a",
        name="Sir A (3-Star Defense Captain)",
        current_alignment=AlignmentVector(x=0.6, y=0.7),
        base_anchor_alignment=AlignmentVector(x=0.6, y=0.7),
        spi_stat=70.0,
        star_rank=3,
        bound_constellation_id="const_player_lg"
    )

    hero_b = HeroAlignmentState(
        hero_id="hero_rogue_b",
        name="Shadow Rogue B (3-Star Smuggler)",
        current_alignment=AlignmentVector(x=0.2, y=-0.1),
        base_anchor_alignment=AlignmentVector(x=0.2, y=-0.1),
        spi_stat=35.0,  # Low SPI -> Vulnerable to LE Contract
        star_rank=3,
        bound_constellation_id="const_player_lg"
    )

    hero_c = HeroAlignmentState(
        hero_id="hero_slayer_c",
        name="Anonymus Hero C (3-Star Seeker)",
        current_alignment=AlignmentVector(x=0.02, y=-0.02),
        base_anchor_alignment=AlignmentVector(x=0.0, y=0.0),
        spi_stat=90.0,  # High SPI + Neutral -> Deicide Candidate
        star_rank=3,
        bound_constellation_id=None
    )

    tensor_field.register_hero(hero_a)
    tensor_field.register_hero(hero_b)
    tensor_field.register_hero(hero_c)

    print("\n[Initialization Complete]")
    print(" Registered Constellations:")
    for const in tensor_field.constellations.values():
        print(f"   - [{const.constellation_id}] {const.name} ({const.faction_type.value}) | Coord: {const.alignment.to_tuple()}")

    print("\n Registered Heroes:")
    for hero in tensor_field.heroes.values():
        print(f"   - [{hero.hero_id}] {hero.name} | Coord: {hero.current_alignment.to_tuple()} | SPI: {hero.spi_stat}")

    # 3. Simulate 10-Turn Campaign
    print("\n" + "=" * 85)
    print(" Starting 10-Turn Polytheistic Chessboard Simulation...")
    print("=" * 85)

    chronicle_events = []

    for turn in range(1, 11):
        print(f"\n>>> [TURN {turn:02d}] ----------------------------------------------------")

        # 1. Process NPC Turn Decisions
        npc_logs = npc_manager.process_npc_turn_decisions()
        for log in npc_logs:
            print(f"   👁️ [{log['constellation_name']}] {log['revelation_msg']}")

        # 2. Turn Specific Events & Ordeals
        if turn == 3:
            # Turn 3: Famine Ordeal for Rogue B -> Revelation Auction Trigger
            print("   ⚠️ EVENT: Severe Famine at South Gate! Shadow Rogue B faces critical ordeal!")
            bids = [
                BidProposal(
                    constellation_id="const_player_lg",
                    constellation_name="철혈과 수호의 성좌 (Player)",
                    causal_power_bid=15.0,
                    revelation_text="Endure famine with honor",
                    offered_artifact_name="Iron Shield",
                    alignment_resonance=0.5
                ),
                BidProposal(
                    constellation_id="const_npc_le",
                    constellation_name="심연과 계약의 성좌 (NPC-LE)",
                    causal_power_bid=60.0,
                    revelation_text="Open subterranean black market for grain smuggling",
                    offered_artifact_name="Shadow Seal",
                    alignment_resonance=0.8
                )
            ]
            auction_res = auction_house.trigger_ordeal_auction(hero_b, "CRITICAL_FAMINE", bids)
            print(f"   🔥 [REVELATION AUCTION] {auction_res['description']}")
            if auction_res["is_apostasy"]:
                event_str = f"Turn {turn:02d}: {hero_b.name} apostatized to NPC LE Constellation!"
                print(f"      ⚡ APOSTASY DETECTED: {event_str}")
                chronicle_events.append(event_str)

        elif turn == 5:
            # Turn 5: Dual Focus on Hero A by Player LG & NPC CE -> Heretical Artifact Synthesis
            print("   ⚠️ EVENT: Dual Focus! Player LG and NPC CE observe Sir A simultaneously!")
            inter_res = interference_engine.evaluate_multi_observation_focus(
                hero_a, ["const_player_lg", "const_npc_ce"]
            )
            print(f"   🌀 Causal Gravity Amplified: {inter_res['amplified_gravity']}x | Phase Distortion: {inter_res['phase_distortion']:.2f}")
            if inter_res["heretical_artifact"]:
                art = inter_res["heretical_artifact"]
                print(f"   🔥 [HERETICAL ARTIFACT SYNTHESIZED] {art.name}")
                print(f"      - Hybrid Power: {art.hybrid_domain_power}")
                print(f"      - Side Effect: {art.atypical_side_effect}")
                chronicle_events.append(f"Turn {turn:02d}: Heretical Artifact '{art.name}' synthesized for {hero_a.name}!")

        elif turn == 7:
            # Turn 7: Divine Diplomacy Trade between Player LG and NPC LE
            player_agent = npc_manager.npc_agents["const_player_lg"]
            le_agent = npc_manager.npc_agents["const_npc_le"]
            trade_res = diplomacy_engine.execute_causal_trade(
                contract_id=f"trade_{turn}",
                initiator=player_agent,
                target=le_agent,
                amount=20.0,
                resource_desc="Protect Subterranean Grain Trade Route"
            )
            print(f"   🤝 [DIVINE DIPLOMACY] {trade_res['log']}")
            chronicle_events.append(f"Turn {turn:02d}: {trade_res['log']}")

        # 3. Update Hero Alignment Drift & Evaluate Narrative Events
        for hero_id in ["hero_sir_a", "hero_rogue_b", "hero_slayer_c"]:
            drift_res = tensor_field.update_hero_alignment_drift(hero_id)
            hero = tensor_field.heroes[hero_id]
            evt = drift_res.get("triggered_event")

            print(f"   - Hero {hero.name}: Coord={hero.current_alignment.to_tuple()}, Trajectory={len(hero.trajectory_history)} steps")
            if evt:
                print(f"     ✨ [EVENT TRIGGERED] {evt['description']}")
                chronicle_events.append(f"Turn {turn:02d}: {evt['description']}")

        # 4. Display Constellation Diplomacy Matrix
        diplomacy_matrix = diplomacy_engine.update_all_diplomatic_relations()
        schism_count = sum(1 for d in diplomacy_matrix.values() if d['state'] == RelationState.SCHISM.value)
        covenant_count = sum(1 for d in diplomacy_matrix.values() if d['state'] == RelationState.COVENANT.value)
        print(f"   - Diplomacy Overview: Covenants={covenant_count}, Schisms={schism_count}")

    # 4. Display Final Chronicle History Log
    print("\n" + "=" * 85)
    print(" Chronicle History Log (사서 역사 기록) ")
    print("=" * 85)
    for idx, evt in enumerate(chronicle_events, 1):
        print(f" [{idx:02d}] {evt}")

    print("\n" + "=" * 85)
    print(" CLI Simulation Completed Successfully! ")
    print("=" * 85)


if __name__ == "__main__":
    run_polytheistic_chessboard_cli()
