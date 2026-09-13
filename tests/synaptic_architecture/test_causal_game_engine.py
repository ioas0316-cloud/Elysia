import pytest
from synaptic_architecture.causal_game_engine import (
    CausalGameMechanicsEngine,
    CCGameNode,
    PerceptualLensController,
    ScaleLevel,
    SealedAttractorVault,
)


def test_causal_game_mechanics_engine_assassination_event():
    engine = CausalGameMechanicsEngine()

    result = engine.execute_player_action(action_type="ASSASSINATE", target_id="NPC_KING_ARTHUR")

    assert result["status"] == "RESTRUCTURED_AND_RECOVERED"
    assert result["anomaly_id"] == "ANOMALY_ROYAL_CAPITAL"
    assert result["new_node_id"] == "FACTION_REGENCY_COUNCIL"
    assert result["new_rule"] == "RULE_INTERACT_WITH_REGENCY_COUNCIL"
    assert result["final_tension"] <= 0.05
    assert result["scale"] == "MACRO_KINGDOM"
    assert "FACTION_REGENCY_COUNCIL" in engine.nodes
    assert "RULE_INTERACT_WITH_REGENCY_COUNCIL" in engine.active_quest_rules
