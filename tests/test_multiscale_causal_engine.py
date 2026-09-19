"""
test_multiscale_causal_engine.py
================================
Unit tests for Elysia Multi-Scale Cosmological Causal Engine modules.
"""

import pytest
import torch

from modules.causal_game_engine.alignment_field import (
    AlignmentVector,
    AlignmentType,
    HeroAlignmentState,
    AlignmentTensorField,
    RelationState
)
from modules.causal_game_engine.multiscale_constellation import (
    MultiscaleCosmologicalEngine,
    MultiscaleConstellationNode,
    ConstellationTier,
    Tier3AffiliationType,
    Tier1PrimordialPrinciple
)
from modules.causal_game_engine.do_calculus_engine import StructuralCausalModel
from modules.causal_game_engine.causal_scm_nn import DifferentiableSCM, CausalLossCalculator
from modules.causal_game_engine.causal_prompt_decoder import CausalPromptDecoder
from modules.causal_game_engine.bidirectional_causal_loop import IntegratedBidirectionalCausalLoop


def test_multiscale_cosmology_initialization():
    tensor_field = AlignmentTensorField()
    engine = MultiscaleCosmologicalEngine(tensor_field)

    assert len(engine.tier1_principles) == 2
    assert "t1_entropy" in engine.tier1_principles
    assert "t2_pantheon_lg" in engine.constellations
    assert engine.constellations["t2_pantheon_lg"].tier == ConstellationTier.TIER_2_PANTHEON


def test_causal_subcontracting_and_tax():
    tensor_field = AlignmentTensorField()
    engine = MultiscaleCosmologicalEngine(tensor_field)

    # Test subcontracting: TIER 2 -> TIER 3
    res = engine.execute_causal_subcontract("t2_pantheon_lg", "t3_vassal_iron", 200.0)
    assert res["success"] is True
    assert res["transferred_power"] == 200.0

    # Test revelation tax
    vassal = engine.constellations["t3_vassal_iron"]
    vassal.causal_power_pool = 100.0
    tax_res = engine.process_revelation_with_tax("t3_vassal_iron", "hero_01", 50.0)
    assert tax_res["success"] is True
    assert tax_res["causal_tax_paid"] == 10.0  # 50.0 * 0.20 tax rate


def test_illegal_ascension_and_usurpation():
    tensor_field = AlignmentTensorField()
    hero = HeroAlignmentState(
        hero_id="hero_test",
        name="테스트 영웅",
        current_alignment=AlignmentVector(0.5, 0.5),
        base_anchor_alignment=AlignmentVector(0.5, 0.5)
    )
    tensor_field.register_hero(hero)
    engine = MultiscaleCosmologicalEngine(tensor_field)

    # Test illegal ascension
    res = engine.process_illegal_ascension("t3_heretical_abyss", "hero_test")
    assert res["success"] is True
    assert hero.is_deicide is True

    # Test ascension / usurpation
    asc_res = engine.check_and_trigger_ascension("const_player", share_boost=0.30)
    assert asc_res["ascended"] is True
    assert engine.constellations["const_player"].tier == ConstellationTier.TIER_2_PANTHEON


def test_do_calculus_engine():
    scm = StructuralCausalModel()
    scm.add_causal_edge("z", "x", 0.5)
    scm.add_causal_edge("x", "y", 0.8)

    scm.nodes["z"].value = 2.0
    scm._propagate_causal_effects("z")
    assert scm.nodes["x"].value == 1.0
    assert scm.nodes["y"].value == 0.8

    # Apply do(x = 5.0)
    surgered = scm.apply_do_intervention("x", 5.0)
    assert surgered.nodes["x"].value == 5.0
    assert surgered.nodes["y"].value == 4.0
    assert len(surgered.nodes["x"].parents) == 0  # Incoming edge severed!

    # ACE
    ace = scm.calculate_average_causal_effect("x", "y", val_a=10.0, val_b=0.0)
    assert ace == 8.0  # 10.0*0.8 - 0.0*0.8


def test_differentiable_scm_and_loss():
    model = DifferentiableSCM(num_nodes=4)
    loss_calc = CausalLossCalculator()

    x = torch.randn(8, 4)
    do_mask = torch.tensor([0.0, 1.0, 0.0, 0.0])
    do_values = torch.tensor([0.0, 2.5, 0.0, 0.0])

    pred = model(x, do_mask=do_mask, do_values=do_values)
    assert pred.shape == (8, 4)

    loss = loss_calc(pred, x, model.get_masked_adj())
    assert loss.item() > 0.0


def test_causal_prompt_decoder():
    decoder = CausalPromptDecoder()

    field_summary = {
        "hero_id": "hero_01",
        "alignment": {"x": 0.85, "y": -0.60},
        "relation_state": "Schism",
        "affinity_score": -0.55,
        "is_deicide": False,
        "spi_stat": 45.0
    }

    payload = decoder.build_agent_payload(field_summary, "금지된 주문을 사용한다.")
    assert "system_prompt" in payload
    assert "parameters" in payload
    assert payload["parameters"]["temperature"] == 0.85


def test_integrated_bidirectional_causal_loop():
    loop = IntegratedBidirectionalCausalLoop()
    hero = HeroAlignmentState(
        hero_id="hero_alpha",
        name="알파 영웅",
        current_alignment=AlignmentVector(0.2, 0.3),
        base_anchor_alignment=AlignmentVector(0.2, 0.3)
    )
    loop.alignment_field.register_hero(hero)

    res = loop.execute_bidirectional_step(
        hero_id="hero_alpha",
        external_action_log="영웅 알파에게 금지된 암시장 마법서를 건넨다.",
        intervention_vector=(-0.5, -0.4)
    )

    assert res["success"] is True
    assert "causal_loss" in res
    assert "agent_prompt_payload" in res
