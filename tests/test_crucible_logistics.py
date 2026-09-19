"""
test_crucible_logistics.py
===========================
Unit & Integration Test Suite for Crucible Equipment Sublimation and Tiered Logistics Chain Engine.
"""

import pytest
import numpy as np
from modules.causal_game_engine.crucible_logistics import (
    CausalVector,
    Equipment,
    HeroProfile,
    OrdealRubric,
    ConditionEvaluator,
    ChronicleHistoryLog,
    CrucibleEngine,
    ResourceTier,
    FacilityNode,
    SupplyChainGraph,
    EquipmentState,
    TieredEquipment,
    LogisticsEngine
)
from core.physics.causal_field import CausalField, InformationVoxel


def test_causal_vector_accumulation():
    gear = Equipment(id="gear1", name="Iron Sword", memory_pct=0.0)
    v1 = CausalVector(w_str=10.0, w_con=5.0)
    v2 = CausalVector(w_str=5.0, w_agi=8.0)

    gear.accumulate_memory(v1, delta_pct=50.0)
    assert gear.memory_pct == 50.0
    assert gear.causal_vector.w_str == 5.0

    gear.accumulate_memory(v2, delta_pct=50.0)
    assert gear.memory_pct == 100.0
    assert gear.causal_vector.w_str == 7.5
    assert gear.causal_vector.w_agi == 4.0

    chroma = gear.causal_vector.chromatic_signature()
    assert len(chroma) == 3
    assert pytest.approx(float(np.sum(chroma)), 1e-4) == 1.0


def test_guard_despair_sublimation_and_chronicle():
    causal_field = CausalField(dimensions=3)
    engine = CrucibleEngine(causal_field=causal_field)

    hero = HeroProfile(
        id="hero_a",
        name="Knight A",
        current_hp_pct=5.0,
        solo_defense_time_min=6.0,
        equipped_gear=Equipment(id="g1", name="Old Shield", memory_pct=100.0)
    )

    # Register hero voxel in causal field
    causal_field.add_voxel(InformationVoxel(
        id="hero_hero_a",
        content="Hero A",
        tensor=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        position=np.zeros(3, dtype=np.float32)
    ))

    sublimed_gear = engine.check_and_trigger_sublimation(hero, current_turn=10)
    assert sublimed_gear is not None
    assert sublimed_gear.is_artifact is True
    assert "[Artifact]" in sublimed_gear.name
    assert "통곡의 방패" in sublimed_gear.name
    assert sublimed_gear.domain_power_desc is not None

    logs = engine.chronicle_log.get_recent_entries()
    assert len(logs) == 1
    assert logs[0].hero_name == "Knight A"
    assert logs[0].timestamp_turn == 10
    assert logs[0].chromatic_wave_color == "RED"


def test_tragedy_revenge_sublimation():
    engine = CrucibleEngine()
    hero = HeroProfile(
        id="hero_b",
        name="Avenger B",
        mentor_slain=True,
        defeated_6star_hero=True,
        equipped_gear=Equipment(id="g2", name="Steel Blade", memory_pct=100.0)
    )

    sublimed_gear = engine.check_and_trigger_sublimation(hero, current_turn=12)
    assert sublimed_gear is not None
    assert "핏빛 유산의 가시검" in sublimed_gear.name
    assert sublimed_gear.stat_bonus["atk"] == 40.0  # 10.0 * 4.0


def test_survival_shadow_sublimation():
    engine = CrucibleEngine()
    hero = HeroProfile(
        id="hero_c",
        name="Shadow C",
        famine_active=True,
        citizens_survival_rate=1.0,
        garrison_zone="subterranean_market",
        equipped_gear=Equipment(id="g3", name="Merchant Seal", memory_pct=100.0)
    )

    sublimed_gear = engine.check_and_trigger_sublimation(hero, current_turn=15)
    assert sublimed_gear is not None
    assert "밀수꾼의 그림자 인장" in sublimed_gear.name


def test_custom_rubric_registration():
    engine = CrucibleEngine()

    def eval_mage_tower(hero: HeroProfile, gear: Equipment) -> bool:
        return gear.memory_pct >= 100.0 and hero.garrison_zone == "mage_tower"

    custom_rubric = OrdealRubric(
        id="ordeal_mage_resonance",
        name="마탑 공명형 (Mage Tower Resonance)",
        ordeal_type="CUSTOM",
        evaluator_func=eval_mage_tower,
        result_artifact_name_template="별의 아카식 지팡이 ({hero_name})",
        domain_power_desc="성채 마법 장막 반경 300% 확장 및 마나 소비 50% 절감",
        chromatic_type="BLUE"
    )

    engine.evaluator.register_rubric(custom_rubric)

    hero = HeroProfile(
        id="hero_mage",
        name="Archmage D",
        garrison_zone="mage_tower",
        equipped_gear=Equipment(id="g4", name="Wooden Staff", memory_pct=100.0)
    )

    sublimed = engine.check_and_trigger_sublimation(hero, current_turn=20)
    assert sublimed is not None
    assert "별의 아카식 지팡이" in sublimed.name


def test_logistics_supply_chain_graph():
    graph = SupplyChainGraph()
    graph.add_node(FacilityNode("n1", "Mine", "SOURCE"))
    graph.add_node(FacilityNode("n2", "Forge", "FACTORY"))
    graph.add_node(FacilityNode("n3", "Armory", "CONSUMER"))

    graph.add_route("n1", "n2")
    graph.add_route("n2", "n3")

    assert graph.is_path_connected("n1", "n3") is True

    graph.set_route_blocked("n1", "n2", blocked=True, cause="Siege Blockade")
    assert graph.is_path_connected("n1", "n3") is False


def test_equipment_fsm_paralyzed_and_sealed():
    logistics = LogisticsEngine()

    gear_magic = TieredEquipment(
        id="m1",
        name="Arcane Barrier Shield",
        tier=ResourceTier.TIER_3_MAGIC,
        magic_aura_shield=50.0
    )

    gear_artifact = TieredEquipment(
        id="a1",
        name="Shield of Lamentation",
        tier=ResourceTier.TIER_4_ARTIFACT,
        domain_power_enabled=True
    )

    logistics.register_equipment(gear_magic)
    logistics.register_equipment(gear_artifact)

    # Standard tick with inventory (Mana & Luxury available)
    res = logistics.tick_logistics_step()
    assert gear_magic.state == EquipmentState.ACTIVE
    assert gear_magic.magic_aura_shield == 50.0
    assert gear_artifact.state == EquipmentState.ACTIVE
    assert gear_artifact.domain_power_enabled is True

    # Block refinery route so no new mana crystal is produced, and set inventory to 0
    logistics.graph.set_route_blocked("mana_refinery", "mage_tower_3star", blocked=True, cause="Refinery Attack")
    logistics.inventory["mana_crystal"] = 0.0
    logistics.tick_logistics_step()
    assert gear_magic.state == EquipmentState.PARALYZED
    assert gear_magic.magic_aura_shield == 0.0

    # Deplete Fine Wine & Luxury Silk
    logistics.inventory["fine_wine"] = 0.0
    logistics.inventory["luxury_silk"] = 0.0
    logistics.tick_logistics_step()
    assert gear_artifact.state == EquipmentState.SEALED
    assert gear_artifact.domain_power_enabled is False
