"""
Unit and Integration Tests for Project Elysia 5D Unified Phase Space Engine
=============================================================================
"""

import pytest
import numpy as np
from elysia_core.unified_phase_space import (
    clip_vector,
    EnvironmentPotential,
    ResourceEntity,
    CraftedItemEntity,
    MonsterEntity,
    NPCEntity,
    ElysiaUnifiedEngine
)


def test_clip_vector_bounds():
    """Verify vector component clipping to [-1.0, 1.0]."""
    raw_vec = np.array([2.5, -3.0, 0.5, 1.0, -1.0])
    clipped = clip_vector(raw_vec)
    assert np.array_equal(clipped, np.array([1.0, -1.0, 0.5, 1.0, -1.0]))


def test_environment_potential_shift():
    """Verify shift_zeitgeist shifts vector correctly and stays bounded."""
    env = EnvironmentPotential("Peace Era", np.array([0.0, 0.0, 0.5, 0.5, 0.0]))
    new_vec = env.shift_zeitgeist(np.array([0.8, -0.8, 1.0, -2.0, 0.2]))
    assert env.vector[0] == pytest.approx(0.8)
    assert env.vector[1] == pytest.approx(-0.8)
    assert env.vector[2] == pytest.approx(1.0)
    assert env.vector[3] == pytest.approx(-1.0)  # clipped from -1.5
    assert env.vector[4] == pytest.approx(0.2)


def test_resource_adaptation_and_tags():
    """Verify resource adaptation to environment and tag description."""
    res = ResourceEntity("Iron Ore", np.array([0.0, 0.5, 0.0, 0.0, 0.0]))
    env_vec = np.array([1.0, 0.0, 1.0, 0.0, 1.0])  # High heat, high structure, high resonance
    res.adapt_to_environment(env_vec, coupling=0.5)

    # 0.0 + 0.5*1.0 = 0.5 for Heat, 0.5 for Struct, 0.5 for Res
    desc = res.get_description()
    assert "화염이 깃든" in desc
    assert "고밀도" in desc
    assert "정제된 결정성" in desc
    assert "위상 공명하는" in desc


def test_crafted_item_superposition_and_quantum_leap():
    """Verify linear superposition of resource vectors and quantum leap energy calculation."""
    res1_vec = np.array([0.6, 0.2, 0.8, 0.0, 0.4])
    res2_vec = np.array([0.4, -0.2, 0.2, 0.6, 0.0])

    item = CraftedItemEntity("Magic Staff", [res1_vec, res2_vec], energy_shell_n=1, base_E0=1000.0)
    expected_vec = np.array([0.5, 0.0, 0.5, 0.3, 0.2])
    assert np.allclose(item.I_vector, expected_vec)
    assert item.get_tier_label() == "Common"

    # Delta E for n=1: 1000 * (1 - 1/4) = 750.0
    cost_n1 = item.calculate_quantum_leap_cost()
    assert cost_n1 == pytest.approx(750.0)

    # Upgrade to n=2
    item.upgrade_tier()
    assert item.energy_shell_n == 2
    assert item.get_tier_label() == "Rare/Epic"
    # Delta E for n=2: 1000 * (1/4 - 1/9) = 1000 * 5/36 = 138.888...
    cost_n2 = item.calculate_quantum_leap_cost()
    assert cost_n2 == pytest.approx(138.8888, rel=1e-3)


def test_monster_mutation_and_phase_patterns():
    """Verify monster ecology mutation and pattern shift."""
    boss = MonsterEntity("Gargoyle", np.array([0.0, 0.0, 0.0, 0.0, 0.0]), energy_shell=3)
    status_p1 = boss.get_status()
    assert status_p1["title"] == "지역 월드 보스 [Gargoyle]"
    assert "1페이즈" in status_p1["pattern"]

    # Mutate with high ferocity and structure environment
    env_vec = np.array([0.9, 0.5, 0.8, 0.0, 0.0])
    boss.mutate(env_vec, env_weight=0.8)
    status_p3 = boss.get_status()
    assert "3페이즈" in status_p3["pattern"]


def test_npc_dissonance_and_speech():
    """Verify NPC cognitive dissonance stress calculation and speech pattern dynamics."""
    npc = NPCEntity("Knight Captain", np.array([0.0, -0.3, 0.8, 0.8, 0.0]), relaxation_rate=0.2)

    # Environment far from NPC phase (High dissonance)
    harsh_env = np.array([1.0, 0.8, -0.8, -0.8, 0.5])
    dissonance = npc.evaluate_dissonance(harsh_env)
    assert dissonance > 1.5

    speech = npc.get_speech_pattern()
    assert "경고: 심각한 사상적 이탈" in speech


def test_unified_engine_cascade_and_dashboard():
    """Verify ElysiaUnifiedEngine step_cascade and dashboard report rendering."""
    engine = ElysiaUnifiedEngine("Peace Era", np.array([0.0, 0.0, 0.5, 0.5, 0.0]))
    res = ResourceEntity("Iron Ore", np.array([0.1, 0.2, 0.1, 0.0, 0.0]))
    npc = NPCEntity("Villager", np.array([0.0, 0.0, 0.5, 0.5, 0.0]))
    mon = MonsterEntity("Wolf", np.array([0.5, 0.5, 0.0, 0.0, 0.0]), energy_shell=1)
    item = CraftedItemEntity("Iron Sword", [res.base_R], energy_shell_n=1)

    engine.add_resource(res)
    engine.add_npc(npc)
    engine.add_monster(mon)
    engine.add_crafted_item(item)

    # Initial cascade step
    snapshot = engine.step_cascade()
    assert len(snapshot["resources"]) == 1
    assert len(snapshot["npcs"]) == 1
    assert len(snapshot["monsters"]) == 1
    assert len(snapshot["crafted_items"]) == 1

    # Render dashboard report
    report = engine.render_one_page_dashboard("[TEST]")
    assert "PROJECT ELYSIA :: ONE-PAGE SYSTEM DESIGN DASHBOARD [TEST]" in report
    assert "MACRO ZEITGEIST POTENTIAL FIELD" in report
    assert "Iron Ore" in report
    assert "Villager" in report
    assert "Wolf" in report
    assert "Iron Sword" in report
