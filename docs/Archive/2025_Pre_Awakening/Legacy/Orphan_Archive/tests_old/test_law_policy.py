from pathlib import Path

import pytest

from Core.FoundationLayer.Foundation.core.world import World
from Core.FoundationLayer.Foundation.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager


def _build_world(tmp_path: Path) -> World:
    kg_file = tmp_path / "law_policy_kg.json"
    kg_manager = KGManager(filepath=kg_file)
    wave_mechanics = WaveMechanics(kg_manager)
    return World({}, wave_mechanics)


def test_altruism_policy_enqueue_share(tmp_path: Path):
    world = _build_world(tmp_path)
    world.add_cell("actor_1", properties={"label": "human", "culture": "wuxia"})
    world.add_cell("kin_1", properties={"label": "human", "culture": "wuxia"})

    actor_idx = world.id_to_idx["actor_1"]
    kin_idx = world.id_to_idx["kin_1"]
    world.hunger[actor_idx] = 90.0
    world.hunger[kin_idx] = 10.0
    world.adjacency_matrix[actor_idx, kin_idx] = 1.0

    action = world._decide_social_or_combat_action(actor_idx, world.adjacency_matrix.tocsr())
    assert action == (kin_idx, "share_food", None)

    reflections = world.last_reflections.get(actor_idx, [])
    assert len(reflections) >= 3
    assert any("responsibility" in text for text in reflections)


def test_mindful_policy_triggers_meditate(tmp_path: Path):
    world = _build_world(tmp_path)
    world.add_cell("actor_1", properties={"label": "human", "culture": "knight"})

    actor_idx = world.id_to_idx["actor_1"]
    world.emotions[actor_idx] = "sorrow"
    world.is_meditating[actor_idx] = False

    action = world._decide_social_or_combat_action(actor_idx, world.adjacency_matrix.tocsr())
    assert action == (-1, "meditate", None)

    reflections = world.last_reflections.get(actor_idx, [])
    assert any("sorrow" in text.lower() for text in reflections)


def test_memory_policy_reflections(tmp_path: Path):
    world = _build_world(tmp_path)
    world.add_cell("actor_1", properties={"label": "human", "culture": "sage"})
    world.add_cell("kin_1", properties={"label": "human", "culture": "sage"})

    actor_idx = world.id_to_idx["actor_1"]
    kin_idx = world.id_to_idx["kin_1"]
    world.age[actor_idx] = 5
    world.hunger[actor_idx] = 60.0
    world.hunger[kin_idx] = 60.0
    world.adjacency_matrix[actor_idx, kin_idx] = 0.9
    world.adjacency_matrix[kin_idx, actor_idx] = 0.9

    world._decide_social_or_combat_action(actor_idx, world.adjacency_matrix.tocsr())
    reflections = world.last_reflections.get(actor_idx, [])
    assert any("recalls past" in text.lower() for text in reflections)


def test_imagination_policy_reflection(tmp_path: Path):
    world = _build_world(tmp_path)
    world.add_cell("actor_1", properties={"label": "human", "culture": "artist"})
    world.add_cell("kin_1", properties={"label": "human", "culture": "artist"})

    actor_idx = world.id_to_idx["actor_1"]
    kin_idx = world.id_to_idx["kin_1"]
    world.emotions[actor_idx] = "joy"
    world.satisfaction[actor_idx] = 70.0
    world.hunger[actor_idx] = 60.0
    world.hunger[kin_idx] = 60.0
    world.adjacency_matrix[actor_idx, kin_idx] = 0.5
    world.adjacency_matrix[kin_idx, actor_idx] = 0.5

    world._decide_social_or_combat_action(actor_idx, world.adjacency_matrix.tocsr())
    reflections = world.last_reflections.get(actor_idx, [])
    assert any("imagination" in text.lower() for text in reflections)


def test_resonance_policy_reflection(tmp_path: Path):
    world = _build_world(tmp_path)
    world.add_cell("actor_1", properties={"label": "human", "culture": "mages"})

    actor_idx = world.id_to_idx["actor_1"]
    world.satisfaction[actor_idx] = 85.0
    world.hunger[actor_idx] = 45.0
    world.emotions[actor_idx] = "calm"
    world._decide_social_or_combat_action(actor_idx, world.adjacency_matrix.tocsr())
    reflections = world.last_reflections.get(actor_idx, [])
    assert any("resonance" in text.lower() for text in reflections)
