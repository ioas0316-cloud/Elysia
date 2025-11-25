import numpy as np
from pathlib import Path

from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.world import World
from tools.kg_manager import KGManager

from Project_Elysia.core import persistence


def _build_sample_world() -> World:
    wave = WaveMechanics(KGManager())
    world = World(
        primordial_dna={"instinct": "connect_create_meaning", "resonance_standard": "love"},
        wave_mechanics=wave,
    )
    world.time_step = 42
    world.cell_ids = ["concept:alpha"]
    world.is_alive_mask = np.array([True], dtype=bool)
    world.hp = np.array([12.5], dtype=np.float32)
    world.max_hp = np.array([15.0], dtype=np.float32)
    world.ki = np.array([1.2], dtype=np.float32)
    world.max_ki = np.array([2.0], dtype=np.float32)
    world.mana = np.array([0.5], dtype=np.float32)
    world.max_mana = np.array([1.0], dtype=np.float32)
    world.faith = np.array([0.3], dtype=np.float32)
    world.max_faith = np.array([0.6], dtype=np.float32)
    world.strength = np.array([3], dtype=np.int32)
    world.agility = np.array([4], dtype=np.int32)
    world.intelligence = np.array([5], dtype=np.int32)
    world.vitality = np.array([6], dtype=np.int32)
    world.wisdom = np.array([7], dtype=np.int32)
    world.hunger = np.array([10.0], dtype=np.float32)
    world.temperature = np.array([37.0], dtype=np.float32)
    world.satisfaction = np.array([55.0], dtype=np.float32)
    world.connection_counts = np.array([2], dtype=np.int32)
    world.element_types = np.array(["meaning"], dtype="<U20")
    world.diets = np.array(["omnivore"], dtype="<U10")
    world.growth_stages = np.array([1], dtype=np.int8)
    world.mating_readiness = np.array([0.4], dtype=np.float32)
    world.age = np.array([5], dtype=np.int32)
    world.max_age = np.array([25], dtype=np.int32)
    world.is_injured = np.array([False], dtype=bool)
    world.is_meditating = np.array([True], dtype=bool)
    world.positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    world.labels = np.array(["alpha"], dtype="<U20")
    world.insight = np.array([0.2], dtype=np.float32)
    world.emotions = np.array(["joy"], dtype="<U10")
    world.continent = np.array(["east"], dtype="<U10")
    world.culture = np.array(["wuxia"], dtype="<U10")
    world.affiliation = np.array(["Wudang"], dtype="<U20")
    return world


def test_world_save_load(tmp_path: Path):
    world = _build_sample_world()
    state_path = tmp_path / "world_state.json"
    persistence.save_world_state(world, str(state_path))

    wave = WaveMechanics(KGManager())
    restored = World(
        primordial_dna={"instinct": "connect_create_meaning", "resonance_standard": "love"},
        wave_mechanics=wave,
    )
    persistence.load_world_state(str(state_path), world=restored, wave_mechanics=wave)

    assert restored.time_step == world.time_step
    assert restored.cell_ids == world.cell_ids
    assert restored.hp.shape == world.hp.shape
    assert restored.positions.shape == world.positions.shape
    assert restored.labels[0] == world.labels[0]
