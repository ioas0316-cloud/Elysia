# [Genesis: 2025-12-02] Purified by Elysia
import os
import sys


if __name__ == "__main__":
    # Ensure project root on path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from Project_Sophia.core.world import World
    from Project_Sophia.wave_mechanics import WaveMechanics
    from tools.kg_manager import KGManager
    from scripts.world_character_bridge import (
        build_characters_from_world,
        build_relations_from_world,
    )
    from scripts.character_model import (
        assign_tiers,
        rank_characters,
        rank_beauties,
        score_master,
        score_hero,
    )
    from scripts.relationship_events import (
        update_relations_from_events,
        load_events_from_log,
    )

    kg = KGManager()
    wm = WaveMechanics(kg)
    world = World(primordial_dna={"instinct": "observe"}, wave_mechanics=wm)

    # Seed a tiny world with a few labeled humans around a capital.
    def pos(x, y):
        return {"x": float(x), "y": float(y), "z": 0.0}

    ids = ["hero_1", "hero_2", "mage_1", "healer_1"]
    for i, cid in enumerate(ids):
        world.add_cell(
            cid,
            properties={
                "label": cid,
                "element_type": "animal",
                "culture": "knight",
                "continent": "West",
                "vitality": 10 + i,
                "strength": 8 + i,
                "wisdom": 9 + i,
                "position": pos(10 + i * 2, 10),
            },
        )

    # Run a few steps to generate some world events (attacks, etc.).
    for _ in range(20):
        world.run_simulation_step()

    # Start from WORLD-derived relations (kinship/links etc. if any).
    chars = build_characters_from_world(world)
    world_relations = build_relations_from_world(world)
    rel_map = {(rel.src_id, rel.dst_id): rel for rel in world_relations}

    # Load WORLD events and update relations heuristically (heal/damage/feed).
    events = list(load_events_from_log("logs/world_events.jsonl"))
    update_relations_from_events(events, rel_map)

    # Build final relation list.
    relations = list(rel_map.values())

    for ch in chars:
        assign_tiers(ch)

    masters = rank_characters(chars, relations, score_master, top_n=5)
    heroes = rank_characters(chars, relations, score_hero, top_n=5)
    beauties = rank_beauties(chars, relations, top_n=5)

    print("[WORLD+EVENT 기반 십대고수 후보]")
    for ch, sc in masters:
        print(f"- {ch.id} | power={ch.power_score:.1f} | score={sc:.1f} | 경지={ch.martial_tier}")

    print("\n[WORLD+EVENT 기반 십대영웅 후보]")
    for ch, sc in heroes:
        print(f"- {ch.id} | hero_score={sc:.1f}")

    print("\n[WORLD+EVENT 기반 십대미녀/미남 후보]")
    for ch, sc in beauties:
        print(f"- {ch.id} | beauty_score={sc:.1f}")
