import os
import sys


if __name__ == "__main__":
    # Ensure project root on path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from Core.FoundationLayer.Foundation.core.world import World
    from Core.FoundationLayer.Foundation.wave_mechanics import WaveMechanics
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

    kg = KGManager()
    wm = WaveMechanics(kg)
    world = World(primordial_dna={"instinct": "observe"}, wave_mechanics=wm)

    # Seed a tiny world with a few labeled humans around a notional capital.
    def pos(x, y):
        return {"x": float(x), "y": float(y), "z": 0.0}

    for i in range(6):
        label = f"human_{i+1}"
        world.add_cell(
            label,
            properties={
                "label": label,
                "element_type": "animal",
                "culture": "HumanKingdom",
                "continent": "West",
                "vitality": 8 + i,
                "strength": 7 + i // 2,
                "wisdom": 5 + i // 2,
                "position": pos(10 + i * 2, 10),
            },
        )

    # Run a few steps to let the world settle a bit.
    for _ in range(10):
        world.run_simulation_step()

    chars = build_characters_from_world(world)
    rels = build_relations_from_world(world)

    # Make sure tiers are filled (bridge already assigns, but be explicit).
    for ch in chars:
        assign_tiers(ch)

    masters = rank_characters(chars, rels, score_master, top_n=3)
    heroes = rank_characters(chars, rels, score_hero, top_n=3)
    beauties = rank_beauties(chars, rels, top_n=3)

    print("[WORLD 기반 십대고수 후보]")
    for ch, sc in masters:
        print(f"- {ch.id} | power={ch.power_score:.1f} | score={sc:.1f} | 경지={ch.martial_tier}")

    print("\n[WORLD 기반 십대영웅 후보]")
    for ch, sc in heroes:
        print(f"- {ch.id} | hero_score={sc:.1f}")

    print("\n[WORLD 기반 십대미녀/미남 후보]")
    for ch, sc in beauties:
        print(f"- {ch.id} | beauty_score={sc:.1f}")

