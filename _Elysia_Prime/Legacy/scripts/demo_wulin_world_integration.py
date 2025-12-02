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

    from scripts.macro_kingdom_model import simulate_wulin
    from scripts.world_macro_bridge import apply_macro_state_to_world

    kg = KGManager()
    wm = WaveMechanics(kg)
    world = World(primordial_dna={"instinct": "observe"}, wave_mechanics=wm)

    # Enable macro-era/disaster events so we can see Wulin eras and omens.
    world.enable_macro_disaster_events = True

    # Seed a small Wulin-flavoured cluster.
    def pos(x, y):
        return {"x": float(x), "y": float(y), "z": 0.0}

    sects = ["Wudang", "Shaolin", "Huashan", "Emei", "Kunlun", "BeggarSect"]
    for i, sect in enumerate(sects):
        label = f"wuxia_{i+1}"
        world.add_cell(
            label,
            properties={
                "label": label,
                "element_type": "animal",
                "culture": "wuxia",
                "continent": "East",
                "affiliation": sect,
                "vitality": 10 + i,
                "strength": 8 + i // 2,
                "wisdom": 7 + i // 2,
                "position": pos(20 + i * 2, 20),
            },
        )

    # Build macro states for Wulin over a shorter horizon (e.g., 80 years).
    macro_states = simulate_wulin(years=80)

    print("year, pop, jianghu(war), unrest, adventure, famine, bounty, war_state")

    for state in macro_states:
        apply_macro_state_to_world(state, world)

        # Run a few fast ticks to let WORLD respond.
        for _ in range(5):
            world.run_simulation_step()

        war_state = getattr(world, "_macro_war_state", "peace")
        famine = bool(getattr(world, "_macro_famine_active", False))
        bounty = bool(getattr(world, "_macro_bounty_active", False))

        print(
            f"{state.year:4d}, "
            f"{int(state.population):6d}, "
            f"{state.war_pressure:5.2f}, "
            f"{state.unrest:5.2f}, "
            f"{state.adventure_pressure:5.2f}, "
            f"{int(famine):1d}, "
            f"{int(bounty):1d}, "
            f"{war_state}"
        )
