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

    from scripts.macro_kingdom_model import simulate_kingdom
    from scripts.world_macro_bridge import apply_macro_state_to_world

    kg = KGManager()
    wm = WaveMechanics(kg)
    world = World(primordial_dna={"instinct": "observe"}, wave_mechanics=wm)

    # Seed a small human cluster so threat_field has something to read from.
    def pos(x, y):
        return {"x": float(x), "y": float(y), "z": 0.0}

    for i in range(10):
        label = f"core_human_{i+1}"
        world.add_cell(
            label,
            properties={
                "label": label,
                "element_type": "animal",
                "culture": "HumanKingdom",
                "continent": "West",
                "vitality": 10,
                "strength": 8,
                "wisdom": 6,
                "position": pos(10 + i * 2, 10),
            },
        )

    # Build macro states for a shorter horizon (e.g., 50 years).
    macro_states = simulate_kingdom(years=50, initial_population=3000, carrying_capacity=50000, target_population=30000)

    print("year, macro_pop, macro_war, macro_monster, avg_threat")

    for state in macro_states:
        apply_macro_state_to_world(state, world)

        # Run a few fast ticks to let WORLD respond to the macro state.
        for _ in range(5):
            world.run_simulation_step()

        # Compute a coarse average threat level for inspection.
        if world.threat_field is not None and world.threat_field.size > 0:
            avg_threat = float(world.threat_field.mean())
        else:
            avg_threat = 0.0

        print(
            f"{state.year:4d}, "
            f"{int(state.population):7d}, "
            f"{state.war_pressure:5.2f}, "
            f"{state.monster_threat:5.2f}, "
            f"{avg_threat:8.4f}"
        )

