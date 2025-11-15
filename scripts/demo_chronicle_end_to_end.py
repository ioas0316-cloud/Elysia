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

    from scripts.macro_kingdom_model import simulate_kingdom
    from scripts.world_macro_bridge import apply_macro_state_to_world
    from scripts.world_character_bridge import (
        build_characters_from_world,
        build_relations_from_world,
    )
    from scripts.relationship_events import (
        load_events_from_log,
        update_relations_from_events,
    )
    from scripts.character_model import (
        assign_tiers,
        rank_characters,
        rank_beauties,
        score_master,
        score_hero,
    )
    from scripts.narrative_summaries import (
        summarize_character_arc,
        summarize_era_flags,
    )
    from scripts.political_events import (
        run_political_events,
        update_war_fronts,
        update_faction_lifecycle,
    )

    kg = KGManager()
    wm = WaveMechanics(kg)
    world = World(primordial_dna={"instinct": "observe"}, wave_mechanics=wm)

    # Enable macro-era/disaster + demon/angel omens so the chronicle has flavor.
    world.enable_macro_disaster_events = True
    # Keep the population broadly nourished so we still have living characters
    # at the end of the macro horizon.
    world.macro_food_model_enabled = True

    # --- 1) Seed a small human kingdom near a capital -----------------------

    def pos(x, y):
        return {"x": float(x), "y": float(y), "z": 0.0}

    factions = ["NorthKingdom", "SouthDuchy", "HolyOrder"]
    for i in range(20):
        cid = f"citizen_{i+1}"
        faction = factions[i % len(factions)]
        world.add_cell(
            cid,
            properties={
                "label": cid,
                "element_type": "animal",
                "culture": "knight",
                "continent": "West",
                "affiliation": faction,
                "vitality": 8 + (i % 4),
                "strength": 7 + (i % 5),
                "wisdom": 6 + (i % 3),
                "position": pos(30 + (i % 5) * 2, 30 + (i // 5) * 2),
            },
        )

    # --- 2) Run macro kingdom sim + WORLD for a few centuries --------------

    years = 120  # modest macro horizon for the demo
    macro_states = simulate_kingdom(
        years=years,
        initial_population=1000,
        carrying_capacity=50000,
        target_population=30000,
    )

    # Fewer fast ticks per macro year (law is unchanged; only sampling rate).
    ticks_per_year = 3

    def _apply_faction_hero_pressure(chars_local) -> None:
        """
        Softly increase macro_war_pressure based on faction hero imbalance.

        - Build Characters from WORLD and score heroes.
        - Aggregate hero scores per faction.
        - If one faction dominates, gently nudge macro_war_pressure upward
          proportional to that imbalance, modelling other factions feeling
          threatened by a hero-heavy sect/kingdom.
        """
        if not chars_local:
            return

        # Aggregate hero scores by faction.
        hero_power_by_faction = {}
        for ch in chars_local:
            if not ch.faction:
                continue
            score = score_hero(ch)
            if score <= 0:
                continue
            hero_power_by_faction[ch.faction] = hero_power_by_faction.get(ch.faction, 0.0) + score

        if len(hero_power_by_faction) < 2:
            return

        # Compute dominance of the strongest faction vs the rest.
        items = sorted(hero_power_by_faction.items(), key=lambda kv: kv[1], reverse=True)
        top_faction, top_power = items[0]
        second_power = items[1][1]
        total_power = sum(p for _, p in items)
        if total_power <= 0:
            return

        dominance = max(0.0, (top_power - second_power) / total_power)
        if dominance <= 0.0:
            return

        # Convert dominance into an extra war pressure term (0..~0.3).
        extra_war = min(0.3, dominance * 0.5)
        base_war = float(getattr(world, "macro_war_pressure", 0.0))
        new_war = max(0.0, min(1.0, base_war + extra_war))
        world.macro_war_pressure = new_war

    for state in macro_states:
        apply_macro_state_to_world(state, world)

        # Build a character snapshot for this macro step.
        yearly_chars = build_characters_from_world(world)
        # Hero-heavy factions softly raise war pressure before the year plays out.
        _apply_faction_hero_pressure(yearly_chars)
        # Maintain multi-front war/diplomacy state between factions.
        update_war_fronts(world, yearly_chars)
        # Update long-horizon faction lifecycle (rise/fall/rebirth).
        update_faction_lifecycle(world, yearly_chars)
        # Run a small political phase: 암살/선동/연애 동맹 등.
        run_political_events(world, yearly_chars)

        # Let WORLD evolve a bit within this macro year.
        for _ in range(ticks_per_year):
            world.run_simulation_step()

    # --- 3) Build Characters + Relations from WORLD ------------------------

    chars = build_characters_from_world(world)
    base_relations = build_relations_from_world(world)
    rel_map = {(rel.src_id, rel.dst_id): rel for rel in base_relations}

    # --- 4) Fold in WORLD event log to update relations --------------------

    events = list(load_events_from_log("logs/world_events.jsonl"))
    update_relations_from_events(events, rel_map)
    relations = list(rel_map.values())

    for ch in chars:
        assign_tiers(ch)

    masters = rank_characters(chars, relations, score_master, top_n=10)
    heroes = rank_characters(chars, relations, score_hero, top_n=10)
    beauties = rank_beauties(chars, relations, top_n=10)

    # --- 5) Print a compact chronicle-style summary ------------------------

    print("=== 시대 요약 ===")
    for line in summarize_era_flags(world):
        print("-", line)

    print("\n=== 십대고수 후보 (WORLD+EVENT) ===")
    for ch, sc in masters:
        arc = summarize_character_arc(ch)
        print(f"- {ch.id:15s} | score={sc:6.1f} | {arc}")

    print("\n=== 십대영웅 후보 (WORLD+EVENT) ===")
    for ch, sc in heroes:
        arc = summarize_character_arc(ch)
        print(f"- {ch.id:15s} | hero={sc:6.1f} | {arc}")

    print("\n=== 십대미녀/미남 후보 (WORLD+EVENT) ===")
    for ch, sc in beauties:
        arc = summarize_character_arc(ch)
        print(f"- {ch.id:15s} | beauty={sc:6.1f} | {arc}")
