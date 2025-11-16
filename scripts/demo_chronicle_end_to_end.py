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
        assign_initial_job,
        maybe_promote_job,
        apply_job_alignment,
        update_alignment_on_kill,
        evaluate_outlaw_penalties,
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
    from scripts.jobs import (
        get_job_border_color,
        get_default_job_candidates_for_race,
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

    def pos(x: float, y: float) -> dict:
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

    years = 1000  # modest macro horizon for the demo
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

        items = sorted(hero_power_by_faction.items(), key=lambda kv: kv[1], reverse=True)
        top_faction, top_power = items[0]
        second_power = items[1][1]
        total_power = sum(p for _, p in items)
        if total_power <= 0:
            return

        dominance = max(0.0, (top_power - second_power) / total_power)
        if dominance <= 0.0:
            return

        extra_war = min(0.3, dominance * 0.5)
        base_war = float(getattr(world, "macro_war_pressure", 0.0))
        new_war = max(0.0, min(1.0, base_war + extra_war))
        world.macro_war_pressure = new_war

    for state in macro_states:
        apply_macro_state_to_world(state, world)

        yearly_chars = build_characters_from_world(world)
        _apply_faction_hero_pressure(yearly_chars)
        update_war_fronts(world, yearly_chars)
        update_faction_lifecycle(world, yearly_chars)
        run_political_events(world, yearly_chars)

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

    # Alignment / notoriety updates from KILL events.
    chars_by_id = {ch.id: ch for ch in chars}
    for ev in events:
        et = ev.get("event_type")
        if et != "KILL":
            continue
        data = ev.get("data", {}) or {}
        killer_id = data.get("killer_id")
        victim_id = data.get("victim_id")
        if not killer_id or not victim_id:
            continue
        killer = chars_by_id.get(str(killer_id))
        victim = chars_by_id.get(str(victim_id))
        if killer is None or victim is None:
            continue
        victim_element = str(data.get("victim_element", "") or "")
        # Heuristic: treat non-human element types as monsters.
        victim_is_monster = victim_element not in ("human", "citizen")
        update_alignment_on_kill(killer, victim, victim_is_monster=victim_is_monster)

    # Assign tiers and jobs (Ragnarok-style tree) for visibility.
    for ch in chars:
        assign_tiers(ch)
        if not ch.job_candidate_ids:
            ch.job_candidate_ids = get_default_job_candidates_for_race(getattr(ch, "race", "human"))
        assign_initial_job(ch)
        while maybe_promote_job(ch):
            pass
        apply_job_alignment(ch)

    # Rankings for mortal experts/heroes; exclude DemonLord (existential calamity).
    non_demon_chars = [ch for ch in chars if ch.id != "DemonLord"]

    masters = rank_characters(non_demon_chars, relations, score_master, top_n=10)
    heroes = rank_characters(non_demon_chars, relations, score_hero, top_n=10)
    beauties = rank_beauties(non_demon_chars, relations, top_n=10)

    # --- 5) Print a compact chronicle-style summary ------------------------

    print("=== ?œë? ?”ì•½ ===")
    for line in summarize_era_flags(world):
        print("-", line)

    print("\n=== ê³ ìˆ˜ ??‚¹ (WORLD+EVENT+JOB) ===")
    for ch, sc in masters:
        arc = summarize_character_arc(ch)
        job = ch.job_id or "unknown"
        border = get_job_border_color(job)
        print(f"- {ch.id:15s} | score={sc:6.1f} | job={job} [{border}] | {arc}")

    print("\n=== ?ì›… ??‚¹ (WORLD+EVENT+JOB) ===")
    for ch, sc in heroes:
        arc = summarize_character_arc(ch)
        job = ch.job_id or "unknown"
        border = get_job_border_color(job)
        print(f"- {ch.id:15s} | hero={sc:6.1f} | job={job} [{border}] | {arc}")

    print("\n=== ë¯¸ì¸/ë¯¸ë‚¨ ??‚¹ (WORLD+EVENT+JOB) ===")
    for ch, sc in beauties:
        arc = summarize_character_arc(ch)
        job = ch.job_id or "unknown"
        border = get_job_border_color(job)
        print(f"- {ch.id:15s} | beauty={sc:6.1f} | job={job} [{border}] | {arc}")
