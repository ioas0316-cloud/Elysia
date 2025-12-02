# [Genesis: 2025-12-02] Purified by Elysia
"""
Export a compact WORLD snapshot as JSON for Godot rendering.

Design
- This script is META-only: it builds or advances a World instance,
  derives Character views, and writes a read-only snapshot file.
- Godot should treat the JSON as a lens: visualize, but never push
  changes back into WORLD physics.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List


def _ensure_repo_root_on_path() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def _build_world(years: int = 300, ticks_per_year: int = 3):
    """
    Build a small sample world + run macro sim.

    This follows the same rough pattern as demo_chronicle_end_to_end but is
    kept separate so Godot can call it without depending on that script.
    """
    _ensure_repo_root_on_path()

    from Project_Sophia.core.world import World
    from Project_Sophia.wave_mechanics import WaveMechanics
    from tools.kg_manager import KGManager

    from scripts.macro_kingdom_model import simulate_kingdom
    from scripts.world_macro_bridge import apply_macro_state_to_world
    from scripts.world_character_bridge import build_characters_from_world
    from scripts.character_model import (
        assign_tiers,
        assign_initial_job,
        maybe_promote_job,
        apply_job_alignment,
    )
    from scripts.jobs import get_default_job_candidates_for_race

    kg = KGManager()
    wm = WaveMechanics(kg)
    world = World(primordial_dna={"instinct": "observe"}, wave_mechanics=wm)

    # Enable macro-era/disaster for flavor; keep food model on so characters survive.
    world.enable_macro_disaster_events = True
    world.macro_food_model_enabled = True

    def pos(x: float, y: float) -> Dict[str, float]:
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

    macro_states = simulate_kingdom(
        years=years,
        initial_population=1000,
        carrying_capacity=50000,
        target_population=30000,
    )

    for state in macro_states:
        apply_macro_state_to_world(state, world)
        for _ in range(ticks_per_year):
            world.run_simulation_step()

    chars = build_characters_from_world(world)
    for ch in chars:
        assign_tiers(ch)
        if not ch.job_candidate_ids:
            ch.job_candidate_ids = get_default_job_candidates_for_race(getattr(ch, "race", "human"))
        assign_initial_job(ch)
        while maybe_promote_job(ch):
            pass
        apply_job_alignment(ch)

    return world, chars, macro_states


def _build_snapshot(world, chars, macro_states) -> Dict[str, Any]:
    from scripts.jobs import get_job_border_color
    from scripts.character_model import evaluate_outlaw_penalties

    # Optional overlay: WORLD symbol usage (signs attached to characters).
    repo_root = _ensure_repo_root_on_path()
    logs_dir = os.path.join(repo_root, "logs")
    symbol_usage_path = os.path.join(logs_dir, "world_symbol_usage.json")
    sign_by_owner: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(symbol_usage_path):
        try:
            with open(symbol_usage_path, "r", encoding="utf-8") as f:
                usage = json.load(f)
            for sign in usage.get("signs", []):
                owner_id = sign.get("owner_id")
                if not owner_id:
                    continue
                sign_by_owner[owner_id] = {
                    "sign_text_ko": sign.get("text_ko"),
                    "sign_text_en": sign.get("text_en"),
                }
        except Exception:
            sign_by_owner = {}

    # Optional overlay: WORLD text objects (books/diaries/etc.).
    text_objects_path = os.path.join(logs_dir, "world_text_objects.json")
    text_objects: List[Dict[str, Any]] = []
    if os.path.exists(text_objects_path):
        try:
            with open(text_objects_path, "r", encoding="utf-8") as f:
                text_usage = json.load(f)
            text_objects = list(text_usage.get("texts", []))
        except Exception:
            text_objects = []

    # Macro snapshot (last macro_state)
    last_state = macro_states[-1] if macro_states else None

    macro = {}
    if last_state is not None:
        macro = {
            "year": int(last_state.year),
            "population": float(last_state.population),
            "war_pressure": float(last_state.war_pressure),
            "monster_threat": float(last_state.monster_threat),
            "power_concentration": float(last_state.power_concentration),
            "unrest": float(last_state.unrest),
            "adventure_pressure": float(last_state.adventure_pressure),
        }

    macro_fields = {
        "war_pressure": float(getattr(world, "macro_war_pressure", 0.0)),
        "monster_threat": float(getattr(world, "macro_monster_threat", 0.0)),
        "unrest": float(getattr(world, "macro_unrest", 0.0)),
        "power_concentration": float(getattr(world, "macro_power_concentration", 0.0)),
        "population": float(getattr(world, "macro_population", 0.0)),
    }

    characters: List[Dict[str, Any]] = []
    for ch in chars:
        cid = ch.id
        idx = world.id_to_idx.get(cid)
        position = None
        try:
            if idx is not None and world.position is not None and idx < world.position.shape[0]:
                pos_vec = world.position[idx]
                position = {
                    "x": float(pos_vec[0]),
                    "y": float(pos_vec[1]),
                    "z": float(pos_vec[2]) if len(pos_vec) > 2 else 0.0,
                }
        except Exception:
            position = None

        job_id = ch.job_id or ""
        border = get_job_border_color(job_id)
        outlaw = evaluate_outlaw_penalties(ch)
        sign_info = sign_by_owner.get(cid, {})

        # Simple alignment tag for Godot-side colour-coding.
        law = float(getattr(ch, "alignment_law", 0.0))
        good = float(getattr(ch, "alignment_good", 0.0))
        if law >= 0.3 and good >= 0.3:
            alignment_tag = "lawful_good"
        elif law <= -0.3 and good >= 0.3:
            alignment_tag = "chaotic_good"
        elif abs(law) < 0.3 and abs(good) < 0.3:
            alignment_tag = "neutral"
        elif law <= -0.3 and good <= -0.3:
            alignment_tag = "chaotic_evil"
        elif law >= 0.3 and good <= -0.3:
            alignment_tag = "lawful_evil"
        elif good >= 0.3:
            alignment_tag = "good"
        elif good <= -0.3:
            alignment_tag = "evil"
        else:
            alignment_tag = "unknown"

        characters.append(
            {
                "id": cid,
                "name": ch.name,
                "race": getattr(ch, "race", "human"),
                "faction": ch.faction,
                "power_score": float(ch.power_score),
                "martial_tier": ch.martial_tier,
                "adventurer_rank": ch.adventurer_rank,
                "alignment_law": law,
                "alignment_good": good,
                "alignment_tag": alignment_tag,
                "notoriety": float(getattr(ch, "notoriety", 0.0)),
                "job_id": job_id,
                "job_border": border,
                "outlaw": outlaw,
                "sign_text_ko": sign_info.get("sign_text_ko"),
                "sign_text_en": sign_info.get("sign_text_en"),
                "position": position,
            }
        )

    snapshot: Dict[str, Any] = {
        "meta": {
            "time_step": int(getattr(world, "time_step", 0)),
            "macro_state": macro,
            "macro_fields": macro_fields,
        },
        "characters": characters,
        "text_objects": text_objects,
    }
    return snapshot


def main() -> None:
    years = 300
    ticks_per_year = 3
    world, chars, macro_states = _build_world(years=years, ticks_per_year=ticks_per_year)
    snapshot = _build_snapshot(world, chars, macro_states)

    repo_root = _ensure_repo_root_on_path()
    out_dir = os.path.join(repo_root, "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "world_snapshot_for_godot.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"Wrote Godot snapshot to: {out_path}")


if __name__ == "__main__":
    main()