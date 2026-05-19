"""
WORLD <-> Character bridge (2025-11-16)

Helpers to build `Character` and `CharacterRelation` views from a running
`World` instance in Project_Sophia. This keeps character-centric logic
decoupled from the low-level simulation arrays.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from Core.FoundationLayer.Foundation.core.world import World
from scripts.character_model import (
    Character,
    CharacterRelation,
    assign_tiers,
)


def _infer_race(origin_civ: str, name: str) -> str:
    """Best-effort race inference from origin/culture/name strings."""
    s = (origin_civ or "") + " " + (name or "")
    s_lower = s.lower()
    if "elf" in s_lower:
        return "elf"
    if "dwarf" in s_lower or "dwarfh" in s_lower:
        return "dwarf"
    if "orc" in s_lower:
        return "orc"
    if "fae" in s_lower or "fairy" in s_lower:
        return "fae"
    if "dragon" in s_lower:
        return "dragon"
    return "human"


def _compute_power_score_from_world(world: World, idx: int, race: str | None = None) -> float:
    """
    Derive a coarse power_score from WORLD stats at index `idx`.

    This is intentionally simple and world-agnostic; individual worlds are
    free to replace it with a more precise mapping.
    """
    hp = float(world.hp[idx]) if world.hp.size > idx else 0.0
    max_hp = float(world.max_hp[idx]) if world.max_hp.size > idx else 0.0
    strength = float(world.strength[idx]) if world.strength.size > idx else 0.0
    agility = float(world.agility[idx]) if world.agility.size > idx else 0.0
    intelligence = float(world.intelligence[idx]) if world.intelligence.size > idx else 0.0
    vitality = float(world.vitality[idx]) if world.vitality.size > idx else 0.0
    wisdom = float(world.wisdom[idx]) if world.wisdom.size > idx else 0.0

    base = 0.3 * hp + 0.2 * max_hp

    r = (race or "human").lower()
    if r == "elf":
        stats = (
            1.2 * strength
            + 2.0 * agility
            + 1.8 * intelligence
            + 1.2 * vitality
            + 1.8 * wisdom
        )
    elif r == "dwarf":
        stats = (
            2.2 * strength
            + 1.0 * agility
            + 1.0 * intelligence
            + 2.0 * vitality
            + 1.2 * wisdom
        )
    elif r == "orc":
        stats = (
            2.5 * strength
            + 1.2 * agility
            + 0.8 * intelligence
            + 2.0 * vitality
            + 0.8 * wisdom
        )
    elif r == "fae":
        stats = (
            1.0 * strength
            + 2.2 * agility
            + 1.6 * intelligence
            + 0.8 * vitality
            + 2.0 * wisdom
        )
    else:
        # Human / default: balanced weights.
        stats = (
            2.0 * strength
            + 1.5 * agility
            + 1.2 * intelligence
            + 1.5 * vitality
            + 1.2 * wisdom
        )
    return base * 0.05 + stats


def build_characters_from_world(world: World) -> List[Character]:
    """
    Create a Character view for each living cell in the World.

    - id: world.cell_ids[i]
    - name: world.labels[i] (fallback to id)
    - origin_civ: best-effort from world.continent/culture; defaults to "World"
    - power_score: derived from WORLD stats at index i
    """
    chars: List[Character] = []

    continent_array = getattr(world, "continent", None)
    culture_array = getattr(world, "culture", None)
    affiliation_array = getattr(world, "affiliation", None)

    for idx, cid in enumerate(world.cell_ids):
        if idx >= world.is_alive_mask.size or not world.is_alive_mask[idx]:
            continue

        name = world.labels[idx] if world.labels.size > idx and world.labels[idx] else cid

        if continent_array is not None and idx < continent_array.size and continent_array[idx]:
            origin_civ = str(continent_array[idx])
        elif culture_array is not None and idx < culture_array.size and culture_array[idx]:
            origin_civ = str(culture_array[idx])
        else:
            origin_civ = "World"

        faction = None
        if affiliation_array is not None and idx < affiliation_array.size and affiliation_array[idx]:
            faction = str(affiliation_array[idx])

        race = _infer_race(origin_civ, str(name))
        power = _compute_power_score_from_world(world, idx, race=race)

        ch = Character(
            id=cid,
            name=str(name),
            origin_civ=origin_civ,
            race=race,
            faction=faction,
            era="unknown",
            birth_place_tags=[],
            class_role="unknown",
            party_role="flex",
            power_score=power,
        )
        assign_tiers(ch)
        chars.append(ch)

    return chars


def build_relations_from_world(world: World) -> List[CharacterRelation]:
    """
    Approximate CharacterRelation list from WORLD materialized cells.

    - Uses Cell.connections to create directed relations with like/trust/respect
      seeded from connection strength.
    - This is a best-effort bridge; detailed emotions are left to higher layers.
    """
    relations: List[CharacterRelation] = []
    seen: Dict[Tuple[str, str], CharacterRelation] = {}

    for cell_id, cell in world.materialized_cells.items():
        for conn in cell.connections:
            src = conn.get("source_id", cell_id)
            dst = conn.get("target_id")
            if not dst:
                continue
            strength = float(conn.get("strength", 0.5) or 0.5)
            key = (src, dst)
            if key in seen:
                rel = seen[key]
                # Merge by averaging existing and new strength-derived values.
                rel.like = (rel.like + strength) * 0.5
                rel.trust = (rel.trust + strength * 0.5) * 0.5
                rel.respect = (rel.respect + strength * 0.5) * 0.5
            else:
                rel = CharacterRelation(
                    src_id=src,
                    dst_id=dst,
                    like=strength,
                    trust=strength * 0.5,
                    respect=strength * 0.5,
                    desire=0.0,
                )
                seen[key] = rel

    relations.extend(seen.values())
    return relations
