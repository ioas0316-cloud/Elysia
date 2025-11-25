"""
Named hero pool and preferences for Elysia chronicles.

This module defines a small set of hand-authored heroes so that
chronicle-style demos do not rely only on technical IDs like
`citizen_1` or `bandit_3`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from scripts.character_model import Character


@dataclass
class HeroSpec:
    id: str
    name: Optional[str] = None
    epithet: Optional[str] = None
    job_candidate_ids: Optional[List[str]] = None
    origin_civ: Optional[str] = None
    faction: Optional[str] = None


HERO_SPECS: Dict[str, HeroSpec] = {
    # Knight / frontline heroes
    "citizen_1": HeroSpec(
        id="citizen_1",
        name="롤랑",
        epithet="백야의 기사",
        job_candidate_ids=[
            "martial.soldier.swordsman",
        ],
        origin_civ="NorthKingdom",
        faction="NorthKingdom",
    ),
    "citizen_2": HeroSpec(
        id="citizen_2",
        name="시그르드",
        epithet="창벽의 수호자",
        job_candidate_ids=[
            "martial.soldier.swordsman",
        ],
        origin_civ="SouthDuchy",
        faction="SouthDuchy",
    ),
    # Mage / scholar heroes
    "citizen_3": HeroSpec(
        id="citizen_3",
        name="엘리안",
        epithet="푸른 불꽃의 마도사",
        job_candidate_ids=[
            "knowledge.scholar.mage",
        ],
        origin_civ="NorthKingdom",
        faction="ArcaneTower",
    ),
    "citizen_4": HeroSpec(
        id="citizen_4",
        name="루미에르",
        epithet="별빛 연금술사",
        job_candidate_ids=[
            "knowledge.scholar.mage",
            "trade.merchant.merchant",
        ],
        origin_civ="SouthDuchy",
        faction="ArcaneTower",
    ),
    # Archer / ranger heroes
    "citizen_5": HeroSpec(
        id="citizen_5",
        name="에린",
        epithet="바람길 추적자",
        job_candidate_ids=[
            "adventure.adventurer.archer",
        ],
        origin_civ="NorthKingdom",
        faction="NorthKingdom",
    ),
    # Faith / priest heroes
    "citizen_6": HeroSpec(
        id="citizen_6",
        name="미카엘라",
        epithet="새벽의 십자가",
        job_candidate_ids=[
            "faith.priest.acolyte_combat",
        ],
        origin_civ="HolyOrder",
        faction="HolyOrder",
    ),
    "citizen_7": HeroSpec(
        id="citizen_7",
        name="요한",
        epithet="가난한 이들의 목자",
        job_candidate_ids=[
            "faith.priest.acolyte_combat",
        ],
        origin_civ="HolyOrder",
        faction="HolyOrder",
    ),
    # Merchant / support heroes
    "citizen_8": HeroSpec(
        id="citizen_8",
        name="세레나",
        epithet="황금마차의 여주인",
        job_candidate_ids=[
            "trade.merchant.merchant",
        ],
        origin_civ="SouthDuchy",
        faction="SouthDuchy",
    ),
}


def apply_hero_specs(chars: List[Character]) -> None:
    """
    Mutate Characters in-place using HERO_SPECS.

    - Sets nicer name/epithet when provided.
    - Seeds job_candidate_ids when provided.
    - Optionally refines origin_civ/faction.
    """
    spec_map = HERO_SPECS
    for ch in chars:
        spec = spec_map.get(ch.id)
        if not spec:
            continue

        if spec.name:
            ch.name = spec.name
        if spec.epithet:
            ch.epithet = spec.epithet
        if spec.origin_civ:
            ch.origin_civ = spec.origin_civ
        if spec.faction:
            ch.faction = spec.faction
        if spec.job_candidate_ids:
            ch.job_candidate_ids = list(spec.job_candidate_ids)

