# [Genesis: 2025-12-02] Purified by Elysia
"""
Configuration scaffold for the East Continent (무림 / 동양 판타지) theme.

The goal here is structure first:
- keep all east‑side constants and IDs in one place,
- let World / Guardian choose a theme without hunting through scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


THEME_ID = "east_continent"


@dataclass(frozen=True)
class EastThemeConfig:
    theme_id: str
    label: str
    description: str
    preferred_job_ids: List[str]
    favored_spell_keys: List[str]
    notes: Dict[str, str]


EAST_THEME = EastThemeConfig(
    theme_id=THEME_ID,
    label="동대륙 (East Continent)",
    description="산맥 동쪽, 문파/강호/무도 중심의 무림 세계를 위한 기본 테마.",
    # These are job ids as strings; actual Job definitions live in scripts/jobs.py.
    preferred_job_ids=[
        "martial.soldier.swordsman",
        "martial.soldier.guard",
        "adventure.adventurer.archer",
        "adventure.adventurer.hunter",
        "faith.priest.acolyte_combat",
    ],
    # Spell keys are defined in Project_Sophia/core/spells.py.
    favored_spell_keys=[
        "firebolt",
        "heal",
    ],
    notes={
        "terrain_hint": "산맥 동쪽, 계곡과 강, 안개 낀 숲과 골짜기.",
        "culture_hint": "문파 간의 의리, 원한, 검술/내공 수련이 중심.",
    },
)
