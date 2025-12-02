# [Genesis: 2025-12-02] Purified by Elysia
"""
Configuration scaffold for the West Continent (서양 판타지) theme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


THEME_ID = "west_continent"


@dataclass(frozen=True)
class WestThemeConfig:
    theme_id: str
    label: str
    description: str
    preferred_job_ids: List[str]
    favored_spell_keys: List[str]
    notes: Dict[str, str]


WEST_THEME = WestThemeConfig(
    theme_id=THEME_ID,
    label="서대륙 (West Continent)",
    description="산맥 서쪽, 기사단/마법 길드/왕국 중심의 서양 판타지 테마.",
    preferred_job_ids=[
        "martial.soldier.knight",
        "knowledge.scholar.mage",
        "faith.priest.priest",
        "trade.merchant.merchant",
        "craft.artisan.blacksmith",
    ],
    favored_spell_keys=[
        "firebolt",
        "heal",
    ],
    notes={
        "terrain_hint": "산맥 서쪽, 평원과 숲, 성채와 도시, 농지.",
        "culture_hint": "기사단의 서약, 왕국의 법, 마법 길드와 성당의 힘.",
    },
)
