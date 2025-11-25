"""
World theme registry for the single CellWorld body.

The physical world lives in Project_Sophia/core/world.py.
Themes describe how that one world is “skinned” or configured:
- east_continent: 무림 / 동양 판타지 축
- west_continent: 서양 판타지 / 기사단 / 마법 축

Runtime engines can look up a theme here and then apply the
corresponding presets (jobs, spells, culture fields, event weights).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class WorldTheme:
    theme_id: str
    label: str
    description: str
    config_module: str


THEME_REGISTRY: Dict[str, WorldTheme] = {}


def register_theme(theme: WorldTheme) -> None:
    THEME_REGISTRY[theme.theme_id] = theme


def get_theme(theme_id: str) -> WorldTheme | None:
    return THEME_REGISTRY.get(theme_id)


# Register the two main continent themes up‑front.
register_theme(
    WorldTheme(
        theme_id="east_continent",
        label="동대륙 (East Continent)",
        description="무림, 기공, 문파, 기개/의리 중심의 동양 판타지 축.",
        config_module="Project_Sophia.world_themes.east_continent.config",
    )
)

register_theme(
    WorldTheme(
        theme_id="west_continent",
        label="서대륙 (West Continent)",
        description="기사단, 마법 길드, 성당과 왕국 중심의 서양 판타지 축.",
        config_module="Project_Sophia.world_themes.west_continent.config",
    )
)

