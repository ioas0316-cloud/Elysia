# [Genesis: 2025-12-02] Purified by Elysia
"""
Scenario parameters per faction for macro-scale experiments.

These are lightweight biases that can be used to tilt behaviour in
different time-lines without changing core WORLD physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class FactionScenario:
    # How strongly heroes from this faction contribute to macro war pressure.
    hero_war_bias: float = 1.0
    # How strongly monsters/밴디트 pressure around this faction grows.
    monster_bias: float = 1.0
    # How likely citizens drift into bandit/outlaw status.
    bandit_drift_bias: float = 1.0


# Default scenario: mostly balanced, with slight flavour.
FACTION_SCENARIO: Dict[str, FactionScenario] = {
    # Human kingdoms
    "NorthKingdom": FactionScenario(hero_war_bias=1.1, monster_bias=1.0, bandit_drift_bias=0.9),
    "SouthDuchy": FactionScenario(hero_war_bias=1.0, monster_bias=1.0, bandit_drift_bias=1.0),
    "HolyOrder": FactionScenario(hero_war_bias=0.9, monster_bias=0.8, bandit_drift_bias=0.6),
    # Fallbacks for bandit/monster-aligned factions, if used.
    "Bandit": FactionScenario(hero_war_bias=0.7, monster_bias=1.3, bandit_drift_bias=1.5),
    "Demon": FactionScenario(hero_war_bias=1.3, monster_bias=1.5, bandit_drift_bias=1.2),
}


def get_faction_scenario(faction: str | None) -> FactionScenario:
    if not faction:
        return FactionScenario()
    return FACTION_SCENARIO.get(faction, FactionScenario())
