"""
World–macro bridge (2025-11-16)

Purpose
- Provide a single place where macro-scale kingdom state (war, adventure,
  monster threat, wealth, tech, unrest...) is attached to a running World
  instance, so higher-level loops (OS/Chronos) can call this once per
  macro_tick/year and let the World read those fields when updating.

Design
- This module does not change World behaviour by itself; it only writes
  macro_* attributes onto the World. The existing WORLD code can then
  sample these attributes when updating fields, spawning events, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.macro_kingdom_model import YearState

if TYPE_CHECKING:  # Avoid circular imports at runtime
    from Project_Sophia.core.world import World


def apply_macro_state_to_world(state: YearState, world: "World") -> None:
    """
    Attach macro-scale kingdom state to a World instance.

    This does not directly mutate fields like threat_field; instead it
    stores macro_* attributes that WORLD update functions can read and
    incorporate as gentle, law-like influences (e.g., scaling threat
    updates by macro_war_pressure).
    """

    # Core macro pressures
    world.macro_war_pressure = float(state.war_pressure)
    world.macro_adventure_pressure = float(state.adventure_pressure)
    world.macro_monster_threat = float(state.monster_threat)

    # Social / economic macro state
    world.macro_unrest = float(state.unrest)
    world.macro_tech_level = float(state.tech_level)
    world.macro_wealth = float(state.wealth)
    world.macro_trade_index = float(state.trade_index)
    world.macro_power_concentration = float(state.power_concentration)
    world.macro_literacy = float(state.literacy)
    world.macro_culture_index = float(state.culture_index)

    # Food / surplus snapshot (useful for tuning hunger/production)
    world.macro_population = float(state.population)
    world.macro_food_stock = float(state.food_stock)
    world.macro_surplus_years = float(state.surplus_years)
