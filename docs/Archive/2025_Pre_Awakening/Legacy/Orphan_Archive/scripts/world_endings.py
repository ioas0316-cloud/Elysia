"""
Heuristic classification of WORLD-scale outcomes (2025-11-16).

This module lives entirely in the META layer:
- It reads macro_* snapshots on World and YearState sequences.
- It reads Character-level hero/villain scores.
- It does NOT change WORLD physics; it only returns a coarse tag +
  human-readable reason string so higher layers can compare timelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, TYPE_CHECKING

from scripts.character_model import Character, score_hero, score_villain
from scripts.macro_kingdom_model import YearState

if TYPE_CHECKING:
    from Core.FoundationLayer.Foundation.core.world import World


@dataclass
class WorldOutcome:
    """Coarse classification of how a timeline ended up."""

    kind: str
    reason: str


def _compute_macro_ranges(states: Sequence[YearState]) -> dict:
    if not states:
        return {
            "last": None,
            "max_pop": 0.0,
            "min_pop": 0.0,
            "max_war": 0.0,
            "avg_monster": 0.0,
        }

    last = states[-1]
    max_pop = max(s.population for s in states)
    min_pop = min(s.population for s in states)
    max_war = max(s.war_pressure for s in states)
    avg_monster = sum(s.monster_threat for s in states) / float(len(states))
    return {
        "last": last,
        "max_pop": max_pop,
        "min_pop": min_pop,
        "max_war": max_war,
        "avg_monster": avg_monster,
    }


def _compute_hero_villain_balance(chars: Iterable[Character]) -> float:
    total_hero = 0.0
    total_villain = 0.0
    for ch in chars:
        h = max(0.0, score_hero(ch))
        v = max(0.0, score_villain(ch))
        total_hero += h
        total_villain += v
    if total_hero <= 0.0:
        return float("inf") if total_villain > 0.0 else 1.0
    return total_villain / total_hero


def classify_world_outcome(
    world: "World",
    chars: List[Character],
    macro_states: Sequence[YearState],
) -> WorldOutcome:
    """
    Classify the overall outcome of a run.

    This is intentionally simple and easy to tune:
    - Uses only macro_* style aggregates + hero/villain balance.
    - Returns a coarse tag like "demonlord_victory" / "unified_empire"
      / "balanced_world" / "collapse" / "unknown".
    """
    metrics = _compute_macro_ranges(macro_states)
    last = metrics["last"]
    if last is None:
        return WorldOutcome(kind="unknown", reason="No macro_states provided; outcome undecided.")

    max_pop = max(metrics["max_pop"], 1.0)
    last_pop_ratio = last.population / max_pop
    last_war = float(last.war_pressure)
    last_monster = float(last.monster_threat)
    last_power = float(last.power_concentration)
    max_war = float(metrics["max_war"])
    avg_monster = float(metrics["avg_monster"])

    villain_ratio = _compute_hero_villain_balance(chars)

    # 1) Demon/monster victory: population collapses under sustained high monster threat.
    if last_pop_ratio < 0.25 and last_monster > 0.7 and avg_monster > 0.5:
        return WorldOutcome(
            kind="demonlord_victory",
            reason=(
                "Population collapsed to <25% of peak while monster threat stayed high; "
                "interpreted as Demon/monster-side victory."
            ),
        )

    # 2) Unified empire: high historical war, now low war + high power concentration.
    if (
        last_pop_ratio > 0.6
        and last_power > 0.7
        and last_war < 0.3
        and max_war > 0.5
        and last_monster < 0.6
    ):
        return WorldOutcome(
            kind="unified_empire",
            reason=(
                "Population remains high with strong power concentration and low current war; "
                "after a period of higher war pressure, interpreted as imperial unification."
            ),
        )

    # 3) General collapse: population near zero and war remains high.
    if last_pop_ratio < 0.15 and last_war > 0.6:
        return WorldOutcome(
            kind="collapse",
            reason="War and instability drove population close to zero; civilization collapse.",
        )

    # 4) Surviving but villain-heavy world.
    if last_pop_ratio > 0.4 and last_war < 0.6 and last_monster < 0.8:
        if villain_ratio > 1.5:
            return WorldOutcome(
                kind="fragile_balance",
                reason=(
                    "World survives with moderate war/monster threat, but villain power "
                    "outweighs hero power; fragile, villain-tilted balance."
                ),
            )
        return WorldOutcome(
            kind="balanced_world",
            reason=(
                "Population and fields settle into a survivable regime with no single "
                "extreme dominance; treated as a balanced, ongoing world."
            ),
        )

    # 5) Fallback.
    return WorldOutcome(
        kind="unknown",
        reason="Metrics in mid-range; no clear dominant outcome pattern.",
    )

