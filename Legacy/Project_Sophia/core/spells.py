from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


class World:  # forward decl for type hints
    pass


@dataclass
class Spell:
    name: str
    description: str
    cost_type: str  # 'mana' or 'faith'
    cost: float
    target: str  # 'self' or 'target'

    def can_cast(self, world: World, actor_idx: int) -> bool:
        if self.cost_type == 'mana':
            return getattr(world, 'mana')[actor_idx] >= self.cost
        if self.cost_type == 'faith':
            return getattr(world, 'faith')[actor_idx] >= self.cost
        return False

    def spend_cost(self, world: World, actor_idx: int) -> None:
        if self.cost_type == 'mana':
            world.mana[actor_idx] = max(0, world.mana[actor_idx] - self.cost)
        elif self.cost_type == 'faith':
            world.faith[actor_idx] = max(0, world.faith[actor_idx] - self.cost)


def _apply_firebolt(world: World, actor_idx: int, target_idx: int) -> Dict[str, float]:
    # Intelligence scales spell strength
    base = 10.0 + float(world.intelligence[actor_idx]) * 0.5
    damage = max(0.0, base)
    world.hp[target_idx] -= damage
    world.is_injured[target_idx] = True
    return {"damage": damage}


def _apply_heal(world: World, actor_idx: int) -> Dict[str, float]:
    base = 8.0 + float(world.wisdom[actor_idx]) * 0.4
    healed = float(min(world.max_hp[actor_idx] - world.hp[actor_idx], base))
    if healed > 0:
        world.hp[actor_idx] += healed
    # small chance to clear injury
    from random import random
    if world.is_injured[actor_idx] and random() < 0.3:
        world.is_injured[actor_idx] = False
    return {"heal": healed}


# Simple spell registry
SPELL_BOOK: Dict[str, Spell] = {
    "firebolt": Spell(
        name="Firebolt",
        description="Launch a bolt of fire that harms a single target.",
        cost_type='mana',
        cost=10.0,
        target='target',
    ),
    "heal": Spell(
        name="Heal",
        description="Restore some of the caster's health.",
        cost_type='mana',
        cost=8.0,
        target='self',
    ),
}

def cast_spell(world: World, spell_key: str, actor_idx: int, target_idx: Optional[int] = None) -> Dict[str, float]:
    spell = SPELL_BOOK.get(spell_key)
    if spell is None:
        return {}
    if not spell.can_cast(world, actor_idx):
        return {}

    # Spend resource
    spell.spend_cost(world, actor_idx)

    if spell_key == 'firebolt' and target_idx is not None:
        return _apply_firebolt(world, actor_idx, target_idx)
    if spell_key == 'heal':
        return _apply_heal(world, actor_idx)
    return {}

