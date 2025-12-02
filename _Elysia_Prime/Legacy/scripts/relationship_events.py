# [Genesis: 2025-12-02] Purified by Elysia
"""
Relationship update helpers from WORLD event logs (2025-11-16)

This module reads `logs/world_events.jsonl` style logs and updates
CharacterRelation objects in a soft, heuristic way. It is intentionally
lightweight and optional; higher layers can choose how strongly to apply
these updates.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, Tuple

from scripts.character_model import CharacterRelation


def _get_rel_key(src: str, dst: str) -> Tuple[str, str]:
    return src, dst


def update_relations_from_events(
    events: Iterable[dict],
    relations: Dict[Tuple[str, str], CharacterRelation],
) -> None:
    """
    Mutate `relations` in place based on a stream of WORLD events.

    The mapping is heuristic:
    - SPELL with heal: target -> caster like/trust/respect up.
    - SPELL with damage: target -> caster grudge up, like/trust down.
    - DRINK/EAT that reference an explicit provider can give mild like up.
    """

    for ev in events:
        etype = ev.get("event_type")
        data = ev.get("data", {}) or {}

        if etype == "SPELL":
            caster = data.get("caster_id")
            target = data.get("target_id")
            spell = str(data.get("spell", "")).lower()
            if not caster or not target:
                continue
            if "heal" in spell:
                # Healing: target appreciates and trusts caster more.
                key = _get_rel_key(target, caster)
                rel = relations.setdefault(key, CharacterRelation(src_id=target, dst_id=caster))
                rel.like += 0.05
                rel.trust += 0.05
                rel.respect += 0.03
            else:
                # Offensive spell: target grows grudge; trust/like drop.
                key = _get_rel_key(target, caster)
                rel = relations.setdefault(key, CharacterRelation(src_id=target, dst_id=caster))
                rel.grudge += 0.07
                rel.like -= 0.03
                rel.trust -= 0.04

        elif etype in ("EAT", "DRINK"):
            # If someone feeds/provides drink, the recipient may like them more.
            actor = data.get("actor_id") or data.get("cell_id")
            provider = data.get("provider_id")
            if actor and provider and provider != actor:
                key = _get_rel_key(actor, provider)
                rel = relations.setdefault(key, CharacterRelation(src_id=actor, dst_id=provider))
                rel.like += 0.02
                rel.trust += 0.01


def load_events_from_log(path: str) -> Iterable[dict]:
    """
    Read events from a JSONL log file.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
