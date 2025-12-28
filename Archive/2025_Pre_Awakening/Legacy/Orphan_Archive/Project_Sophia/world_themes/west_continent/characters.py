"""
Example character pool for the West Continent theme.

This is deliberately lightweight: a structured list of human characters
with roles and simple metadata that WORLD / seeding scripts can sample
from when populating CELLWORLD.

All identifiers and strings are kept ASCII-only to avoid encoding issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CharacterTemplate:
    """Minimal template for a human character in the West Continent."""

    id: str
    display_name: str
    role: str
    culture: str
    element_type: str
    notes: Dict[str, str]


_FIRST_NAMES: List[str] = [
    "Aldrin",
    "Bran",
    "Celia",
    "Darian",
    "Elaine",
    "Faris",
    "Gwen",
    "Hector",
    "Isolde",
    "Joran",
    "Kael",
    "Liora",
    "Merrin",
    "Nadia",
    "Orin",
    "Perin",
    "Rhea",
    "Seren",
    "Theron",
    "Valen",
]

_FAMILY_NAMES: List[str] = [
    "Brightwind",
    "Stormguard",
    "Dawnfield",
    "Ironvale",
    "Riversong",
    "Silvercrest",
    "Nightbloom",
    "Goldbranch",
    "Ashford",
    "Winterfall",
]

_ROLES = [
    {
        "role": "knight",
        "label": "Knight",
        "notes": {"class": "martial", "faction": "order"},
    },
    {
        "role": "squire",
        "label": "Squire",
        "notes": {"class": "martial", "faction": "order"},
    },
    {
        "role": "mage",
        "label": "Mage",
        "notes": {"class": "arcane", "faction": "guild"},
    },
    {
        "role": "cleric",
        "label": "Cleric",
        "notes": {"class": "faith", "faction": "church"},
    },
    {
        "role": "ranger",
        "label": "Ranger",
        "notes": {"class": "scout", "faction": "wild"},
    },
    {
        "role": "merchant",
        "label": "Merchant",
        "notes": {"class": "civil", "faction": "trade"},
    },
    {
        "role": "artisan",
        "label": "Artisan",
        "notes": {"class": "craft", "faction": "guild"},
    },
    {
        "role": "bandit",
        "label": "Bandit",
        "notes": {"class": "outlaw", "faction": "rogue"},
    },
    {
        "role": "scout",
        "label": "Scout",
        "notes": {"class": "scout", "faction": "order"},
    },
    {
        "role": "villager",
        "label": "Villager",
        "notes": {"class": "civil", "faction": "common"},
    },
]


def _build_pool(max_count: int = 150) -> List[CharacterTemplate]:
    """
    Build a deterministic pool of example characters by combining
    first names, family names, and roles.
    """
    pool: List[CharacterTemplate] = []
    idx = 0
    culture = "west_continent"
    element_type = "human"

    for role_spec in _ROLES:
        role = role_spec["role"]
        role_label = role_spec["label"]
        role_notes = dict(role_spec["notes"])

        for first in _FIRST_NAMES:
            for family in _FAMILY_NAMES:
                if idx >= max_count:
                    return pool
                char_id = f"{culture}.{role}.{idx:03d}"
                display_name = f"{first} {family} the {role_label}"
                notes = {
                    "full_name": display_name,
                    "role_label": role_label,
                    "culture_hint": "West Continent",
                }
                notes.update(role_notes)
                pool.append(
                    CharacterTemplate(
                        id=char_id,
                        display_name=display_name,
                        role=role,
                        culture=culture,
                        element_type=element_type,
                        notes=notes,
                    )
                )
                idx += 1

    return pool


WEST_CHARACTER_POOL: List[CharacterTemplate] = _build_pool(max_count=150)

