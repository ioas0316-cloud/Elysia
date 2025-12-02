# [Genesis: 2025-12-02] Purified by Elysia
"""
Initialize core value nodes in the KG if they don't exist yet.

Usage:
  python -m scripts.init_core_values
"""
from __future__ import annotations

from tools.kg_manager import KGManager


CORE_VALUES = [
    "value:love",
    "value:clarity",
    "value:creativity",
    "value:verifiability",
    "value:relatedness",
]


def main():
    kg = KGManager()
    created = []
    for vid in CORE_VALUES:
        if not kg.get_node(vid):
            kg.add_node(vid, properties={"type": "value", "mass": 0.0})
            created.append(vid)
    if created:
        kg.save()
    print("Core values present. Created:", created)


if __name__ == "__main__":
    main()
