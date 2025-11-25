"""
Helpers to place WORLD_KITS into a running World (2025-11-16).

These loaders are intentionally lightweight: they spawn monsters as cells
near a given origin point and leave finer dungeon geometry to future
WORLD_KIT/OS-level work.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from scripts.world_kits import WORLD_KITS

if TYPE_CHECKING:
    from Project_Sophia.core.world import World


def place_world_kit(world: "World", kit_id: str, origin_x: float, origin_y: float, radius: float = 5.0) -> None:
    """
    Spawn basic monsters from a WORLD_KIT around (origin_x, origin_y).

    - For each monster entry, a random count in count_range is spawned.
    - Monsters are simple animal cells with a generic 'monster' culture.
    """
    kit = WORLD_KITS.get(kit_id)
    if not kit:
        return

    floors = kit.get("floors", [])
    for floor in floors:
        for monster in floor.get("monsters", []):
            monster_id = str(monster.get("id", "monster"))
            threat = str(monster.get("threat", "low")).lower()
            count_min, count_max = monster.get("count_range", [1, 1])
            try:
                count = random.randint(int(count_min), int(count_max))
            except Exception:
                count = 1

            # Coarse stats based on threat level.
            if threat == "low":
                strength, vitality = 5, 10
            elif threat == "medium":
                strength, vitality = 12, 20
            else:
                strength, vitality = 18, 30

            for i in range(count):
                dx = random.uniform(-radius, radius)
                dy = random.uniform(-radius, radius)
                label = f"{monster_id}_{i+1}"
                world.add_cell(
                    label,
                    properties={
                        "label": label,
                        "element_type": "animal",
                        "culture": "monster",
                        "continent": "SpineMountains",
                        "strength": strength,
                        "vitality": vitality,
                        "position": {"x": float(origin_x + dx), "y": float(origin_y + dy), "z": 0.0},
                    },
                )

