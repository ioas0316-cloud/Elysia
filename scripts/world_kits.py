"""
World kits (2025-11-16)

Lightweight, code-side descriptions of notable world locations. These are
not canonical protocols; they are helpers for simulations and tools.
"""

DWARF_DEEP_HOLD_01 = {
    "id": "DWARF_DEEP_HOLD_01",
    "name": "드워프 심층 요새 1호",
    "location": "SpineMountains",
    "floors": [
        {
            "id": "F1",
            "name": "상층 폐갱 지구",
            "depth": 1,
            "description": "한때 번성했던 광맥과 공방이 지금은 침묵한 채 남아 있는 구역.",
            "rooms": [
                {"id": "entrance", "type": "gate", "tags": ["safe", "checkpoint"]},
                {"id": "old_mine_1", "type": "mine_tunnel", "tags": ["collapsed", "ore_rich"]},
                {"id": "workshop_row", "type": "workshop", "tags": ["forges", "scrap", "traps_possible"]},
                {"id": "storage", "type": "storage", "tags": ["supplies", "vermin"]},
            ],
            "monsters": [
                {"id": "cave_rat", "threat": "low", "count_range": [3, 8]},
                {"id": "stone_spider", "threat": "low", "count_range": [1, 4]},
            ],
            "hazards": [
                {"id": "unstable_ceiling", "type": "collapse", "severity": "medium"},
                {"id": "old_machinery", "type": "mechanical_trap", "severity": "low"},
            ],
            "rewards": [
                {"id": "rich_ore_vein", "type": "resource", "rarity": "uncommon"},
                {"id": "forgotten_tools", "type": "gear", "rarity": "common"},
            ],
        }
    ],
}

WORLD_KITS = {
    DWARF_DEEP_HOLD_01["id"]: DWARF_DEEP_HOLD_01,
}

