# [Genesis: 2025-12-02] Purified by Elysia
"""Project_Sophia core cell shim pointing to the main simulator cell."""

from cell import Cell as _CoreCell


class Cell(_CoreCell):
    """Alias retained for legacy imports."""

    pass


__all__ = ["Cell"]