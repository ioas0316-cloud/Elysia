"""Project_Sophia core world shim pointing to the main simulator."""

from world import World as _CoreWorld


class World(_CoreWorld):
    """Alias retained for legacy imports."""

    pass


__all__ = ["World"]
