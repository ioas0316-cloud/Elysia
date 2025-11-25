"""ElysiaWorld cell wrapper.

This module exposes the Project_Sophia.core.cell.Cell class under the
new packaging namespace so external users can import
`elysia_world.cell.Cell` without depending on the legacy folder layout.
"""

from Project_Sophia.core.cell import Cell as _CoreCell

class Cell(_CoreCell):
    """Thin alias to retain backwards compatibility."""

    pass

__all__ = ["Cell"]
