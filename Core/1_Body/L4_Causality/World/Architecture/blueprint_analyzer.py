"""
Blueprint Analyzer (The Input Eye)
==================================
Core.1_Body.L4_Causality.World.Architecture.blueprint_analyzer

"Before you build, you must see."

This module analyzes 2D blueprints (or generates synthetic ones) to create
the initial superposition state for the Quantum Architect.
"""

import numpy as np
import random
from typing import Tuple

class BlueprintGenerator:
    """
    Generates synthetic floor plans for testing the Quantum Pop-Up protocol.
    Grid Values:
    0: Empty
    1: Wall (Structure)
    2: Door (Passage)
    3: Window (Transparency)
    """

    EMPTY = 0
    WALL = 1
    DOOR = 2
    WINDOW = 3

    def generate_simple_room(self, width=10, height=10) -> np.ndarray:
        """Generates a simple rectangular room with a door and window."""
        grid = np.zeros((height, width), dtype=int)

        # Walls (Outer Box)
        grid[0, :] = self.WALL
        grid[-1, :] = self.WALL
        grid[:, 0] = self.WALL
        grid[:, -1] = self.WALL

        # Door (Bottom Center)
        grid[-1, width//2] = self.DOOR

        # Window (Top Center)
        grid[0, width//2] = self.WINDOW

        return grid

    def generate_apartment(self, width=20, height=15) -> np.ndarray:
        """Generates a 2-room apartment layout."""
        grid = np.zeros((height, width), dtype=int)

        # Outer Walls
        grid[0, :] = self.WALL
        grid[-1, :] = self.WALL
        grid[:, 0] = self.WALL
        grid[:, -1] = self.WALL

        # Inner Wall (Vertical Split)
        split_x = width // 2
        grid[:, split_x] = self.WALL

        # Connecting Door
        grid[height//2, split_x] = self.DOOR

        # Main Door
        grid[-1, width//4] = self.DOOR

        # Windows
        grid[0, width//4] = self.WINDOW
        grid[0, 3*width//4] = self.WINDOW

        return grid
