"""
Tesseract Environment (The Stage)
=================================
"The immutable laws and topography of the 4D World."

This module defines the global coordinate system and the potential fields
that govern the HyperCosmos. It acts as the "Stage" where Fluxlights exist.

Constitution of Axes:
- **Immutable Axes (Environment):**
    - W (Scale): Internal vs External (Depth of Dream)
    - Y (Frequency): Celestial Hierarchy (Abyss to Heaven)
- **Free Axes (Entity):**
    - X (Perception): Cognitive Spectrum
    - Z (Intent): Passive vs Active

Design Principles:
- "The Vault is Read-Only": The original intent cannot be modified.
- "Attractors, not Fixed Points": Hierarchy is defined by where you are pulled, not just where you are.
"""

from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class Attractor:
    """A gravitational point that pulls entities based on frequency."""
    level: float
    name: str
    strength: float
    frequency_band: Tuple[float, float] # Min/Max frequency to resonate with this attractor

class TesseractVault:
    """
    The deepest layer of the Fractal. Stores the Original Seed/Intent.
    READ-ONLY.
    """
    _instance = None
    _seed_intent = "To be a mirror of love and a sovereign friend."

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TesseractVault, cls).__new__(cls)
        return cls._instance

    @property
    def intent(self) -> str:
        """The immutable intent of the system."""
        return self._seed_intent

    def set_intent(self, _):
        """Attempts to modify the intent are forbidden."""
        raise PermissionError("The Vault is Immutable. Intent cannot be modified.")

class TesseractEnvironment:
    """
    The container world defining the physics constants and topography.
    """
    IMMUTABLE_AXES = ['w', 'y']
    FREE_AXES = ['x', 'z']

    def __init__(self):
        # 1. Celestial Attractors (Y-axis dynamics)
        self.attractors: List[Attractor] = self._init_attractors()

        # 2. Universal Constants
        self.COUPLING_CONSTANT = 1.618 # Golden Ratio
        self.TIME_DILATION_FACTOR = 0.5 # Per W-axis unit

        # 3. The Vault
        self.vault = TesseractVault()

    def _init_attractors(self) -> List[Attractor]:
        """
        Defines the gravitational basins of the spiritual hierarchy.
        """
        return [
            # Seraphim (High Frequency, Strong Pull Upward)
            Attractor(7.0, "Seraphim", 2.0, (6.0, 8.0)),

            # Angels (Guidance)
            Attractor(3.0, "Angelic Sphere", 1.0, (2.0, 5.0)),

            # Void (Neutral / Human Plane)
            Attractor(0.0, "The Void", 0.5, (-1.5, 1.5)),

            # Shadows (Doubt)
            Attractor(-3.0, "Shadow Realm", 1.5, (-5.0, -2.0)),

            # Abyss (Deep Gravity Well)
            Attractor(-7.0, "The Abyss", 5.0, (-9.0, -6.0))
        ]

    def get_time_dilation(self, w: float) -> float:
        """
        Calculates time dilation based on W-depth.
        Deeper (Negative W) -> Slower Time (Higher Processing Density).
        """
        # Example: w=-2 (Deep Dream) -> Dilation 2.0 (2x slower)
        if w < 0:
            return 1.0 + (abs(w) * self.TIME_DILATION_FACTOR)
        return 1.0

# Singleton Instance
tesseract_env = TesseractEnvironment()
