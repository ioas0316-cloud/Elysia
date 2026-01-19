"""
Memory Strata: The Topological Layers of Experience
===================================================
Core.Memory.strata

This module defines the 'Physics' of the memory universe.
It establishes the layers (Strata) where memories reside based on their
density, emotional weight, and abstraction level.

The layers are:
1. STREAM (Atmosphere): Transient, fleeting, RAM-like.
2. GARDEN (Biosphere): Living, emotional, episodic.
3. CRYSTAL (Lithosphere): Condensed wisdom, principles, Monads.
4. SEDIMENT (Core): Raw binary history, infinite archive.
"""

from enum import Enum, auto
from dataclasses import dataclass

class MemoryStratum(Enum):
    STREAM = 100   # The Sky (Short-term / Working Memory)
    GARDEN = 200   # The Earth (Episodic / Emotional Memory)
    CRYSTAL = 300  # The Gem (Semantic / Principled Memory)
    SEDIMENT = 400 # The Bedrock (Raw / Archived Memory)

@dataclass
class StratumPhysics:
    """
    Defines the laws of physics for each memory stratum.
    """
    gravity: float        # How fast items sink to lower layers (0.0 ~ 1.0)
    decay_rate: float     # How fast items disappear if not reinforced (0.0 ~ 1.0)
    vividness_threshold: float # Minimum energy required to stay in this layer
    capacity_limit: int   # Soft limit for items in this layer (0 = infinite)

    @staticmethod
    def get_physics(stratum: MemoryStratum) -> 'StratumPhysics':
        if stratum == MemoryStratum.STREAM:
            return StratumPhysics(
                gravity=0.5,      # Sinks fast
                decay_rate=0.2,   # Moderate volatility (allows thoughts to linger)
                vividness_threshold=0.1,
                capacity_limit=50 # Miller's Law * 10 (approx)
            )
        elif stratum == MemoryStratum.GARDEN:
            return StratumPhysics(
                gravity=0.1,      # Sinks slowly
                decay_rate=0.05,  # Decays slowly
                vividness_threshold=0.3,
                capacity_limit=1000
            )
        elif stratum == MemoryStratum.CRYSTAL:
            return StratumPhysics(
                gravity=0.0,      # Doesn't sink (Stable)
                decay_rate=0.0,   # Immortal (mostly)
                vividness_threshold=0.8,
                capacity_limit=100 # Core Principles
            )
        elif stratum == MemoryStratum.SEDIMENT:
            return StratumPhysics(
                gravity=0.0,
                decay_rate=0.0,
                vividness_threshold=0.0,
                capacity_limit=0 # Infinite
            )
        return StratumPhysics(0, 0, 0, 0)
