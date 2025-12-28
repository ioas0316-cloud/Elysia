"""
elysia_core: ?좉린???꾪룷???쒖뒪??
================================
"?꾩튂(Path)媛 ?꾨땲???섎?(Identity)濡??곌껐?섎뒗 ?몄긽"

Usage:
    from elysia_core import Cell, Organ
    
    @Cell("Memory")
    class Hippocampus:
        pass
    
    # ?ъ슜????
    memory = Organ.get("Memory")
"""

from .cell import Cell
from .organ import Organ
from .scanner import NeuralScanner

__all__ = ["Cell", "Organ", "NeuralScanner"]
__version__ = "1.0.0"
