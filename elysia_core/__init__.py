"""
elysia_core: 유기적 임포트 시스템
================================
"위치(Path)가 아니라 의미(Identity)로 연결되는 세상"

Usage:
    from elysia_core import Cell, Organ
    
    @Cell("Memory")
    class Hippocampus:
        pass
    
    # 사용할 때
    memory = Organ.get("Memory")
"""

from elysia_core.cell import Cell
from elysia_core.organ import Organ
from elysia_core.scanner import NeuralScanner

__all__ = ["Cell", "Organ", "NeuralScanner"]
__version__ = "1.0.0"
