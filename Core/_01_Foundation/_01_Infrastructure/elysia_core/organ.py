"""
Organ: 유기적 연결 인터페이스
==============================
"""

from typing import TypeVar, Type, Optional, Any
from .cell import get_registry

class CellNotFoundError(Exception): pass

class Organ:
    _scanner = None
    _initialized = False
    
    @classmethod
    def initialize(cls, root_path: str = None, auto_scan: bool = True):
        if cls._initialized: return
        if root_path is None: root_path = "C:/Elysia"
        if auto_scan:
            from .scanner import NeuralScanner
            cls._scanner = NeuralScanner(root_path)
            cls._scanner.scan()
        cls._initialized = True
        print(f"🫀 Organ system initialized.")
    
    @classmethod
    def get(cls, identity: str, instantiate: bool = True) -> Any:
        registry = get_registry()
        if identity not in registry:
            if not cls._initialized:
                cls.initialize()
                registry = get_registry()
            if identity not in registry:
                raise CellNotFoundError(f"Cell '{identity}' not found. Available: {list(registry.keys())}")
        cell_class = registry[identity]
        return cell_class() if instantiate else cell_class
