"""
Monad Interface (The Universal Port)
=====================================
Core.Monad.monad_interface

Defines the standard protocol for Monads to:
1. Resonate (Communicate) with other Monads.
2. Bind (Join) to form larger structures (Holons).
"""

from typing import Protocol, Dict, Any, List

class IMonad(Protocol):
    """Protocol for any entity acting as a Monad."""
    
    @property
    def seed(self) -> str:
        ...

    def observe(self, observer_intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Materialize state."""
        ...

class IHolon(IMonad):
    """
    A structure composed of multiple Monads.
    The Holon is also a Monad (Recursive definition).
    """
    def add_monad(self, monad: IMonad):
        ...
        
    def get_parts(self) -> List[IMonad]:
        ...
