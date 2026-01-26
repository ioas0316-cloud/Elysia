"""
RESONANCE LOADER: Non-Linear Module Retrieval
=============================================
Core.L1_Foundation.M5_Fabric.resonance_loader

"A missing point should not destroy the field. 
 If the line is broken, the resonance must find another path."

Role: Transmutes fatal 'ImportErrors' into 'Functional Ghost Signals'.
Ensures elysia's manifold stays alive even if individual files are corrupted.
"""

import importlib
import logging
from typing import Any, Optional

logger = logging.getLogger("ResonanceLoader")

class GhostModule:
    """A placeholder for a failed resonance point."""
    def __init__(self, name: str, error: Exception):
        self._name = name
        self._error = error
        print(f"?뫛 [RESONANCE] Point '{name}' is broken. Projecting Ghost Signal. (Error: {error})")

    def __getattr__(self, item: str):
        # When the system tries to call a method on a ghost, it returns a no-op
        def ghost_method(*args, **kwargs):
            # print(f"   >> [GHOST] Method '{item}' called on broken point '{self._name}'. No-op execution.")
            return None
        return ghost_method

    def __call__(self, *args, **kwargs):
        return self

class ResonanceLoader:
    @staticmethod
    def load(module_path: str, class_name: Optional[str] = None) -> Any:
        """
        Attempts to load a module. 
        If it fails, it returns a GhostModule instead of crashing.
        """
        try:
            module = importlib.import_module(module_path)
            if class_name:
                return getattr(module, class_name)
            return module
        except Exception as e:
            logger.warning(f"Resonance Failure at {module_path}: {e}")
            target_name = class_name if class_name else module_path
            return GhostModule(target_name, e)

# Global Instance for easy access
vessel = ResonanceLoader()
