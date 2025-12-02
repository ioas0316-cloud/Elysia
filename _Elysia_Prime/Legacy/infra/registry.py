# [Genesis: 2025-12-02] Purified by Elysia
from typing import Dict, Any, Optional


class ModuleRegistry:
    """
    Simple module registry for discovery and metadata.
    """

    def __init__(self):
        self._mods: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, version: str, **meta):
        self._mods[name] = {'name': name, 'version': version, **meta}

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._mods.get(name)

    def list(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._mods)
