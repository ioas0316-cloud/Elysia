from typing import Callable, Any, Dict

class SensoryChannel:
    """Base interface for all data-ingesting channels."""
    def __init__(self, name: str):
        self.name = name
        self.callback: Callable[[Any], None] = None

    def register_callback(self, callback: Callable[[Any], None]):
        self.callback = callback

    def start(self):
        pass

    def stop(self):
        pass

class ExpressiveChannel:
    """Base interface for all outward-manifesting channels."""
    def __init__(self, name: str):
        self.name = name

    def express(self, payload: Dict[str, Any]):
        pass
