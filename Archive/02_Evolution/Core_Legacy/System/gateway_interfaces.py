from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

class SensoryChannel(ABC):
    """
    Interface for any system that provides input to Elysia.
    It runs independently and pushes data to the central engine.
    """
    def __init__(self, name: str):
        self.name = name
        self.callback: Callable[[str], None] = None

    def register_callback(self, callback: Callable[[str], None]):
        """The core engine registers its listening function here."""
        self.callback = callback

    @abstractmethod
    def start(self):
        """Start the sensory loop (e.g., listening to mic, polling chat)."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the sensory loop."""
        pass


class ExpressiveChannel(ABC):
    """
    Interface for any system that renders Elysia's state and voice.
    It receives rich payloads from the engine and translates them to output.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def express(self, payload: Dict[str, Any]):
        """
        Process the expression payload.
        Example payload:
        {
            "text": "Hello, Architect.",
            "voice_hz": 120,
            "stress": 0.5,
            "monad_state": {
                "joy": 85.0,
                "coherence": 0.9,
                "entropy": 0.1
            }
        }
        """
        pass
