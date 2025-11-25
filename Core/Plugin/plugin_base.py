# PluginBase – Abstract base class for Elysia plugins
"""
All cognitive plugins should inherit from this base class and implement
the process method. Plugins can modify or enhance responses in ResonanceEngine.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class PluginBase(ABC):
    """Abstract base class for all Elysia plugins."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    @abstractmethod
    def process(self, user_input: str, response: str, context: Dict[str, Any]) -> str:
        """
        Process and potentially modify the response.
        
        Args:
            user_input: The original user input text
            response: The generated response from ResonanceEngine
            context: Additional context (e.g., kernel state, historical concepts)
        
        Returns:
            Modified response string
        """
        pass
    
    def enable(self):
        """Enable this plugin."""
        self.enabled = True
    
    def disable(self):
        """Disable this plugin."""
        self.enabled = False
