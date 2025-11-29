"""
Context Manager (Dual-Layer Language System)
============================================
Restored from Legacy/Language/dual_layer_language.py

Manages the "Symbol Layer" (Language) and "Khala Layer" (Context/Emotion).
Integrates with Hippocampus to store Symbols and Narrative History.
"""

import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from Core.Mind.hippocampus import Hippocampus

logger = logging.getLogger("ContextManager")

@dataclass
class Symbol:
    """
    A Symbol is a crystallized concept with meaning and usage stats.
    """
    name: str
    meaning: str
    complexity: int = 1  # 1=Proto, 5=Narrative
    usage_count: int = 0
    misunderstanding_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "meaning": self.meaning,
            "complexity": self.complexity,
            "usage_count": self.usage_count,
            "misunderstanding_count": self.misunderstanding_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Symbol':
        return cls(
            name=data.get("name", ""),
            meaning=data.get("meaning", ""),
            complexity=data.get("complexity", 1),
            usage_count=data.get("usage_count", 0),
            misunderstanding_count=data.get("misunderstanding_count", 0)
        )

class ContextManager:
    """
    Manages the conversational context and vocabulary (Lexicon).
    """
    def __init__(self, hippocampus: Hippocampus):
        self.memory = hippocampus
        self.active_context: List[str] = [] # Short-term context window
        self.narrative_history: List[Dict[str, Any]] = [] # Long-term history
        
        # Cache for frequently used symbols
        self.symbol_cache: Dict[str, Symbol] = {}
        
        logger.info("✅ Context Manager (Dual-Layer) initialized.")

    def process_input(self, text: str, role: str = "user") -> None:
        """
        Process incoming text:
        1. Extract symbols
        2. Update usage stats
        3. Update context window
        """
        words = text.lower().split()
        
        # 1. Update Context
        self.active_context.append(f"{role}: {text}")
        if len(self.active_context) > 10:
            self.active_context.pop(0)
            
        # 2. Process Symbols
        for word in words:
            if len(word) < 2: continue
            
            # Check if symbol exists in Memory
            symbol = self._get_symbol(word)
            
            if symbol:
                # Update stats
                symbol.usage_count += 1
                self._save_symbol(symbol)
            else:
                # New Symbol? Or just a word?
                # For now, we auto-create symbols for significant words
                # In the future, this should be more selective
                if len(word) > 3:
                    new_symbol = Symbol(name=word, meaning="Unknown", complexity=1)
                    self._save_symbol(new_symbol)

    def _get_symbol(self, name: str) -> Optional[Symbol]:
        """Retrieve symbol from Cache or Memory."""
        if name in self.symbol_cache:
            return self.symbol_cache[name]
        
        # Check Hippocampus
        data = self.memory.storage.get_concept(f"symbol_{name}")
        if data:
            # It's stored as a concept with type="symbol"
            # Data might be compact list or dict. 
            # Assuming we store it as dict in metadata or directly.
            # Let's assume we store it in metadata for now.
            if isinstance(data, dict) and "symbol_data" in data:
                symbol = Symbol.from_dict(data["symbol_data"])
                self.symbol_cache[name] = symbol
                return symbol
            elif isinstance(data, dict): # Legacy/Standard concept
                 # Convert standard concept to Symbol wrapper
                 return Symbol(name=name, meaning=str(data), complexity=1)
                 
        return None

    def _save_symbol(self, symbol: Symbol):
        """Save symbol to Hippocampus."""
        self.symbol_cache[symbol.name] = symbol
        
        # Store as concept
        self.memory.add_concept(
            f"symbol_{symbol.name}", 
            concept_type="symbol",
            metadata={"symbol_data": symbol.to_dict()}
        )

    def get_context_summary(self) -> str:
        """Return a summary of the active context."""
        return "\n".join(self.active_context)

    def get_khala_state(self) -> str:
        """
        Get the 'Khala' (Emotional) state of the context.
        Uses Synesthesia to analyze the context words.
        """
        # This would use SynesthesiaEngine to analyze the active_context
        # For now, return a placeholder or simple analysis
        return "Neutral"
