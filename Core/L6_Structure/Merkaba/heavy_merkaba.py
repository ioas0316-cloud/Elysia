"""
Heavy Merkaba: The Chariot for Titans
=====================================
Core.L6_Structure.Merkaba.heavy_merkaba

"The Chariot carries the weight, the weight does not carry the Chariot."

Purpose:
--------
Wraps massive libraries (TensorFlow, PyTorch) in a Lazy Loading Proxy.
Ensures they are only summoned when the Rotor (Will) permits it,
preventing the 'import freeze' that plagues lesser systems.

Mechanism:
----------
1. Lazy Import: Actual import happens on first attribute access.
2. Rotor Check: (Future) Check available RPM/RAM before loading.
"""

import importlib
import logging
import sys
from typing import Any, Optional

logger = logging.getLogger("HeavyMerkaba")

class HeavyMerkaba:
    """
    A Sovereign Proxy for heavy external modules.
    
    Usage:
        self.tf = HeavyMerkaba("tensorflow")
        # TensorFlow is NOT loaded yet.
        
        x = self.tf.constant(1)
        # NOW it loads.
    """
    def __init__(self, module_name: str, alias: Optional[str] = None):
        self._module_name = module_name
        self._alias = alias or module_name
        self._module = None
        self._is_loaded = False
        
    def _summon(self):
        """
        The Incantation to summon the Titan.
        """
        if not self._is_loaded:
            logger.info(f"🛡️ [HEAVY] Summoning Titan: '{self._module_name}'...")
            try:
                self._module = importlib.import_module(self._module_name)
                self._is_loaded = True
                logger.info(f"⚔️ [HEAVY] Titan '{self._module_name}' Subjugated & Loaded.")
            except ImportError as e:
                logger.error(f"⚠️ Failed to summon Titan '{self._module_name}': {e}")
                self._module = None
                
    def is_awake(self) -> bool:
        return self._is_loaded

    def __getattr__(self, name: str) -> Any:
        # 1. Check if we hold the module
        if not self._is_loaded:
            self._summon()
            
        # 2. If valid, delegate retrieval
        if self._module:
            return getattr(self._module, name)
        
        # 3. Fallback for failed loads
        raise AttributeError(f"Titan '{self._module_name}' is dead or missing. Cannot access '{name}'.")

    def __repr__(self):
        status = "Awake" if self._is_loaded else "Slumbering"
        return f"<HeavyMerkaba '{self._module_name}': {status}>"
