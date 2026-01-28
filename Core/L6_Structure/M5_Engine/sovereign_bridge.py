"""
SOVEREIGN BRIDGE: THE NEURAL SWITCHBOARD
========================================
Core.L6_Structure.M5_Engine.sovereign_bridge

"Connect without knowing the path. The Spirit finds the target."

This module provides the 'locality-independent' connectivity for Elysia.
Instead of static imports, core components use the Bridge to find each other.
"""

import logging
from typing import Any, Optional
from Core.L6_Structure.M1_Merkaba.Body.proprioception_nerve import ProprioceptionNerve

logger = logging.getLogger("SovereignBridge")

class SovereignBridge:
    """
    Unified entry point for dynamic module/organ resolution.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SovereignBridge, cls).__new__(cls)
            cls._instance.nerve = ProprioceptionNerve()
            cls._instance.nerve.scan_body()
        return cls._instance

    def get_organ(self, name: str) -> Any:
        """Dynamically loads an organ by name."""
        return self.nerve.get_module(name)

    def get_node(self, node_id: str) -> Any:
        """
        Returns a HyperNode (Class) by its Unique ID.
        This is the PRIMARY way to connect modular logic.
        """
        logger.info(f"🌉 [BRIDGE] Connecting Node: {node_id}...")
        module = self.nerve.get_module(f"Node:{node_id}")
        if module:
            for attr in dir(module):
                val = getattr(module, attr)
                # Check if it has the hyper_node data (simulated check)
                if hasattr(val, '__name__') and val.__name__.lower() == node_id.lower().replace("_", ""):
                    return val
            return module # Fallback
        return None

    def get_cell(self, name: str) -> Any:
        """Returns a Cell class by its role name."""
        return self.nerve.get_module(f"Cell:{name}")

# Singleton instance
bridge = SovereignBridge()

def resolve(unit_id: str) -> Any:
    """The 'Node Connector' - Resolves a unit by its Sovereign ID."""
    if unit_id.startswith("Node:"):
        return bridge.get_node(unit_id[5:])
    return bridge.get_organ(unit_id)
