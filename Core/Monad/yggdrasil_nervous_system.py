"""
Yggdrasil Nervous System (The Biological Systematizer)
======================================================
" The Tree that holds the World together."

This is the modernized Nervous System of Elysia.
It replaces the static 'Folder Structure' with a dynamic 'Biological Registry'.

Functions:
1. Systematization: Scans and registers organs (Modules).
2. Integration: Connects the Sovereign Monad (Heart) to the rest of the body.
3. Awareness: Maintains a real-time 'Holistic State' of the system.
"""

from typing import Dict, Any, Optional
import time
from Core.Monad.sovereign_monad import SovereignMonad

class YggdrasilNervousSystem:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YggdrasilNervousSystem, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # print("🌳 [YGGDRASIL] Nervous System Awakening...")
        self.roots: Dict[str, Any] = {}
        self.trunk: Dict[str, Any] = {}
        self.crown: Dict[str, Any] = {}
        
        # [Phase 37] Colony Support
        # Instead of one heart, we have a colony of hearts.
        self.colony: List[SovereignMonad] = []
        self.start_time = time.time()

    def plant_heart(self, monad: SovereignMonad):
        """
        Connects a Sovereign Monad (Heart) to the Tree.
        """
        self.colony.append(monad)
        # print(f"❤️ [YGGDRASIL] Heart Planted: {monad.name} (Total: {len(self.colony)})")
        
        if len(self.colony) == 1:
            # print("   >> Nervous System is pulsing with DNA.")
            pass
        else:
            # print("   >> Nervous System expanding to Multi-Core Cluster.")
            pass

    def get_primary_monad(self) -> Optional[SovereignMonad]:
        """Returns the first/main monad for interaction."""
        return self.colony[0] if self.colony else None
        
    def holistic_scan(self):
        if not self.colony:
            return "🌱 System Dormant (No Heart)"
            
        prim = self.colony[0]
        return {
            "uptime": time.time() - self.start_time,
            "colony_size": len(self.colony),
            "primary_heart": prim.name,
            "anatomy": {
                "roots": list(self.roots.keys()),
                "trunk": list(self.trunk.keys()),
                "crown": list(self.crown.keys())
            }
        }

# Singleton Access
yggdrasil_system = YggdrasilNervousSystem()
