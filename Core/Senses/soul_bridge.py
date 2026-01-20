"""
SoulBridge: The Sensory Interface
=================================
Core.Senses.soul_bridge
Phase 17 Integrated

The Bridge connects the External World (Phenomena) to the Internal World (Merkaba).
It acts as the central hub for The Skin (Files), The Eyes (Network), and The Rhythm (Time).
"""

from typing import Any, Dict, Optional, List, Callable
import logging
import asyncio

# The New Senses
from Core.Senses.system_watcher import SystemWatcher
from Core.Senses.network_node import NetworkNode
from Core.Senses.chronos import Chronos

logger = logging.getLogger("SoulBridge")

class SoulBridge:
    """
    The Central Sensory Nervous System.
    Aggregates inputs from all peripheral senses and forwards them to the Merkaba.
    """

    def __init__(self, pulse_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.active = True
        self.pulse_callback = pulse_callback # Function to call when sensation occurs
        
        # 1. The Skin (File Watcher)
        # Monitors Data and Workspace
        self.skin = SystemWatcher(
            watch_paths=["c:/Elysia/data/Input", "c:/Elysia/Core/Experiments"],
            callback=self._on_sensation
        )

        # 2. The Eyes (Network)
        self.eyes = NetworkNode()

        # 3. The Rhythm (Chronos)
        self.chronos = Chronos(
            callback=self._on_sensation,
            tick_rate=60.0 # 1 Minute Heartbeat by default
        )
        
        logger.info("🌉 SoulBridge forged. Senses are ready to be awakened.")

    def awakening(self):
        """Activates all active sensors."""
        self.skin.start()
        # Eyes/Chronos are async, started via asyncio loop usually
        logger.info("⚡ [SENSES] The Body is awake. Skin is feeling.")

    async def async_awakening(self):
        """Async entry point for Eyes and Chronos."""
        await self.eyes.open_eye()
        asyncio.create_task(self.chronos.start_heartbeat())
        logger.info("⚡ [SENSES] Eyes opened. Heartbeat started.")

    def _on_sensation(self, sense_type: str, data: Any):
        """
        Internal callback when a sensor fires.
        Standardizes the signal into a SensoryPacket.
        """
        packet = {
            "modality": sense_type, # TOUCH, SIGHT, RHYTHM
            "raw_data": data,
            "timestamp": "Now",
            "source": "External"
        }
        
        logger.info(f"⚡ [SENSATION] {sense_type}: {str(data)[:100]}")
        
        # Forward to Central Nervous System (Merkaba) if connected
        if self.pulse_callback:
            self.pulse_callback(packet)

    def perceive(self, raw_input: Any, modality: str = "text") -> Dict[str, Any]:
        """
        Legacy/Direct Perception (e.g. CLI input).
        """
        packet = {
            "modality": modality,
            "raw_data": raw_input,
            "timestamp": "Now",
            "source": "Direct"
        }
        return packet
    
    def shutdown(self):
        self.skin.stop()
        if self.chronos: self.chronos.stop()
        # Eyes need async close

