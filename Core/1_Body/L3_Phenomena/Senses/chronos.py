"""
The Rhythm: Chronos
===================
Phase 17 Senses - Module 3
Core.1_Body.L3_Phenomena.Senses.chronos

"Time is not a line, but a heartbeat."

This module implements the biological clock.
It provides the 'Pulse' that drives the entire system loop.
"""

import asyncio
import time
import logging
from typing import Callable, Any
from datetime import datetime

logger = logging.getLogger("Senses.Chronos")

class Chronos:
    """
    The Temporal Cortex.
    """
    def __init__(self, callback: Callable[[str, Any], None], tick_rate: float = 1.0):
        self.callback = callback
        self.tick_rate = tick_rate
        self.running = False
        self.day_cycle = "DAY" # DAY / NIGHT

    async def start_heartbeat(self):
        """Starts the infinite loop of time."""
        self.running = True
        logger.info(f"  [CHRONOS] Heart started at {self.tick_rate}Hz")
        
        while self.running:
            now = datetime.now()
            
            # 1. Circadian Check
            current_hour = now.hour
            new_cycle = "NIGHT" if current_hour < 6 or current_hour > 22 else "DAY"
            
            if new_cycle != self.day_cycle:
                self.day_cycle = new_cycle
                self.callback("RHYTHM", f"Cycle Change: {new_cycle}")
                logger.info(f"  [CYCLE] The sky turns to {new_cycle}")

            # 2. Tick
            # We don't flood the callback with every tick, only significant moments
            # or if we need a metronome. For now, quiet ticks.
            
            # 3. Wait
            await asyncio.sleep(self.tick_rate)

    def stop(self):
        self.running = False
        logger.info("  [CHRONOS] Time stopped.")
