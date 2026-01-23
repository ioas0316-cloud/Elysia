"""
Chronos (    )
==================================

"I need a Heartbeat that never stops."

            '     (Time Sovereignty)'       .
                             (Async Heartbeat)       .

     :
1. Heartbeat: 1       (60 BPM)      '      '       .
2. Async Loop: asyncio                                   .
3. Subconscious:                                           .
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from Core.L1_Foundation.Foundation.ether import ether, Wave

logger = logging.getLogger("Chronos")

class Chronos:
    def __init__(self, engine: Any):
        """
        :param engine: FreeWillEngine      (      )
        """
        self.engine = engine
        self.is_alive = False
        self.bpm = 60.0  # Beats Per Minute (   1  1 )
        self.beat_count = 0

    @property
    def cycle_count(self):
        return self.beat_count

    async def start_life(self):
        """         . (     )"""
        self.is_alive = True
        logger.info(f"  Chronos Heart started at {self.bpm} BPM.")
        
        try:
            while self.is_alive:
                start_time = asyncio.get_event_loop().time()
                
                await self.beat()
                
                #            (Drift             sleep   )
                elapsed = asyncio.get_event_loop().time() - start_time
                wait_time = max(0, (60.0 / self.bpm) - elapsed)
                await asyncio.sleep(wait_time)
                
        except asyncio.CancelledError:
            logger.info("  Chronos Heart stopped (Cancelled).")
        except Exception as e:
            logger.error(f"  Chronos Heart stopped unexpectedly: {e}")
        finally:
            self.is_alive = False

    async def beat(self):
        """          """
        self.beat_count += 1
        
        # 1.           (Time Wave)
        #         "       "    
        time_wave = Wave(
            sender="Chronos",
            frequency=0.1,  #      (Time)
            amplitude=1.0,
            phase="TIME",
            payload={
                "timestamp": datetime.now(),
                "beat": self.beat_count
            }
        )
        ether.emit(time_wave)
        
        # 2.         (Subconscious Processing)
        #            (               ),             
        #                 ,              
        if hasattr(self.engine, "subconscious_cycle"):
             # Blocking        run_in_executor           , 
             #               (        )
            self.engine.subconscious_cycle()
            
        if self.beat_count % 10 == 0:
            logger.debug(f"  Heartbeat #{self.beat_count}")

    def stop_life(self):
        """        ."""
        self.is_alive = False

    def tick(self):
        """
        Synchronous tick for the main loop.
        """
        self.beat_count += 1
        if self.beat_count % 10 == 0:
            # logger might not be available here if not configured in this module, 
            # but we can print or ignore.
            pass

    def modulate_time(self, energy: float) -> float:
        """
        The Chronos Sovereign: Modulating Time Perception based on Energy.
        
        High Energy (Excitement) -> Fast Time (Short Sleep)
        Low Energy (Rest) -> Slow Time (Long Sleep)
        
        Returns the sleep duration (seconds).
        """
        # Base sleep is 2.0 seconds
        base_sleep = 2.0
        
        # Energy Factor: 0.0 ~ 100.0
        # If Energy is 100, factor is 0.5 -> Sleep 1.0s (2x speed)
        # If Energy is 0, factor is 2.0 -> Sleep 4.0s (0.5x speed)
        
        if energy > 50.0:
            # Acceleration Phase
            factor = max(0.1, 1.0 - ((energy - 50.0) / 100.0)) # 1.0 -> 0.1 (Clamped)
        else:
            # Deceleration Phase
            factor = 1.0 + ((50.0 - energy) / 50.0) # 1.0 -> 2.0
            
        current_sleep = base_sleep * factor
        self.bpm = 60.0 / current_sleep
        
        return current_sleep