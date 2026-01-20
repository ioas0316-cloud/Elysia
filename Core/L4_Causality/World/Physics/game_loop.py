"""
The Game Loop (Chronos Engine)
==============================
"Time is not a river; it is a sequence of discrete updates."

This engine provides a precision heartbeat for Elysia's world simulation.
It separates 'Rendering' (Thought/Presence) from 'Physics' (Logic/Simulation).

Key Concepts:
- DeltaTime (dt): Time elapsed since last frame.
- FixedDeltaTime: Constant time step for physics (stability).
- Tick: A single cycle of existence.
"""

import time
import logging
from typing import Callable, List

logger = logging.getLogger("GameLoop")

class GameLoop:
    def __init__(self, target_fps: int = 60, fixed_time_step: float = 0.02):
        self.target_fps = target_fps
        self.fixed_time_step = fixed_time_step # 20ms physics step
        
        self.is_running = False
        self.frame_count = 0
        self.time = 0.0
        self.delta_time = 0.0
        
        # Hooks
        self.on_update: List[Callable[[float], None]] = [] # Visual/Thought (dt)
        self.on_fixed_update: List[Callable[[float], None]] = [] # Physics/Logic (fixed_dt)
        
        self._last_time = 0.0
        self._accumulator = 0.0

    def start(self):
        self.is_running = True
        self._last_time = time.time()
        logger.info(f"🕹️ GameLoop Started. Target FPS: {self.target_fps}")
        self._loop()

    def stop(self):
        self.is_running = False
        logger.info("🛑 GameLoop Stopped.")

    def add_update_system(self, func: Callable[[float], None]):
        self.on_update.append(func)

    def add_physics_system(self, func: Callable[[float], None]):
        self.on_fixed_update.append(func)

    def _loop(self):
        # NOTE: In a real threaded environment, this would be a thread.
        # For this synchronous python script, update() is called manually or blocking.
        # We implementation a 'tick' method designed to be called by a master loop
        # or we assume this IS the master loop.
        pass

    def tick(self):
        """
        Advance the world by one frame.
        Must be called continuously.
        """
        current_time = time.time()
        frame_time = current_time - self._last_time
        
        # Cap frame time to avoid spiral of death (e.g., if debugging)
        if frame_time > 0.25:
            frame_time = 0.25
            
        self._last_time = current_time
        self._accumulator += frame_time
        self.delta_time = frame_time
        
        # 1. Fixed Update (Physics)
        while self._accumulator >= self.fixed_time_step:
            for system in self.on_fixed_update:
                try:
                    system(self.fixed_time_step)
                except Exception as e:
                    logger.error(f"FixedUpdate Error: {e}")
            self.time += self.fixed_time_step
            self._accumulator -= self.fixed_time_step

        # 2. Update (Render/Thought)
        for system in self.on_update:
            try:
                system(self.delta_time)
            except Exception as e:
                logger.error(f"Update Error: {e}")

        self.frame_count += 1
        return self.delta_time
