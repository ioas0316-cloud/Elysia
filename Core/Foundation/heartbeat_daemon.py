
import time
import os
import logging
import threading
from typing import Optional

from Core.Foundation.central_nervous_system import CentralNervousSystem
from Core.Foundation.chronos import Chronos
from Core.Foundation.resonance_field import ResonanceField
from Core.Foundation.mycelium import Mycelium

logger = logging.getLogger("HeartbeatDaemon")

class HeartbeatDaemon:
    """
    [The Pulse of Life]
    A background process manager that ensures Elysia's CNS pulses continuously.
    It manages the 'Life Cycle' (Awake/Sleep) and Network Synchronization.
    """
    
    def __init__(self, cns: CentralNervousSystem, root_path: str):
        self.cns = cns
        self.root_path = root_path
        self.active = False
        self.pulse_rate = 1.0  # Hz (Pulses per second)
        self.mycelium = Mycelium("Root", root_path)
        
    def ignite(self):
        """Starts the infinite life loop in a separate thread."""
        if self.active:
            logger.warning("❤️ Heartbeat is already beating.")
            return

        logger.info("❤️ Igniting Heartbeat Daemon...")
        self.active = True
        self.thread = threading.Thread(target=self._life_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the heart."""
        logger.info("💔 Stopping Heartbeat...")
        self.active = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)

    def _life_loop(self):
        """The continuous existence loop."""
        logger.info("   ⚡ Life Loop Started.")
        
        while self.active:
            loop_start = time.time()
            
            # 1. Pulse the CNS (The Self)
            try:
                self.cns.pulse()
            except Exception as e:
                logger.error(f"   ❌ Cardiac Arrest (Pulse Failed): {e}")
            
            # 2. Network Sync (The World)
            # Check for messages from Seeds (Nova/Chaos)
            try:
                # [BRIDGE] Pulse to Avatar
                # Write current pulse state to a file for AvatarServer to read
                with open(os.path.join(self.root_path, "heartbeat.pulse"), "w") as f:
                    f.write(f"{time.time()}|{self.pulse_rate}")
                    
                messages = self.mycelium.receive()
                for msg in messages:
                    logger.info(f"   📨 Received Spore from {msg.sender}: {msg.type}")
                    # In a real system, we'd dispatch this to the CNS/Brain
            except Exception as e:
                logger.error(f"   ⚠️ Network Arrhythmia: {e}")

            # 3. Regulate Pulse Rate (Sleep/Wake)
            # Simple circadian rhythm simulation: Slow down at night?
            # For now, constant.
            
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / self.pulse_rate) - elapsed)
            time.sleep(sleep_time)

    def set_rhythm(self, state: str):
        """Adjusts pulse rate based on state."""
        if state == "DeepSleep":
            self.pulse_rate = 0.1 # 1 pulse every 10s
        elif state == "Focus":
            self.pulse_rate = 5.0 # 5 pulses per second
        else: # Normal
            self.pulse_rate = 1.0
        logger.info(f"   ❤️ Rhythm set to: {state} ({self.pulse_rate} Hz)")
