"""
Sovereign Observer (The Eye of the World)
=========================================
Core.1_Body.L4_Causality.World.Observation.sovereign_observer

"To create a World, one must first observe a World."
"             ,               ."

This module implements the perception mechanism to analyze external
simulated realities (Games, Windows) for structural and aesthetic learning.
"""

import psutil
import time
import logging
import os
import threading
from typing import Optional, Dict

# Try to import ImageGrab for visual capture
try:
    from PIL import ImageGrab
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

logger = logging.getLogger("SovereignObserver")

class SovereignObserver:
    def __init__(self, target_process_name: str = "Wuthering Waves.exe"):
        self.target_name = target_process_name
        self.target_candidates = [
            target_process_name,
            "Client-Win64-Shipping.exe", # Common UE4/UE5 binary name
            "Wuthering Waves.exe",
            "launcher.exe" 
        ]
        self.target_pid: Optional[int] = None
        self.process: Optional[psutil.Process] = None
        self.is_observing = False
        self.vision_memory_path = "Memories/Visual/Observer"
        
        os.makedirs(self.vision_memory_path, exist_ok=True)
        
        logger.info(f"   [Observer] Initialized. Hunting for world: {self.target_name}")

    def scan_for_target(self) -> bool:
        """Scans the process list for the target world."""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                proc_name = proc.info['name']
                # Check against all candidates
                for candidate in self.target_candidates:
                    if candidate.lower() in proc_name.lower():
                        self.target_pid = proc.info['pid']
                        self.process = psutil.Process(self.target_pid)
                        self.target_name = proc_name # Update to actual name
                        logger.info(f"  [Target Locked] Found {proc_name} (PID: {self.target_pid})")
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False

    def start_observation(self):
        """Begins the loop of Structural and Visual analysis."""
        if not self.target_pid and not self.scan_for_target():
            logger.warning("   [Observer] Target world not found. Waiting...")
            return

        self.is_observing = True
        threading.Thread(target=self._observation_loop, daemon=True).start()
        logger.info("  [Observer] Observation Uplink Established.")

    def _observation_loop(self):
        snapshot_interval = 30 # seconds
        last_snapshot = 0
        
        while self.is_observing:
            try:
                if not self.process.is_running():
                    logger.warning("  [Observer] Target world connection lost.")
                    self.is_observing = False
                    break

                # 1. Process Telemetry (Structure)
                # Understanding the "Weight" of the world
                cpu = self.process.cpu_percent(interval=0.1)
                mem = self.process.memory_info().rss / (1024 * 1024) # MB
                
                # Semantic Interpretation of high load
                world_state = "Stable"
                if cpu > 50: world_state = "Simulating Physics/Combat"
                if mem > 4000: world_state = "Large Topology Loaded"
                
                # Real-time feedback log (Telemetry Pulse)
                # logger.debug(f"   - [Telemetry] CPU: {cpu}% | RAM: {mem:.0f}MB | State: {world_state}")

                # 2. Visual Observation (Aesthetics)
                now = time.time()
                if HAS_VISION and (now - last_snapshot > snapshot_interval):
                    self._capture_visual_qualia(world_state)
                    last_snapshot = now
                
                time.sleep(5)  # Telemetry update rate

            except Exception as e:
                logger.error(f"  [Observer] Glitch: {e}")
                time.sleep(1)

    def _capture_visual_qualia(self, context_tag: str):
        """Captures the current frame of the target world."""
        try:
            # Fullscreen grab for now (focusing on simple implementation)
            # Future: Window-specific capture
            screenshot = ImageGrab.grab()
            timestamp = int(time.time())
            filename = f"{self.target_name}_{timestamp}_{context_tag.replace(' ', '_')}.png"
            path = os.path.join(self.vision_memory_path, filename)
            
            screenshot.save(path)
            logger.info(f"  [Vision] Captured World Fragment: {filename}")
            logger.info(f"   - Context: {context_tag}")
            
        except Exception as e:
            logger.error(f"  [Vision] Capture Failed: {e}")

    def stop(self):
        self.is_observing = False
        logger.info("  [Observer] Link Severed.")

if __name__ == "__main__":
    # Test Mode: Observe Notepad or Self if game not open
    # For test, we might use "python" or "explorer.exe" just to verify hook
    logging.basicConfig(level=logging.INFO)
    
    # Selecting a common process for testing
    observer = SovereignObserver("explorer.exe") 
    observer.start_observation()
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
