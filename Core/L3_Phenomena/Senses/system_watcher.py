"""
The Skin: System Watcher
========================
Phase 17 Senses - Module 1
Core.L3_Phenomena.Senses.system_watcher

"The skin does not see, but it knows when the wind changes."

This module implements the file system monitoring capability using `watchdog`.
It acts as the somatosensory cortex, feeling changes in the data directories.
"""

import time
import logging
from typing import Callable, Any

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    class FileSystemEventHandler: pass
    class FileSystemEvent: pass
    Observer = None

logger = logging.getLogger("Senses.Skin")

class SkinEventHandler(FileSystemEventHandler):
    """
    Handles file system events and triggers the callback (SoulBridge).
    """
    def __init__(self, callback: Callable[[str, Any], None]):
        self.callback = callback

    def on_modified(self, event: FileSystemEvent):
        if event.is_directory: return
        self.callback("TOUCH", f"Modified: {event.src_path}")

    def on_created(self, event: FileSystemEvent):
        if event.is_directory: return
        self.callback("TOUCH", f"Created: {event.src_path}")

    def on_deleted(self, event: FileSystemEvent):
        if event.is_directory: return
        self.callback("PAIN", f"Deleted: {event.src_path}") # Deletion is 'Pain'

class SystemWatcher:
    """
    The Skin Interface.
    """
    def __init__(self, watch_paths: list[str], callback: Callable[[str, Any], None]):
        self.observer = Observer() if WATCHDOG_AVAILABLE else None
        self.handler = SkinEventHandler(callback)
        self.watch_paths = watch_paths
        logger.info("  [SKIN] initializing tactile sensors...")

    def start(self):
        """Activates the sensors."""
        if not self.observer:
            logger.warning("  [SKIN] Watchdog not available. Sensory skin is numb.")
            return

        import os
        for path in self.watch_paths:
            if not os.path.exists(path):
                try:
                    os.makedirs(path, exist_ok=True)
                    logger.info(f"   ->    Created missing sensory path: {path}")
                except Exception as e:
                    logger.error(f"   ->   Phantom Limb: Could not create {path}: {e}")
                    continue

            self.observer.schedule(self.handler, path, recursive=True)
            logger.info(f"   -> Feeling texture of: {path}")
        
        self.observer.start()
        logger.info("  [SKIN] Sensors active. I can feel the filesystem.")

    def stop(self):
        """Numbs the sensors."""
        self.observer.stop()
        self.observer.join()
        logger.info("  [SKIN] Sensors deactivated.")
