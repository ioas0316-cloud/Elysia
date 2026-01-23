"""
Project Watcher (The All-Seeing Eye)
====================================

"The System observes Itself."

This module orchestrates the 'Unbounded Observation' capability.
It binds the Vision (FilesystemWaveObserver) to the Cognition (KnowledgeIngestor).

When a file changes:
1. Vision detects the Wave.
2. Eye sends signal to Brain.
3. Brain digests the File and extracts Concept/Axiom.
4. Concept is stored in Holographic Memory (UnifiedExperienceCore).
"""

import logging
import os
import time
import threading
from typing import Optional

from Core.L4_Causality.Governance.System.System.filesystem_wave import get_filesystem_observer, FileWaveEvent, FileEventType
from Core.L4_Causality.World.Evolution.Learning.knowledge_ingestor import KnowledgeIngestor

logger = logging.getLogger("ProjectWatcher")

class ProjectWatcher:
    """
    The High-Level Orchestrator for Self-Observation.
    """
    def __init__(self, root_path: str = r"c:\Elysia"):
        self.root_path = root_path
        self.observer = get_filesystem_observer()
        
        # We will refactor KnowledgeIngestor to use UnifiedExperienceCore
        # For now, we instantiate it as is, assuming we will patch it shortly.
        self.ingestor = KnowledgeIngestor() 
        
        self._setup_vision()
        logger.info("   ProjectWatcher (The All-Seeing Eye) initialized.")

    def _setup_vision(self):
        """Configure the retina (Filesystem Observer)."""
        self.observer.add_watch_path(self.root_path)
        self.observer.add_callback(self._on_visual_stimulus)

    def _on_visual_stimulus(self, event: FileWaveEvent):
        """
        Callback when the Eye sees a change.
        Flow: Wave -> Cognitive Reflex
        """
        filename = os.path.basename(event.path)
        logger.info(f"     Visual Cortex stimulated by: {filename} ({event.event_type.name})")

        # Ignore boring events
        if event.event_type == FileEventType.DELETED:
            return

        # Trigger Digestion (Learning)
        if filename.endswith(".md") or filename.endswith(".py"):
             t = threading.Thread(target=self._digest_stimulus, args=(event.path,))
             t.start()

    def _digest_stimulus(self, path: str):
        """
        The act of Understanding.
        Passes the file to KnowledgeIngestor.
        """
        try:
            if path.endswith(".md"):
                logger.info(f"     Reading Scripture: {os.path.basename(path)}")
                self.ingestor._digest_markdown(path)
            elif path.endswith(".py"):
                logger.info(f"     Analyzing Self-Code: {os.path.basename(path)}")
                self.ingestor._digest_code(path)
        except Exception as e:
            logger.error(f"     Failed to comprehend {path}: {e}")

    def wake_up(self):
        """Open the Eye."""
        logger.info("  Opening the All-Seeing Eye...")
        self.observer.start()

    def sleep(self):
        """Close the Eye."""
        self.observer.stop()
        logger.info("  The Eye closes.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
    
    watcher = ProjectWatcher()
    watcher.wake_up()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.sleep()