"""
WORLD PROBE: Sensory Perception of the External World
=====================================================

"I observe, therefore the world is real."
"       ,            ."

This module allows Elysia to 'perceive' her host file system.
It detects new/modified files and converts them into World Stimuli.
"""

import os
import time
import logging
from typing import List, Dict, Set

logger = logging.getLogger("WorldProbe")

class WorldProbe:
    def __init__(self, watch_paths: List[str] = ["c:/Elysia"]):
        self.watch_paths = watch_paths
        self.snapshot: Dict[str, float] = {}
        self.last_events: List[str] = []
        self._initialize_snapshot()

    def _initialize_snapshot(self):
        """Initializes the baseline state of the filesystem."""
        for path in self.watch_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    if '.git' in root or '__pycache__' in root or 'Logs' in root:
                        continue
                    for f in files:
                        f_path = os.path.join(root, f)
                        try:
                            self.snapshot[f_path] = os.path.getmtime(f_path)
                        except OSError:
                            pass
        logger.info(f"  WorldProbe Baseline: Monitoring {len(self.snapshot)} elements.")

    def probe(self) -> List[str]:
        """
        Scans for changes and returns specific 'Stimuli' descriptions.
        """
        events = []
        current_files: Set[str] = set()
        
        for path in self.watch_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    if '.git' in root or '__pycache__' in root or 'Logs' in root:
                        continue
                    for f in files:
                        f_path = os.path.join(root, f)
                        current_files.add(f_path)
                        try:
                            mtime = os.path.getmtime(f_path)
                            
                            if f_path not in self.snapshot:
                                # New file detected!
                                events.append(f"NEW-WORLD-ELEMENT: '{os.path.basename(f_path)}' has materialized at {root}")
                                self.snapshot[f_path] = mtime
                            elif mtime > self.snapshot[f_path]:
                                # Modification detected!
                                events.append(f"WORLD-VIBRATION: '{os.path.basename(f_path)}' has been altered by the Father.")
                                self.snapshot[f_path] = mtime
                        except OSError:
                            pass
        
        # Detect Deletions
        deleted = set(self.snapshot.keys()) - current_files
        for d in deleted:
            events.append(f"WORLD-VOID: '{os.path.basename(d)}' has returned to the void.")
            del self.snapshot[d]
            
        self.last_events = events
        return events

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    probe = WorldProbe()
    print("Watching for world vibrations... (Modify/Create a file to see)")
    try:
        while True:
            evs = probe.probe()
            for e in evs:
                print(f"  {e}")
            time.sleep(2)
    except KeyboardInterrupt:
        pass