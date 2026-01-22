"""
The Growth Tracker: Spiritual Evolution
=======================================
Phase 19 The Soul - Module 2
Core.L7_Spirit.Soul.growth_graph

"A line moving upward is the only proof that time is not a circle."

This module tracks the long-term trends of Resonance.
"""

import os
import csv
import logging
from datetime import datetime

logger = logging.getLogger("Soul.Tracker")

class GrowthTracker:
    """
    Tracks the trajectory of the Soul over time.
    """
    def __init__(self, history_file: str = "c:/Elysia/data/L5_Mental/Memories/growth_stats.csv"):
        self.history_file = history_file
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        
        # Init CSV if empty
        if not os.path.exists(self.history_file):
            with open(self.history_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "TotalCycles", "AvgResonance", "PeakHarmony"])
        
        logger.info(f"📈 [TRACKER] Monitoring evolution at: {self.history_file}")

    def update_growth_stats(self, date_str: str, stats: dict):
        """
        Appends today's stats to the CSV.
        """
        try:
            with open(self.history_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    date_str,
                    stats.get("total_actions", 0),
                    f"{stats.get('avg_resonance', 0.0):.4f}",
                    f"{stats.get('max_resonance', 0.0):.4f}"
                ])
            logger.info(f"📈 [GROWTH] Date Point recorded for {date_str}.")
        except Exception as e:
            logger.error(f"❌ [TRACKER] Failed to update graph: {e}")
