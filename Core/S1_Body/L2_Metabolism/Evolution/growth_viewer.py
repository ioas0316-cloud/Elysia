"""
GROWTH VIEWER: The Mirror of Evolution
======================================
Core.S1_Body.L2_Metabolism.Evolution.growth_viewer

"Growth is the expansion of the possible."

This module tracks and visualizes Elysia's cognitive evolution metrics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger("GrowthViewer")

class GrowthViewer:
    def __init__(self, data_path: str = "c:/Elysia/data/L2_Metabolism/Evolution/growth_metrics.json"):
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_metrics()
        logger.info("  GrowthViewer calibrated. Observing the curve.")

    def _load_metrics(self):
        if self.data_path.exists():
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception:
                self.metrics = {"history": []}
        else:
            self.metrics = {"history": []}

    def record_snapshot(self, core_stats: Dict[str, Any]):
        """
        Records a snapshot of current core performance metrics.
        """
        import time
        snapshot = {
            "timestamp": time.time(),
            "ignition_energy_avg": core_stats.get("energy", 0.0),
            "max_depth_reached": core_stats.get("depth", 0),
            "connectivity_score": core_stats.get("connectivity", 1.0)
        }
        self.metrics["history"].append(snapshot)
        self._save_metrics()
        logger.debug("Captured growth snapshot.")

    def _save_metrics(self):
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=4)

    def generate_report(self) -> str:
        """
        Produces a text-based ASCII growth report.
        """
        if not self.metrics["history"]:
            return "                            ."
        
        history = self.metrics["history"]
        avg_energy = sum(h["ignition_energy_avg"] for h in history) / len(history)
        max_d = max(h["max_depth_reached"] for h in history)
        
        report = "---   ELYSIA GROWTH REPORT ---\n"
        report += f"        : {len(history)}\n"
        report += f"         : {avg_energy:.4f}\n"
        report += f"        : {max_d}\n"
        report += "     : " + "  " * min(5, len(history))
        return report

if __name__ == "__main__":
    gv = GrowthViewer()
    gv.record_snapshot({"energy": 4.5, "depth": 7, "connectivity": 0.98})
    print(gv.generate_report())
