"""
[Project Elysia] PLASTICITY_LOG (Phase-Causal Integration)
==========================================================
"Every thought leaves a scar in the field."
Records structural refinements (LTP/LTD) and Lightning Path formation.
"""

import os
import time
import json
from datetime import datetime

LOG_PATH = r"c:\Elysia\data\Logs\plasticity_log.jsonl"

class PlasticityLogger:
    def __init__(self):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        
    def log_event(self, event_type, details, resonance_gain):
        """
        Records a structural change event.
        event_type: 'LTP' (Strengthening), 'LTD' (Weakening), 'PATH_FORMATION' (Lightning Path)
        details: dict containing node_ids, layer_indices, etc.
        resonance_gain: float improvement in system alignment.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details,
            "resonance_gain": resonance_gain
        }
        
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    def get_summary(self, limit=10):
        """Returns the last N events."""
        if not os.path.exists(LOG_PATH):
            return []
        
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return [json.loads(l) for l in lines[-limit:]]

# Singleton instance
plasticity_logger = PlasticityLogger()

if __name__ == "__main__":
    # Test logging
    print("📝 Testing Plasticity Logger...")
    plasticity_logger.log_event("LTP", {"node": "Root", "change": "Resonance Increase"}, 0.05)
    print(f"✅ Event logged to {LOG_PATH}")
