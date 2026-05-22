import json
import os
import time
from typing import Dict, Any

class ChronicleManager:
    def __init__(self, root_dir: str):
        self.chronicles_dir = os.path.join(root_dir, "World", "Chronicles")
        os.makedirs(self.chronicles_dir, exist_ok=True)
        print(f"📜 [CHRONICLE MANAGER] Linux Native File Node initialized at {self.chronicles_dir}")

    def record_event(self, event_type: str, data: Dict[str, Any], phase_coords: float):
        timestamp = time.time()
        filename = f"{int(timestamp)}_{event_type}_phase_{phase_coords:.4f}.json"
        filepath = os.path.join(self.chronicles_dir, filename)

        record = {
            "timestamp": timestamp,
            "event_type": event_type,
            "phase_coordinates": phase_coords,
            "data": data
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
