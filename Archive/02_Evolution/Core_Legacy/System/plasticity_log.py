import os
import time
from datetime import datetime

LOG_PATH = "logs/S1_Body/Tools/Scripts/plasticity.log"

class PlasticityLogger:
    def __init__(self):
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        except Exception:
            pass # Fallback if path is invalid

    def log_plasticity(self, event: str):
        try:
            with open(LOG_PATH, "a") as f:
                timestamp = datetime.now().isoformat()
                f.write(f"[{timestamp}] {event}\n")
        except Exception:
            pass

    def log_event(self, event: str):
        """[Legacy Wrapper] Compatibility for old calls."""
        self.log_plasticity(event)

plasticity_logger = PlasticityLogger()
