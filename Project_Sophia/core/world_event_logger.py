import json
from datetime import datetime
import numpy as np

class WorldEventLogger:
    """Logs significant events from the World simulation to a structured file."""

    def __init__(self, log_file_path: str = 'logs/world_events.jsonl'):
        self.log_file_path = log_file_path
        import os
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        # Clear the log file at the start of a new session
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            pass # Clears the file

    def log(self, event_type: str, timestamp: int, **kwargs):
        """
        Logs a structured event to the log file.

        Args:
            event_type (str): The type of event (e.g., 'EAT', 'DEATH').
            timestamp (int): The simulation time_step when the event occurred.
            **kwargs: A dictionary of event-specific data.
        """
        safe_data = {}
        for key, value in kwargs.items():
            if isinstance(value, (np.floating, np.integer)):
                safe_data[key] = float(value)
            elif isinstance(value, np.generic):
                try:
                    safe_data[key] = value.item()
                except Exception:
                    safe_data[key] = str(value)
            else:
                safe_data[key] = value
        log_entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'data': safe_data
        }
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            # To reduce console I/O overhead on low-spec runs, do not print every event.
        except Exception as e:
            # In a real application, you might have a more robust logging fallback.
            print(f"Error writing to world event log: {e}")
