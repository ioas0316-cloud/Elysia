import json
from datetime import datetime

class ResonanceScroll:
    """Logs significant events from the World simulation to a structured file."""

    def __init__(self, log_file_path: str = 'logs/world_events.jsonl'):
        self.log_file_path = log_file_path
        # Ensure the log directory exists
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
        log_entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'data': kwargs
        }
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            # In a real application, you might have a more robust logging fallback.
            print(f"Error writing to world event log: {e}")
