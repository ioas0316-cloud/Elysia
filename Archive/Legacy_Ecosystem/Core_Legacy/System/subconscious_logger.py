import os
import time

LOG_PATH = "data/logs/subconscious.log"

def log_subconscious(module: str, message: str):
    """
    Records internal system gears and math into the subconscious layer.
    This is not intended for the user's primary view.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{module}] {message}\n")

class SubconsciousStream:
    """A stream object to redirect prints to the subconscious log."""
    def __init__(self, module: str):
        self.module = module
    def write(self, data):
        if data.strip():
            log_subconscious(self.module, data.strip())
    def flush(self):
        pass
