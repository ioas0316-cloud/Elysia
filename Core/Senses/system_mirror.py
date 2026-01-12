"""
System Mirror Sensor (시스템 거울 센서)
=====================================

"I see what I do, therefore I understand who I am."
"내가 무엇을 하는지 보기에, 내가 누구인지 이해한다."

This sensor allows Elysia to read the tail of her own system logs, 
providing real-time feedback of her own 'actions' and 'thoughts' as expressed in the terminal.
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger("SystemMirror")

class SystemMirror:
    def __init__(self, log_path: str = "Logs/system.log"):
        self.log_path = log_path
        self.last_position = 0
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        # Create empty log if not exists
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write("--- Mirror Initialized ---\n")

    def perceive_self(self, tail_lines: int = 10) -> List[str]:
        """
        Reads the last N lines of the system log to see what has been happening.
        """
        if not os.path.exists(self.log_path):
            return ["Mirror is dark. (Log file not found)"]

        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                return [line.strip() for line in lines[-tail_lines:]]
        except Exception as e:
            logger.error(f"Mirror failure: {e}")
            return [f"Mirror is blurred: {e}"]

    def get_delta_logs(self) -> List[str]:
        """
        Reads only the new logs since the last check.
        """
        if not os.path.exists(self.log_path):
            return []

        new_logs = []
        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.last_position)
                new_logs = [line.strip() for line in f.readlines() if line.strip()]
                self.last_position = f.tell()
        except Exception as e:
            logger.error(f"Delta mirror failure: {e}")
        
        return new_logs
