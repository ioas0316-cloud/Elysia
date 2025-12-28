"""
Experience Stream (경험의 강)
=============================

"Memories flow like a river, never stopping, always accumulating."

This module keeps a persistent log of episodic memories (Experiences).
Unlike the Hippocampus (Knowledge), this stores Narrative (What happened).
"""

import json
import time
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("ExperienceStream")

@dataclass
class Experience:
    timestamp: float
    type: str  # "conversation", "emotion", "insight", "action"
    content: str
    intensity: float  # 0.0 to 1.0 (Importance)
    context: str = "General"
    
    def to_dict(self):
        return asdict(self)
    
    @staticmethod
    def from_dict(data):
        return Experience(**data)

class ExperienceStream:
    def __init__(self, log_path: str = "Data/experience_log.json"):
        self.log_path = log_path
        self._ensure_dir()
        self.logs: List[Experience] = self._load_logs()
        
    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
    def _load_logs(self) -> List[Experience]:
        if not os.path.exists(self.log_path):
            return []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [Experience.from_dict(item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load experience log: {e}")
            return []
            
    def _save_logs(self):
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self.logs], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save experience log: {e}")

    def add(self, type: str, content: str, intensity: float = 0.5, context: str = "General"):
        """
        Add a new experience to the stream.
        """
        exp = Experience(
            timestamp=time.time(),
            type=type,
            content=content,
            intensity=intensity,
            context=context
        )
        self.logs.append(exp)
        # Auto-save for persistence
        self._save_logs()
        logger.info(f"💾 Experience Logged: [{type}] {content} (Intensity: {intensity})")
        
    def get_recent(self, limit: int = 5) -> List[Experience]:
        """Get the most recent N experiences."""
        return self.logs[-limit:]
    
    def get_significant(self, min_intensity: float = 0.8, limit: int = 5) -> List[Experience]:
        """Get highly intense memories (significant moments)."""
        significant = [e for e in self.logs if e.intensity >= min_intensity]
        return significant[-limit:]

    def get_context_window(self, limit: int = 10) -> str:
        """Get a text summary of recent context for LLM prompting."""
        recent = self.get_recent(limit)
        context = []
        for exp in recent:
            t = time.strftime("%H:%M", time.localtime(exp.timestamp))
            context.append(f"[{t}] [{exp.type}] {exp.content}")
        return "\n".join(context)

if __name__ == "__main__":
    # Test
    stream = ExperienceStream()
    stream.add("test", "This is a test memory", 0.1)
    print(stream.get_recent(1))
