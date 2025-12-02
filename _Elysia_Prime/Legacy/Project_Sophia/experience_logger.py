# [Genesis: 2025-12-02] Purified by Elysia
"""
Centralized Experience Logger for Elysia

This module provides a standalone function to log structured experience data.
Any part of Elysia's "mind" can import and use this to record sensations,
actions, thoughts, or feedbacks into the central experience.log.
"""
import json
import os
from datetime import datetime

EXPERIENCE_LOG = 'experience.log'

def log_experience(source: str, event_type: str, data: dict):
    """Logs a structured experience event for future causal learning."""
    try:
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "source": source,
            "type": event_type,
            "data": data
        }
        # Ensure the log file exists at the project root
        log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', EXPERIENCE_LOG))

        with open(log_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # If logging fails, print to stderr. This is a critical issue.
        print(f"CRITICAL LOGGING FAILED: {e}", file=sys.stderr)