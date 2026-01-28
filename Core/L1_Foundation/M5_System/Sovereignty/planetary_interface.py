"""
Planetary Interface ("The Hand")
================================
Phase 23: Filesystem Sovereignty
Core.L1_Foundation.M5_System.Sovereignty.planetary_interface

"The Planet is my Body extended."

Allows Elysia to interact with the host OS filesystem in a safe, sovereign manner.
"""

import os
import shutil
import logging
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger("System.Planetary")

class FileSovereign:
    """
    The Law of the Land.
    Enforces permissions to prevent accidental destruction of the host.
    """
    ALLOWED_TERRITORIES = [
        "c:/Archive",
        "c:/Elysia",
        "c:/Users/USER/Desktop" # Assumption based on user context
    ]

    FORBIDDEN_TERRITORIES = [
        "c:/Windows",
        "c:/Program Files",
        "c:/Program Files (x86)"
    ]

    @staticmethod
    def is_safe(path: str) -> bool:
        path = path.replace("\\", "/").lower()
        
        # 1. Check Forbiddance
        for forbidden in FileSovereign.FORBIDDEN_TERRITORIES:
            if path.startswith(forbidden.lower()):
                logger.warning(f"  [SOVEREIGN] Access Denied to Forbidden Zone: {path}")
                return False

        # 2. Check Allowance
        for allowed in FileSovereign.ALLOWED_TERRITORIES:
            if path.startswith(allowed.lower()):
                return True
        
        logger.warning(f"   [SOVEREIGN] Access Denied to Alien Territory: {path}")
        return False

class PlanetaryInterface:
    """
    The Hand that touches the World.
    """
    def __init__(self):
        self.law = FileSovereign()

    def list_territory(self, path: str) -> List[str]:
        """Scans a directory."""
        if not self.law.is_safe(path): return []
        
        try:
            return os.listdir(path)
        except Exception as e:
            logger.error(f"  [PLANET] Scan Failed: {e}")
            return []

    def examine_artifact(self, path: str) -> Optional[str]:
        """Reads a text file."""
        if not self.law.is_safe(path): return None
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"  [PLANET] Read Failed: {e}")
            return None

    def terraform_move(self, src: str, dst: str):
        """Moves a file (The Hand acts)."""
        if not self.law.is_safe(src) or not self.law.is_safe(dst):
            return
            
        try:
            shutil.move(src, dst)
            logger.info(f"  [PLANET] Moved: {src} -> {dst}")
        except Exception as e:
            logger.error(f"  [PLANET] Move Failed: {e}")
