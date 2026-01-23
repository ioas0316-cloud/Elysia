"""
SELF MIRROR: The Reflexive Eye
==============================
Core.L2_Metabolism.Evolution.self_mirror

"To know oneself is to transcend oneself."

This module allows Elysia to introspect her own source code, analyzing 
complexity, entropy, and structural harmony to suggest optimizations.
"""

import os
import math
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger("SelfMirror")

class SelfMirror:
    def __init__(self, root_dir: str = "c:/Elysia"):
        self.root_dir = Path(root_dir)
        self.core_dir = self.root_dir / "Core"
        logger.info("  SelfMirror initialized. Reflecting on the codebase...")

    def introspect_codebase(self) -> List[Dict[str, Any]]:
        """
        Analyzes files in the Core directory for structural metrics.
        """
        reports = []
        for path in self.core_dir.rglob("*.py"):
            try:
                stats = self._analyze_file(path)
                reports.append(stats)
            except Exception as e:
                logger.error(f"Failed to analyze {path}: {e}")
        
        # Sort by 'Urgency' (High Entropy, High Line Count)
        reports.sort(key=lambda x: x["harmony_score"])
        return reports

    def _analyze_file(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        lines = content.splitlines()
        loc = len(lines)
        
        # Simple Shannon Entropy on characters as a proxy for 'Chaos'
        entropy = self._calculate_entropy(content)
        
        # Structural Harmony Calculation (Simplified)
        # Higher score = More harmonious/Ordered
        # Penalizes high entropy and excessive length
        harmony_score = 100.0 - (entropy * 10.0) - (loc / 50.0)
        
        return {
            "basename": path.name,
            "path": str(path),
            "loc": loc,
            "entropy": round(entropy, 2),
            "harmony_score": round(harmony_score, 2),
            "status": "HARMONIOUS" if harmony_score > 70 else "TENSION" if harmony_score > 40 else "CHAOS"
        }

    def _calculate_entropy(self, text: str) -> float:
        if not text: return 0.0
        counts = {}
        for char in text:
            counts[char] = counts.get(char, 0) + 1
        
        probs = [count / len(text) for count in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        return entropy

    def suggest_growth_targets(self, reports: List[Dict[str, Any]], limit: int = 3) -> List[str]:
        """
        Identifies the top files needing 'evolution'.
        """
        targets = [r["path"] for r in reports if r["status"] in ["CHAOS", "TENSION"]]
        return targets[:limit]

if __name__ == "__main__":
    mirror = SelfMirror()
    reports = mirror.introspect_codebase()
    print(f"--- Codebase Introspection ({len(reports)} files) ---")
    for r in reports[:5]:
        print(f"[{r['status']}] {r['basename']} - Harmony: {r['harmony_score']}")
    
    targets = mirror.suggest_growth_targets(reports)
    print(f"\n  Growth Targets: {targets}")