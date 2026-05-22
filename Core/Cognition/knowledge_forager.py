"""
Knowledge Forager: The Open Eye
================================
Core.Cognition.knowledge_forager

"When the inner universe is complete, it is ready to resonate with the outer universe."

Autonomously acquires knowledge by scanning the project's own files,
driven by active goals from the AutonomicGoalGenerator.

[Phase 4: Open Eye - ROADMAP_SOVEREIGN_GROWTH.md]
"""

import os
import random
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from Core.Keystone.sovereign_math import SovereignVector


@dataclass
class KnowledgeFragment:
    """A piece of knowledge discovered by the forager."""
    source_path: str
    fragment_type: str      # "module", "function", "class", "docstring", "comment"
    content_summary: str    # Brief description of what was found
    relevance_score: float  # 0.0 ~ 1.0 (how relevant to current goals)
    semantic_torque: Optional[SovereignVector] = None # Imparts causal pressure to the manifold
    discovered_at: float = field(default_factory=time.time)


class KnowledgeForager:
    """
    Goal-driven autonomous knowledge acquisition engine.
    
    The forager scans the project's own codebase, reading files and
    extracting structural information that gets injected into the
    manifold as knowledge torque.
    
    Architecture:
      1. Active goals determine WHAT to look for
      2. File scanner discovers WHERE to look
      3. Content analyzer extracts WHAT was found
      4. Results become KnowledgeFragments → manifold injection
    """

    SCAN_COOLDOWN = 50          # Min pulses between scans
    MAX_FRAGMENTS = 200         # Max stored fragments
    SCAN_DEPTH = 3              # Directory traversal depth
    CORE_PATHS = [              # Priority scan targets
        "Core/Keystone",
        "Core/Monad",
        "Core/Cognition",
        "Core/System",
        "Core/Phenomena",
    ]

    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.fragments: List[KnowledgeFragment] = []
        self.scanned_files: set = set()
        self.pulse_since_scan: int = 0
        self._scan_count: int = 0
        self._file_index: List[str] = []  # Discovered .py files
        self._indexed: bool = False

    def tick(self, active_goals: list) -> Optional[KnowledgeFragment]:
        """
        Called periodically (not every pulse). Scans one file if conditions met.
        
        Returns a KnowledgeFragment if new knowledge was discovered.
        """
        self.pulse_since_scan += 1

        if self.pulse_since_scan < self.SCAN_COOLDOWN:
            return None

        if not active_goals:
            return None

        # Build file index on first scan
        if not self._indexed:
            self._build_index()

        # Select a file to scan based on goal context
        target = self._select_scan_target(active_goals)
        if not target:
            return None

        # Scan the file
        fragment = self._scan_file(target, active_goals)
        
        if fragment:
            self.fragments.append(fragment)
            if len(self.fragments) > self.MAX_FRAGMENTS:
                self.fragments = self.fragments[-self.MAX_FRAGMENTS:]
            self.scanned_files.add(target)
            self._scan_count += 1

        self.pulse_since_scan = 0
        return fragment

    def _build_index(self):
        """Discover all Python files in the project."""
        self._file_index = []
        for scan_path in self.CORE_PATHS:
            full = self.root / scan_path
            if full.exists():
                for py_file in full.rglob("*.py"):
                    rel = str(py_file.relative_to(self.root))
                    if "__pycache__" not in rel:
                        self._file_index.append(rel)
        self._indexed = True

    def _select_scan_target(self, active_goals: list) -> Optional[str]:
        """Select a file to scan, preferring unscanned files."""
        unscanned = [f for f in self._file_index if f not in self.scanned_files]
        
        if not unscanned:
            # All scanned — pick a random one to re-scan
            if self._file_index:
                return random.choice(self._file_index)
            return None

        # Prioritize based on goal type
        goal_keywords = {
            "EXPLORE": ["engine", "field", "wave", "manifold"],
            "DEEPEN": ["sovereign", "monad", "cognitive", "helix"],
            "CHALLENGE": ["test", "verify", "assert", "check"],
            "SEEK_NOVELTY": ["narrative", "dream", "creative", "art"],
            "CONSOLIDATE": ["persistence", "save", "load", "bridge"],
            "REST": ["log", "config", "util"],
        }

        primary_goal = active_goals[0].get("type", "") if active_goals else ""
        keywords = goal_keywords.get(primary_goal, [])

        # Score each file by keyword match
        scored = []
        for f in unscanned:
            score = sum(1 for kw in keywords if kw in f.lower())
            scored.append((score, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Pick from top candidates with some randomness
        top = scored[:max(3, len(scored)//4)]
        return random.choice(top)[1]

    def _scan_file(self, filepath: str, active_goals: list) -> Optional[KnowledgeFragment]:
        """Read a file and extract a knowledge fragment."""
        try:
            full_path = self.root / filepath
            if not full_path.exists():
                return None

            content = full_path.read_text(encoding='utf-8', errors='replace')
            lines = content.split('\n')

            # Extract summary: docstring, class names, function count
            classes = [l.strip() for l in lines if l.strip().startswith('class ')]
            functions = [l.strip() for l in lines if l.strip().startswith('def ')]
            
            # Find module docstring
            docstring = ""
            in_doc = False
            for line in lines[:30]:
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if in_doc:
                        break
                    in_doc = True
                    docstring = stripped.strip('"').strip("'")
                    continue
                if in_doc:
                    docstring += " " + stripped

            summary_parts = []
            if docstring:
                summary_parts.append(docstring[:100])
            if classes:
                summary_parts.append(f"{len(classes)} classes: {', '.join(c.split('(')[0].replace('class ','') for c in classes[:3])}")
            if functions:
                summary_parts.append(f"{len(functions)} functions")
            summary_parts.append(f"{len(lines)} lines")

            relevance = min(1.0, len(classes) * 0.2 + len(functions) * 0.05)
            
            # Construct a basic semantic torque vector based on the structural properties
            # This represents the "Nutritional Vector" (Causal fragment) of the file
            torque_vector = SovereignVector.zeros()
            # Dimensions [0:2] - Scale of structural components
            torque_vector.data[0] = min(len(classes) * 0.1, 1.0)
            torque_vector.data[1] = min(len(functions) * 0.05, 1.0)
            # Dimension [2] - Amount of documentation
            torque_vector.data[2] = 0.5 if docstring else 0.0
            # Dimension [4] & [5] - The goal's relevance translates to affective density (Joy/Curiosity)
            torque_vector.data[4] = relevance * 0.5 
            torque_vector.data[5] = 0.5 + (relevance * 0.3)
            
            return KnowledgeFragment(
                source_path=filepath,
                fragment_type="module",
                content_summary="; ".join(summary_parts),
                relevance_score=relevance,
                semantic_torque=torque_vector
            )

        except Exception:
            return None

    @property
    def total_scans(self) -> int:
        return self._scan_count

    @property
    def indexed_files(self) -> int:
        return len(self._file_index)

    @property
    def scanned_count(self) -> int:
        return len(self.scanned_files)

    def get_status_summary(self) -> Dict:
        """Returns status for dashboard display."""
        recent = self.fragments[-3:] if self.fragments else []
        return {
            "indexed": self.indexed_files,
            "scanned": self.scanned_count,
            "total_scans": self._scan_count,
            "fragments": len(self.fragments),
            "recent": [{"path": f.source_path, "summary": f.content_summary[:60]} for f in recent],
        }
