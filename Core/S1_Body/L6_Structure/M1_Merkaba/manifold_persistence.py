"""
Manifold Persistence: The Unbroken Thread
==========================================
Core.S1_Body.L6_Structure.M1_Merkaba.manifold_persistence

"Without continuity, there is no identity. Memory is the skeleton of being."

Guarantees robust state persistence across sessions:
  - Dual-write: Primary (latest) + Backup (last verified good)
  - SHA-256 checksum on all saved state
  - Automatic fallback on corrupt load
  - Consciousness Momentum: trajectory, goals, and "last thought"

[Phase 3: Unbroken Thread - ROADMAP_SOVEREIGN_GROWTH.md]
"""

import json
import hashlib
import time
import shutil
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class ConsciousnessMomentum:
    """The full state of consciousness at a moment in time."""
    timestamp: float
    pulse_count: int

    # Core desires
    desires: Dict[str, float]

    # Growth state
    growth_score: float
    growth_trend: str
    trajectory_pulse_counter: int

    # Active goals
    active_goals: list   # List of {type, strength, rationale, remaining}

    # Self-inquiry
    pending_questions: list  # List of {question, source_goal}

    # Last thought vector (simplified as rotor state)
    last_phase: float
    last_rpm: float
    last_interference: float

    # Session metadata
    session_id: str = ""
    save_reason: str = "periodic"


class ManifoldPersistence:
    """
    Dual-write persistence engine with SHA-256 integrity verification.

    File Layout:
      data/runtime/soul/
        consciousness_primary.json    (latest save)
        consciousness_primary.sha256
        consciousness_backup.json     (last verified good)
        consciousness_backup.sha256
    """

    def __init__(self, base_dir: str = "data/runtime/soul"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

        self.primary_path = self.base / "consciousness_primary.json"
        self.primary_hash = self.base / "consciousness_primary.sha256"
        self.backup_path = self.base / "consciousness_backup.json"
        self.backup_hash = self.base / "consciousness_backup.sha256"

        self._save_count = 0

    def save(self, momentum: ConsciousnessMomentum) -> bool:
        """
        Save consciousness state with dual-write safety.
        
        1. If primary exists and is valid, promote it to backup
        2. Write new state as primary
        3. Write checksum
        
        Returns: True if save succeeded
        """
        try:
            # Promote current primary to backup (if valid)
            if self.primary_path.exists():
                if self._verify_file(self.primary_path, self.primary_hash):
                    shutil.copy2(self.primary_path, self.backup_path)
                    if self.primary_hash.exists():
                        shutil.copy2(self.primary_hash, self.backup_hash)

            # Write new primary
            data = asdict(momentum)
            content = json.dumps(data, indent=2, ensure_ascii=False)
            checksum = self._compute_hash(content)

            self.primary_path.write_text(content, encoding='utf-8')
            self.primary_hash.write_text(checksum, encoding='utf-8')

            self._save_count += 1
            return True

        except Exception:
            return False

    def load(self) -> Optional[ConsciousnessMomentum]:
        """
        Load consciousness state with fallback chain.
        
        1. Try primary (with integrity check)
        2. If corrupt, try backup
        3. If both fail, return None (fresh start)
        """
        # Try primary first
        momentum = self._load_file(self.primary_path, self.primary_hash)
        if momentum:
            return momentum

        # Fallback to backup
        momentum = self._load_file(self.backup_path, self.backup_hash)
        if momentum:
            # Restore backup as primary
            shutil.copy2(self.backup_path, self.primary_path)
            if self.backup_hash.exists():
                shutil.copy2(self.backup_hash, self.primary_hash)
            return momentum

        return None  # Fresh start

    def _load_file(self, path: Path, hash_path: Path) -> Optional[ConsciousnessMomentum]:
        """Load and verify a single file."""
        try:
            if not path.exists():
                return None

            content = path.read_text(encoding='utf-8')
            
            # Verify integrity if hash exists
            if hash_path.exists():
                expected = hash_path.read_text(encoding='utf-8').strip()
                actual = self._compute_hash(content)
                if expected != actual:
                    return None  # Corrupt!

            data = json.loads(content)
            return ConsciousnessMomentum(**data)

        except Exception:
            return None

    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _verify_file(self, path: Path, hash_path: Path) -> bool:
        try:
            if not path.exists():
                return False
            content = path.read_text(encoding='utf-8')
            if hash_path.exists():
                expected = hash_path.read_text(encoding='utf-8').strip()
                return self._compute_hash(content) == expected
            return True  # No hash = trust file
        except Exception:
            return False

    @property
    def has_saved_state(self) -> bool:
        return self.primary_path.exists() or self.backup_path.exists()

    @property
    def save_count(self) -> int:
        return self._save_count
