# [Genesis: 2025-12-02] Purified by Elysia
"""
FileSystemCortex

A minimal, safe file manipulation cortex confined to a root directory.
Provides list, read, and write operations with path normalization and
escape prevention. Intended for educational tasks and experience capture.

Design notes:
- All operations are sandboxed under `root_dir` (default: `data`).
- Attempts to escape the root (e.g., "..") are rejected.
- Text I/O only; callers can extend for binary needs later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class FSResult:
    ok: bool
    message: str
    path: Optional[str] = None
    data: Optional[str] = None


class FileSystemCortex:
    """
    Purpose: Safely manipulate files as part of learning experiences.
    Role: Agent Sophia component for controlled file I/O.
    """

    def __init__(self, root_dir: str | Path = "data"):
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel_path: str | Path) -> Path | None:
        p = (self.root / Path(rel_path)).resolve()
        # Prevent path traversal above root
        try:
            p.relative_to(self.root)
        except Exception:
            return None
        return p

    def list_files(self, rel_dir: str | Path = ".") -> FSResult:
        target = self._resolve(rel_dir)
        if target is None:
            return FSResult(False, "Access denied: outside root.")
        if not target.exists():
            return FSResult(False, f"Directory not found: {rel_dir}")
        if not target.is_dir():
            return FSResult(False, f"Not a directory: {rel_dir}")
        files: List[str] = []
        for p in sorted(target.rglob("*")):
            if p.is_file():
                files.append(str(p.relative_to(self.root)))
        return FSResult(True, "OK", data="\n".join(files))

    def read_file(self, rel_path: str | Path, encoding: str = "utf-8") -> FSResult:
        target = self._resolve(rel_path)
        if target is None:
            return FSResult(False, "Access denied: outside root.")
        if not target.exists() or not target.is_file():
            return FSResult(False, f"File not found: {rel_path}")
        try:
            text = target.read_text(encoding=encoding)
            return FSResult(True, "OK", path=str(target), data=text)
        except Exception as e:
            return FSResult(False, f"Read error: {e}")

    def write_file(self, rel_path: str | Path, content: str, encoding: str = "utf-8") -> FSResult:
        target = self._resolve(rel_path)
        if target is None:
            return FSResult(False, "Access denied: outside root.")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding=encoding)
            return FSResult(True, "OK", path=str(target))
        except Exception as e:
            return FSResult(False, f"Write error: {e}")
