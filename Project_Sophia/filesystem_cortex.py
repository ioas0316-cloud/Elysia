from __future__ import annotations

import fnmatch
import hashlib
import json
import mimetypes
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union


# Exceptions
class FsError(Exception):
    pass


class FsPermissionError(FsError):
    pass


class FsPathError(FsError):
    pass


class FsIOError(FsError):
    pass


class FsIndexError(FsError):
    pass


@dataclass
class FileMeta:
    """Metadata describing a file.

    Purpose: Provide lightweight, serializable facts about a file for indexing/search.
    Role: Returned from scan/index/write/move; used by search results.
    """

    path: str
    name: str
    size: int
    mtime: float
    mime: Optional[str] = None
    ext: Optional[str] = None
    sha1: Optional[str] = None


@dataclass
class Document:
    """Content wrapper with metadata.

    Purpose: Represent file content with its meta and encoding.
    Role: Returned by read(); used by downstream cortex components.
    """

    meta: FileMeta
    content: Union[bytes, str]
    encoding: Optional[str] = None


@dataclass
class FsConfig:
    """Runtime configuration for FileSystemCortex.

    - ignore_globs: patterns to exclude from scan/index operations
    - allowed_exts: if provided, only allow these extensions (e.g., {".py", ".md"})
    - max_file_mb: max file size allowed for read/write/index in megabytes
    - hash_algo: if set (e.g., "sha1"), compute content hash for index/write
    - telemetry_namespace: label used in emitted telemetry events
    """

    ignore_globs: List[str] = field(default_factory=lambda: ["**/.git/**", "**/__pycache__/**"])
    allowed_exts: Optional[Iterable[str]] = None
    max_file_mb: int = 16
    hash_algo: Optional[str] = None
    telemetry_namespace: str = "Project_Sophia.FileSystemCortex"


class PermissionPolicy:
    """Enforces sandbox and write protections.

    Purpose: Defend against path traversal and unwanted mutations.
    Role: Centralized checks used by FileSystemCortex.
    """

    def __init__(self, root: Path, read_only: bool) -> None:
        self.root = root.resolve()
        self.read_only = read_only

    def ensure_within_root(self, p: Path) -> Path:
        try:
            rp = p.resolve()
        except FileNotFoundError:
            # resolve(strict=False) equivalent for compatibility
            rp = (self.root / p).resolve()
        if not _is_relative_to(rp, self.root):
            raise FsPermissionError(f"Path escapes sandbox: {rp}")
        return rp

    def ensure_writable(self) -> None:
        if self.read_only:
            raise FsPermissionError("Cortex is read-only; mutation is disabled")


class FsIndexer:
    """In-memory index of files for quick lookup/search.

    Purpose: Provide minimal name/ext/size/mtime based search for MVP.
    Role: Built by FileSystemCortex.index(); queried by search().
    """

    def __init__(self) -> None:
        self._by_path: Dict[str, FileMeta] = {}

    def add(self, meta: FileMeta) -> None:
        self._by_path[meta.path] = meta

    def clear(self) -> None:
        self._by_path.clear()

    def items(self) -> Iterable[Tuple[str, FileMeta]]:
        return self._by_path.items()

    def get(self, path: str) -> Optional[FileMeta]:
        return self._by_path.get(path)


class Telemetry:
    """Lightweight telemetry sink wrapper.

    Purpose: Emit operation metrics without hard dependency on infra/telemetry.
    Role: Used internally; can be swapped with a real collector.
    """

    def __init__(self, namespace: str, emitter: Optional[Callable[[Dict[str, Union[str, int, float, None]]], None]] = None) -> None:
        self.namespace = namespace
        self._emit = emitter or (lambda event: None)

    @contextmanager
    def span(self, op: str, path: str) -> Iterator[Dict[str, Union[str, int, float, None]]]:
        start = time.perf_counter()
        event: Dict[str, Union[str, int, float, None]] = {
            "ns": self.namespace,
            "op": op,
            "path": path,
            "bytes": 0,
            "status": "ok",
            "latency_ms": 0.0,
            "error": None,
        }
        try:
            yield event
        except Exception as exc:  # noqa: BLE001
            event["status"] = "error"
            event["error"] = type(exc).__name__
            raise
        finally:
            event["latency_ms"] = (time.perf_counter() - start) * 1000.0
            try:
                self._emit(event)
            except Exception:
                # Telemetry must never break the operation path
                pass


class FileSystemCortex:
    """File system facade for safe, observable I/O.

    Purpose: Provide read/write/delete/move/scan/index/search with sandboxing and telemetry.
    Role: Core I/O provider for Project Sophia components.
    """

    def __init__(
        self,
        root: Union[str, Path],
        read_only: bool = False,
        config: Optional[FsConfig] = None,
        telemetry_emitter: Optional[Callable[[Dict[str, Union[str, int, float, None]]], None]] = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.exists() or not self.root.is_dir():
            raise FsPathError(f"Root does not exist or not a directory: {self.root}")
        self.config = config or FsConfig()
        self.policy = PermissionPolicy(self.root, read_only)
        self.telemetry = Telemetry(self.config.telemetry_namespace, telemetry_emitter)
        self.indexer = FsIndexer()

    # --------------- Public API ---------------
    def scan(self, rel_path: str = ".", depth: int = 1) -> List[FileMeta]:
        """Scan for files under rel_path up to depth.

        - Applies ignore globs and allowed_exts.
        - Skips files exceeding max_file_mb.
        """
        base = self._normalize(rel_path)
        metas: List[FileMeta] = []
        with self.telemetry.span("scan", str(base)) as evt:
            for dirpath, dirnames, filenames in os.walk(base):
                rel_depth = _depth_from(base, Path(dirpath))
                if rel_depth > depth:
                    # prune deeper traversal
                    dirnames[:] = []
                    continue
                # apply ignore to directories in-place to prune
                dirnames[:] = [d for d in dirnames if not self._is_ignored(Path(dirpath) / d)]
                for name in filenames:
                    p = Path(dirpath) / name
                    if self._is_ignored(p):
                        continue
                    if not self._is_allowed_ext(p):
                        continue
                    try:
                        meta = self._meta_for(p)
                    except FsError:
                        continue
                    if meta.size > self._max_bytes():
                        continue
                    metas.append(meta)
            evt["bytes"] = sum(m.size for m in metas)
        return metas

    def read(self, rel_path: str, as_text: bool = False, encoding: str = "utf-8") -> Document:
        path = self._normalize(rel_path)
        with self.telemetry.span("read", str(path)) as evt:
            if not path.exists() or not path.is_file():
                raise FsPathError(f"Not a file: {path}")
            if not self._is_allowed_ext(path):
                raise FsPermissionError(f"Extension not allowed: {path.suffix}")
            size = path.stat().st_size
            if size > self._max_bytes():
                raise FsPermissionError("File exceeds max size")
            if as_text:
                data = path.read_text(encoding=encoding)
                content: Union[str, bytes] = data
            else:
                content = path.read_bytes()
            evt["bytes"] = len(content) if isinstance(content, (bytes, bytearray)) else len(content.encode(encoding, errors="ignore"))
        return Document(meta=self._meta_for(path), content=content, encoding=(encoding if as_text else None))

    def write(
        self,
        rel_path: str,
        data: Union[bytes, str],
        overwrite: bool = False,
        encoding: str = "utf-8",
    ) -> FileMeta:
        self.policy.ensure_writable()
        path = self._normalize(rel_path, must_exist_parent=True)
        with self.telemetry.span("write", str(path)) as evt:
            if path.exists() and not overwrite:
                raise FsIOError(f"File exists: {path}")
            if not self._is_allowed_ext(path):
                raise FsPermissionError(f"Extension not allowed: {path.suffix}")
            parent = path.parent
            parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, str):
                encoded = data.encode(encoding)
                path.write_text(data, encoding=encoding)
            else:
                encoded = data
                path.write_bytes(data)
            if len(encoded) > self._max_bytes():
                # cleanup to avoid leaving oversized file
                try:
                    path.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
                raise FsPermissionError("Write exceeds max size")
            evt["bytes"] = len(encoded)
        return self._meta_for(path)

    def delete(self, rel_path: str) -> None:
        self.policy.ensure_writable()
        path = self._normalize(rel_path)
        with self.telemetry.span("delete", str(path)):
            if path.is_dir():
                raise FsIOError("Refusing to delete directories in MVP")
            if path.exists():
                path.unlink()

    def move(self, src_rel: str, dst_rel: str, overwrite: bool = False) -> FileMeta:
        self.policy.ensure_writable()
        src = self._normalize(src_rel)
        dst = self._normalize(dst_rel, must_exist_parent=True)
        with self.telemetry.span("move", f"{src} -> {dst}"):
            if not src.exists() or not src.is_file():
                raise FsPathError(f"Source not a file: {src}")
            if not self._is_allowed_ext(dst):
                raise FsPermissionError(f"Extension not allowed: {dst.suffix}")
            if dst.exists():
                if overwrite:
                    dst.unlink()
                else:
                    raise FsIOError(f"Destination exists: {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        return self._meta_for(dst)

    def index(self, rel_paths: Optional[List[str]] = None) -> None:
        base_paths = [self._normalize(p) for p in (rel_paths or ["."])]
        with self.telemetry.span("index", ",".join(map(str, base_paths))):
            self.indexer.clear()
            for base in base_paths:
                for meta in self.scan(str(base.relative_to(self.root)), depth=32):
                    self.indexer.add(meta)

    def search(self, query: str, limit: int = 50) -> List[FileMeta]:
        """Very simple search: supports name: and ext: tokens, otherwise substring by name.
        Example: "name:report ext:.md"
        """
        name_q: Optional[str] = None
        exts: List[str] = []
        tokens = query.split()
        for t in tokens:
            if t.startswith("name:"):
                name_q = t[len("name:") :].lower()
            elif t.startswith("ext:"):
                exts.append(t[len("ext:") :].lower())
        results: List[FileMeta] = []
        for _p, meta in self.indexer.items():
            if name_q and name_q not in meta.name.lower():
                continue
            if exts and (meta.ext or "").lower() not in exts:
                continue
            results.append(meta)
            if len(results) >= limit:
                break
        return results

    # --------------- Persistence ---------------
    def save_index(self, rel_path: str = "index.json") -> Path:
        """Persist the in-memory index to a JSON file under root.

        Purpose: Fast startup by reloading index later.
        Role: Complements index(); not subject to allowed_ext restrictions.
        """
        self.policy.ensure_writable()
        out_path = self._normalize(rel_path, must_exist_parent=True)
        with self.telemetry.span("save_index", str(out_path)) as evt:
            data = [meta.__dict__ for _, meta in self.indexer.items()]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(out_path)
            evt["bytes"] = out_path.stat().st_size
        return out_path

    def load_index(self, rel_path: str = "index.json") -> int:
        """Load index from a JSON file under root. Returns number of entries loaded."""
        in_path = self._normalize(rel_path)
        with self.telemetry.span("load_index", str(in_path)):
            if not in_path.exists():
                raise FsPathError(f"Index file not found: {in_path}")
            raw = in_path.read_text(encoding="utf-8")
            items = json.loads(raw)
            if not isinstance(items, list):
                raise FsIndexError("Invalid index format")
            self.indexer.clear()
            count = 0
            for obj in items:
                if not isinstance(obj, dict):
                    continue
                meta = FileMeta(
                    path=obj.get("path"),
                    name=obj.get("name"),
                    size=int(obj.get("size", 0)),
                    mtime=float(obj.get("mtime", 0.0)),
                    mime=obj.get("mime"),
                    ext=obj.get("ext"),
                    sha1=obj.get("sha1"),
                )
                self.indexer.add(meta)
                count += 1
        return count

    # --------------- Watch (polling) ---------------
    def watch(
        self,
        rel_path: str = ".",
        interval: float = 1.0,
        stop_after: Optional[float] = None,
    ) -> Iterator["FsEvent"]:
        """Yield file change events under rel_path by polling.

        - Emits 'created', 'modified', 'deleted' events (moves appear as delete+create).
        - Applies ignore/allowed_ext filters.
        - stop_after: seconds to run before stopping; None for infinite.
        """
        base = self._normalize(rel_path)
        deadline = (time.time() + stop_after) if stop_after else None
        prev = self._snapshot(base)
        while True:
            time.sleep(max(0.01, interval))
            curr = self._snapshot(base)
            events = self._diff_snapshots(prev, curr)
            for ev in events:
                yield ev
            prev = curr
            if deadline and time.time() >= deadline:
                break

    def watch_once(self, rel_path: str = ".", prev_state: Optional[Dict[str, Tuple[int, float]]] = None) -> Tuple[List["FsEvent"], Dict[str, Tuple[int, float]]]:
        """Perform a single poll step and return (events, state). Useful for tests."""
        base = self._normalize(rel_path)
        curr = self._snapshot(base)
        events = self._diff_snapshots(prev_state or {}, curr)
        return events, curr

    # --------------- Internal helpers ---------------
    def _normalize(self, rel_path: str, must_exist_parent: bool = False) -> Path:
        if os.path.isabs(rel_path):
            # absolute paths are reinterpreted relative to root to avoid leaks
            rel_path = os.path.relpath(rel_path, start=str(self.root))
        p = (self.root / rel_path)
        rp = self.policy.ensure_within_root(p)
        if must_exist_parent and not rp.parent.exists():
            # parent will be created by writer; existence not required here
            pass
        return rp

    def _is_ignored(self, p: Path) -> bool:
        rel = str(p.absolute()).replace("\\", "/")
        for pat in self.config.ignore_globs:
            if fnmatch.fnmatch(rel, pat):
                return True
        return False

    def _is_allowed_ext(self, p: Path) -> bool:
        if not self.config.allowed_exts:
            return True
        ext = p.suffix
        return ext in set(self.config.allowed_exts)

    def _max_bytes(self) -> int:
        return self.config.max_file_mb * 1024 * 1024

    def _meta_for(self, p: Path) -> FileMeta:
        st = p.stat()
        mime, _ = mimetypes.guess_type(p.name)
        ext = p.suffix or None
        sha1: Optional[str] = None
        if self.config.hash_algo and st.st_size <= self._max_bytes():
            if self.config.hash_algo.lower() == "sha1":
                sha1 = _hash_file(p, hashlib.sha1())
        return FileMeta(
            path=str(p.relative_to(self.root)),
            name=p.name,
            size=st.st_size,
            mtime=st.st_mtime,
            mime=mime,
            ext=ext,
            sha1=sha1,
        )

    def _snapshot(self, base: Path) -> Dict[str, Tuple[int, float]]:
        """Return mapping of rel path -> (size, mtime) for allowed files under base."""
        snap: Dict[str, Tuple[int, float]] = {}
        for dirpath, dirnames, filenames in os.walk(base):
            # prune ignored dirs
            dirnames[:] = [d for d in dirnames if not self._is_ignored(Path(dirpath) / d)]
            for name in filenames:
                p = Path(dirpath) / name
                if self._is_ignored(p) or not self._is_allowed_ext(p):
                    continue
                try:
                    st = p.stat()
                except FileNotFoundError:
                    continue
                if st.st_size > self._max_bytes():
                    continue
                rel = str(p.relative_to(self.root))
                snap[rel] = (int(st.st_size), float(st.st_mtime))
        return snap

    def _diff_snapshots(self, prev: Dict[str, Tuple[int, float]], curr: Dict[str, Tuple[int, float]]) -> List["FsEvent"]:
        events: List[FsEvent] = []
        now = time.time()
        prev_keys = set(prev.keys())
        curr_keys = set(curr.keys())
        for created in sorted(curr_keys - prev_keys):
            events.append(FsEvent(op="created", path=created, when=now))
        for deleted in sorted(prev_keys - curr_keys):
            events.append(FsEvent(op="deleted", path=deleted, when=now))
        for common in sorted(prev_keys & curr_keys):
            if prev[common] != curr[common]:
                events.append(FsEvent(op="modified", path=common, when=now))
        return events


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _hash_file(p: Path, hasher: "hashlib._Hash", chunk: int = 1024 * 1024) -> str:
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            hasher.update(b)
    return hasher.hexdigest()


def _depth_from(base: Path, current: Path) -> int:
    try:
        rel = current.relative_to(base)
    except Exception:
        return 0
    if str(rel) == ".":
        return 0
    return len(str(rel).split(os.sep))


@dataclass
class FsEvent:
    op: str  # 'created' | 'modified' | 'deleted'
    path: str  # relative to cortex root
    when: float
