import io
import os
from pathlib import Path

import pytest

from Project_Sophia.filesystem_cortex import (
    Document,
    FileMeta,
    FileSystemCortex,
    FsConfig,
    FsIOError,
    FsPathError,
    FsPermissionError,
    FsEvent,
)


def _tmp(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def test_basic_write_read(tmp_path: Path):
    cfg = FsConfig(allowed_exts={".txt"})
    events = []
    cortex = FileSystemCortex(tmp_path, read_only=False, config=cfg, telemetry_emitter=events.append)
    meta = cortex.write("a/b.txt", "hello", overwrite=False)
    assert isinstance(meta, FileMeta)
    doc = cortex.read("a/b.txt", as_text=True)
    assert isinstance(doc, Document)
    assert doc.content == "hello"
    assert any(e.get("op") == "write" for e in events)
    assert any(e.get("op") == "read" for e in events)


def test_overwrite_and_move(tmp_path: Path):
    cfg = FsConfig(allowed_exts={".txt"})
    cortex = FileSystemCortex(tmp_path, read_only=False, config=cfg)
    cortex.write("x.txt", "v1")
    with pytest.raises(FsIOError):
        cortex.write("x.txt", "v2")
    cortex.write("x.txt", "v2", overwrite=True)
    meta = cortex.move("x.txt", "y.txt")
    assert meta.name == "y.txt"
    assert Path(tmp_path / "y.txt").exists()


def test_delete(tmp_path: Path):
    cfg = FsConfig(allowed_exts={".txt"})
    cortex = FileSystemCortex(tmp_path, read_only=False, config=cfg)
    cortex.write("d.txt", "del")
    cortex.delete("d.txt")
    with pytest.raises(FsPathError):
        cortex.read("d.txt")


def test_readonly_blocks_mutation(tmp_path: Path):
    cfg = FsConfig(allowed_exts={".txt"})
    cortex = FileSystemCortex(tmp_path, read_only=True, config=cfg)
    with pytest.raises(FsPermissionError):
        cortex.write("a.txt", "nope")
    with pytest.raises(FsPermissionError):
        cortex.delete("a.txt")


def test_sandbox_blocks_escape(tmp_path: Path):
    cfg = FsConfig(allowed_exts={".txt"})
    cortex = FileSystemCortex(tmp_path, config=cfg)
    outside = Path(tmp_path).parent / "escape.txt"
    outside.write_text("out")
    with pytest.raises(FsPermissionError):
        cortex.read("../escape.txt")


def test_absolute_path_escape_blocked(tmp_path: Path):
    cfg = FsConfig(allowed_exts={".txt"})
    cortex = FileSystemCortex(tmp_path, config=cfg)
    outside = Path(tmp_path).parent / "abs_escape.txt"
    outside.write_text("out")
    with pytest.raises(FsPermissionError):
        cortex.read(str(outside))


def test_symlink_escape_blocked(tmp_path: Path):
    if os.name == 'nt' or not hasattr(os, 'symlink'):
        pytest.skip("symlink requires non-Windows or admin rights")
    cfg = FsConfig(allowed_exts={".txt"})
    cortex = FileSystemCortex(tmp_path, config=cfg)
    outside = Path(tmp_path).parent / "real.txt"
    outside.write_text("out")
    link = tmp_path / "link.txt"
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("symlink not permitted in environment")
    with pytest.raises(FsPermissionError):
        cortex.read("link.txt")


def test_ignore_and_ext_filter(tmp_path: Path):
    cfg = FsConfig(allowed_exts={".md"}, ignore_globs=["**/.git/**"])  # allow only .md
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/ignored.txt").write_text("x")
    (tmp_path / "doc.md").write_text("md")
    (tmp_path / "note.txt").write_text("txt")
    cortex = FileSystemCortex(tmp_path, config=cfg)
    metas = cortex.scan(depth=1)
    names = {m.name for m in metas}
    assert "doc.md" in names
    assert "note.txt" not in names


def test_max_file_size(tmp_path: Path):
    # max 1 KB
    cfg = FsConfig(allowed_exts={".bin"}, max_file_mb=0)
    cortex = FileSystemCortex(tmp_path, read_only=False, config=cfg)
    big = os.urandom(2048)
    with pytest.raises(FsPermissionError):
        cortex.write("big.bin", big, overwrite=True)


def test_index_and_search(tmp_path: Path):
    cfg = FsConfig(allowed_exts={".txt", ".md"})
    cortex = FileSystemCortex(tmp_path, read_only=False, config=cfg)
    cortex.write("a/report.txt", "x", overwrite=True)
    cortex.write("a/readme.md", "x", overwrite=True)
    cortex.index()
    res = cortex.search("name:report ext:.txt")
    assert len(res) == 1 and res[0].name == "report.txt"


def test_index_persistence_json(tmp_path: Path):
    cfg = FsConfig(allowed_exts={".txt", ".json"})
    cortex = FileSystemCortex(tmp_path, read_only=False, config=cfg)
    cortex.write("a/file1.txt", "x", overwrite=True)
    cortex.write("a/file2.txt", "y", overwrite=True)
    cortex.index()
    out = cortex.save_index("idx/index.json")
    assert out.exists()
    # clear and reload
    cortex.indexer.clear()
    count = cortex.load_index("idx/index.json")
    assert count >= 2
    # search should still work using loaded index
    res = cortex.search("name:file1 ext:.txt")
    assert any(m.name == "file1.txt" for m in res)


def test_watch_once_detects_changes(tmp_path: Path):
    cfg = FsConfig(allowed_exts={".txt"})
    cortex = FileSystemCortex(tmp_path, read_only=False, config=cfg)
    # baseline
    events, state = cortex.watch_once(".")
    assert isinstance(events, list)
    # create
    cortex.write("w.txt", "1", overwrite=True)
    events, state = cortex.watch_once(".", prev_state=state)
    assert any(isinstance(e, FsEvent) and e.op == "created" and e.path.endswith("w.txt") for e in events)
    # modify
    cortex.write("w.txt", "12", overwrite=True)
    events, state = cortex.watch_once(".", prev_state=state)
    assert any(e.op == "modified" and e.path.endswith("w.txt") for e in events)
    # delete
    cortex.delete("w.txt")
    events, state = cortex.watch_once(".", prev_state=state)
    assert any(e.op == "deleted" and e.path.endswith("w.txt") for e in events)
