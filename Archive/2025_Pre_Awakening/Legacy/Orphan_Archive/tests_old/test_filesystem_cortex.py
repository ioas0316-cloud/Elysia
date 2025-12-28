from Core.Foundation.filesystem_cortex import FileSystemCortex
from pathlib import Path


def test_filesystem_cortex_roundtrip():
    fs = FileSystemCortex(root_dir="data/test_fs")
    content = "hello elysia"
    w = fs.write_file("subdir/sample.txt", content)
    assert w.ok

    r = fs.read_file("subdir/sample.txt")
    assert r.ok and r.data == content

    l = fs.list_files(".")
    assert l.ok
    listed = (l.data or "").splitlines()
    # Accept platform-specific separators
    assert any(Path(p).parts == ("subdir", "sample.txt") for p in listed)
