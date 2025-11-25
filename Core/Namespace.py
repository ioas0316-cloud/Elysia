"""
Logical namespace loader for Elysia.

Maps conceptual namespaces (Core, Apps.Tools, Assets.Images, Runtime.Saves, Legacy, ...)
to concrete Python modules/paths using MIRROR_MAP.yaml.

This is the first step toward decoupling code from physical folder layout.
"""

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
MIRROR_MAP_PATH = ROOT / "MIRROR_MAP.yaml"


def _load_namespaces() -> Dict[str, str]:
    """Lightweight YAML section parser for 'namespaces' (no external deps)."""
    mapping: Dict[str, str] = {}
    if not MIRROR_MAP_PATH.exists():
        return mapping
    lines = MIRROR_MAP_PATH.read_text(encoding="utf-8").splitlines()
    in_ns = False
    for line in lines:
        if line.strip().startswith("namespaces:"):
            in_ns = True
            continue
        if in_ns:
            # stop when dedented back to top level
            if line and not line.startswith("  "):
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" in stripped:
                key, val = stripped.split(":", 1)
                ns = key.strip()
                path = val.strip().strip('"').strip("'")
                if ns and path:
                    mapping[ns] = path
    return mapping


NAMESPACE_MAP = _load_namespaces()


def load_module(namespace: str) -> Any:
    """
    Load a Python module by logical namespace.

    Example:
        load_module("Apps.Tools.live_loop")
    """
    if "." not in namespace:
        # Direct module name, fall back to standard import
        return importlib.import_module(namespace)

    prefix, mod = namespace.rsplit(".", 1)
    base = NAMESPACE_MAP.get(prefix)
    if base is None:
        # Unknown namespace; try importing as-is
        return importlib.import_module(namespace)

    # Ensure project root is on sys.path
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    module_path = f"{base}.{mod}" if base else mod
    return importlib.import_module(module_path)


def resolve_path(namespace: str) -> Path:
    """
    Resolve a logical namespace to a filesystem path (for scripts/resources).

    Example:
        resolve_path("Runtime.Saves.elysia_state.json")
    """
    if "." not in namespace:
        return ROOT / namespace

    prefix, rest = namespace.split(".", 1)
    base = NAMESPACE_MAP.get(prefix)
    if base is None:
        return ROOT / namespace.replace(".", os.sep)

    return ROOT / base / rest.replace(".", os.sep)

