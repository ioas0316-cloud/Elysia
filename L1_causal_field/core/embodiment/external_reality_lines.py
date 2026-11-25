r"""
ELYSIA External Reality Lines (Raw, Unlabelled Contact Prisms)
=============================================================

Provides pure contact prisms (`CausalLine`s) and an open external probe action
over real OS / filesystem / Python runtime / screen realities.
There are ZERO hardcoded labels, ZERO developer taxonomies, ZERO `expected` tags.
Every Line reports only real physical/structural observables directly returned by
the outside environment onto a continuous bipolar spectrum `[-1.0, +1.0]`.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np

from core.embodiment.pc_sensory_motor_body import (
    ScreenRetinaSensor,
    WindowsForegroundSensor,
    _measure_rgb_image_sensory,
)
from core.topology.self_weaving_loom import CausalLine, LineRefraction, _clamp


def _read_referent_text(raw_referent: Dict[str, Any]) -> Tuple[str, Optional[Path], int]:
    """Reads actual UTF-8/raw text and size directly from disk or raw payload."""
    raw_text = raw_referent.get("raw_text")
    path_val = raw_referent.get("path")
    p = Path(path_val) if path_val else None
    if raw_text is not None:
        txt = str(raw_text)
        byte_len = len(txt.encode("utf-8", errors="ignore"))
        return txt, p, byte_len
    if p is not None and p.exists() and p.is_file():
        data = p.read_bytes()
        return data.decode("utf-8", errors="replace"), p, len(data)
    return "", p, 0


class FileSurfaceLine(CausalLine):
    """
    Refracts the raw byte/character surface geometry of a real file:
      - Negative pole: loose prose / sparse punctuation / flat unindented layout
      - Positive pole: dense symbolic operators / hierarchical indentation / compact formal lines
    """

    def __init__(self) -> None:
        super().__init__(
            line_id="line::file_surface",
            scale=1,
            chromatic_bias=(0.45, 0.35, 0.20),
            description="실제 파일 바이트·문자 표면(들여쓰기·기호밀도·행분산) 굴절선",
        )

    def refract(
        self,
        referent_id: str,
        raw_referent: Dict[str, Any],
        step: int,
    ) -> Optional[LineRefraction]:
        text, path_obj, byte_len = _read_referent_text(raw_referent)
        if not text and path_obj is None:
            return None

        lines = text.splitlines()
        non_empty = [ln for ln in lines if ln.strip()]
        n_lines = max(1, len(non_empty))

        indented = sum(1 for ln in non_empty if ln.startswith(("    ", "\t", "  ")))
        indent_ratio = float(indented) / float(n_lines)

        punct_chars = sum(1 for ch in text if ch in "()[]{}=:.,*/+-<>_#@")
        char_len = max(1, len(text))
        symbol_density = float(punct_chars) / float(char_len)

        lengths = [len(ln) for ln in non_empty] or [0]
        mean_len = float(sum(lengths)) / float(len(lengths))
        len_std = float(np.std(np.asarray(lengths, dtype=np.float64)))
        compactness = math.exp(-mean_len / 95.0)

        # Continuous bipolar surface coordinate without hard category thresholds
        surface_score = (
            1.65 * (symbol_density - 0.085)
            + 0.75 * (indent_ratio - 0.30)
            + 0.35 * (compactness - 0.50)
            - 0.15 * min(1.0, len_std / 80.0)
        )
        # If an external probe revealed cross-file references or docstring span, retain mild surface nuance
        probe = raw_referent.get("probe_evidence") or {}
        if "docstring_ratio" in probe:
            surface_score -= 0.20 * float(probe["docstring_ratio"])

        spectrum = _clamp(math.tanh(surface_score * 1.4), -1.0, 1.0)
        return self._record_refraction(
            referent_id=referent_id,
            spectrum_value=spectrum,
            raw_observables={
                "byte_len": float(byte_len),
                "line_count": float(len(lines)),
                "indent_ratio": indent_ratio,
                "symbol_density": symbol_density,
                "len_std": len_std,
            },
            chromatic=(0.30 + 0.4 * symbol_density, 0.35 + 0.35 * indent_ratio, 0.25),
            evidence={"path": str(path_obj) if path_obj else referent_id},
            step=step,
        )


class PythonRuntimeLine(CausalLine):
    """
    Tests raw text directly against Python's real `compile()` / `ast.parse()` causality:
      - Negative pole: rejected by Python compiler (SyntaxError / zero executable AST)
      - Positive pole: accepted by Python compiler with active executable constructs (`import`/`def`/`class`/`call`)
    Note: Performs ONLY safe compilation/AST reaction; NEVER `exec()`s untrusted target files.
    """

    def __init__(self) -> None:
        super().__init__(
            line_id="line::python_runtime",
            scale=1,
            chromatic_bias=(0.25, 0.60, 0.15),
            description="실제 파이썬 컴파일러/AST 실행 반응성 굴절선",
        )

    def refract(
        self,
        referent_id: str,
        raw_referent: Dict[str, Any],
        step: int,
    ) -> Optional[LineRefraction]:
        text, path_obj, _ = _read_referent_text(raw_referent)
        if not text and path_obj is None:
            return None

        compiles = False
        parse_error = ""
        exec_nodes = 0
        doc_only_exprs = 0
        total_stmts = 0

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                compile(text, str(path_obj or referent_id), "exec")
                tree = ast.parse(text)
            compiles = True
            total_stmts = max(1, len(tree.body))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AugAssign, ast.Return, ast.Call)):
                    exec_nodes += 1
            for stmt in tree.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    doc_only_exprs += 1
        except SyntaxError as exc:
            compiles = False
            parse_error = f"SyntaxError: {exc.msg} (line {exc.lineno})"
        except Exception as exc:
            compiles = False
            parse_error = str(exc)

        probe = raw_referent.get("probe_evidence") or {}
        if not compiles:
            # Failed real compilation sits strongly on the negative pole
            spectrum = -0.88
            exec_density = 0.0
        else:
            exec_density = min(1.0, float(exec_nodes) / max(4.0, float(total_stmts) * 3.0))
            doc_ratio = float(doc_only_exprs) / float(max(1, total_stmts))
            external_refs = min(1.0, float(probe.get("workspace_ref_count", 0)) / 4.0)
            raw_val = 0.25 + 0.65 * exec_density - 0.35 * doc_ratio + 0.15 * external_refs
            spectrum = _clamp(raw_val, -1.0, 1.0)

        return self._record_refraction(
            referent_id=referent_id,
            spectrum_value=spectrum,
            raw_observables={
                "compiles": 1.0 if compiles else 0.0,
                "exec_nodes": float(exec_nodes),
                "doc_only_exprs": float(doc_only_exprs),
                "exec_density": exec_density,
            },
            chromatic=(0.25 if compiles else 0.10, 0.65 if compiles else 0.15, 0.10 if compiles else 0.75),
            evidence={
                "compiles": compiles,
                "parse_error": parse_error,
                "exec_nodes": exec_nodes,
            },
            step=step,
        )


class FilesystemTopologyLine(CausalLine):
    """
    Observes the real OS directory neighborhood around a file:
      - Negative pole: isolated artifact with sparse siblings / few shared references
      - Positive pole: densely connected module neighborhood with same-suffix siblings & inbound references
    """

    def __init__(self, max_siblings: int = 24) -> None:
        super().__init__(
            line_id="line::fs_topology",
            scale=1,
            chromatic_bias=(0.30, 0.45, 0.25),
            description="실제 파일시스템 이웃 분포·참조 연결망 굴절선",
        )
        self.max_siblings = max(4, int(max_siblings))

    def refract(
        self,
        referent_id: str,
        raw_referent: Dict[str, Any],
        step: int,
    ) -> Optional[LineRefraction]:
        path_val = raw_referent.get("path")
        if not path_val:
            return None
        p = Path(path_val)
        if not p.exists() or not p.is_file():
            return None

        parent = p.parent
        siblings = []
        try:
            for item in parent.iterdir():
                if item.is_file() and item != p:
                    siblings.append(item)
                    if len(siblings) >= self.max_siblings:
                        break
        except Exception:
            siblings = []

        same_ext = sum(1 for s in siblings if s.suffix.lower() == p.suffix.lower())
        sib_count = len(siblings)
        same_ext_ratio = float(same_ext) / float(max(1, sib_count))

        stem = p.stem
        inbound_refs = 0
        if len(stem) >= 3:
            for s in siblings[:16]:
                if s.suffix.lower() not in {".py", ".md", ".txt", ".json"}:
                    continue
                try:
                    if s.stat().st_size > 180_000:
                        continue
                    content = s.read_text(encoding="utf-8", errors="ignore")
                    if stem in content:
                        inbound_refs += 1
                except Exception:
                    continue

        probe = raw_referent.get("probe_evidence") or {}
        probe_refs = int(probe.get("workspace_ref_count", 0))
        total_refs = max(inbound_refs, probe_refs)

        density_term = min(1.0, float(sib_count) / 14.0)
        ref_term = min(1.0, float(total_refs) / 4.0)
        score = (0.85 * same_ext_ratio + 0.55 * density_term + 0.80 * ref_term) - 0.85
        spectrum = _clamp(math.tanh(score), -1.0, 1.0)

        return self._record_refraction(
            referent_id=referent_id,
            spectrum_value=spectrum,
            raw_observables={
                "sibling_count": float(sib_count),
                "same_ext_count": float(same_ext),
                "inbound_refs": float(total_refs),
            },
            chromatic=(0.25, 0.45 + 0.3 * ref_term, 0.30),
            evidence={"parent_dir": str(parent), "stem": stem, "inbound_refs": total_refs},
            step=step,
        )


class LiveScreenLine(CausalLine):
    """
    Observes the live desktop retina (`ScreenRetinaSensor`) or explicit RGB frame:
      - Negative pole: quiet / low-contrast / cool surface
      - Positive pole: dynamic / high-contrast / active foreground surface
    Returns `None` when a referent carries neither a `frame` nor `capture_screen=True`.
    """

    def __init__(
        self,
        retina: Optional[ScreenRetinaSensor] = None,
        fg_sensor: Optional[WindowsForegroundSensor] = None,
    ) -> None:
        super().__init__(
            line_id="line::live_screen",
            scale=1,
            chromatic_bias=(0.50, 0.25, 0.25),
            description="실시간 화면 픽셀 에너지·대비·전경 창 변화 굴절선",
        )
        self.retina = retina or ScreenRetinaSensor()
        self.fg_sensor = fg_sensor or WindowsForegroundSensor()

    def refract(
        self,
        referent_id: str,
        raw_referent: Dict[str, Any],
        step: int,
    ) -> Optional[LineRefraction]:
        frame_rgb = raw_referent.get("screen_rgb")
        capture_screen = bool(raw_referent.get("capture_screen", False))
        if frame_rgb is None and not capture_screen:
            return None

        if frame_rgb is not None:
            stats = _measure_rgb_image_sensory(self.retina._to_rgb_pil(np.asarray(frame_rgb)))
            contrast = float(stats["contrast_std"])
            sharp = float(stats["edge_sharpness"])
            warm_cool = float(stats["warm_cool_ratio"])
            motion = 0.0
            window_title = str(raw_referent.get("window_title", "provided_frame"))
            capture_ok = True
        else:
            snap = self.retina.capture()
            if not snap.capture_ok:
                return None
            contrast = float(snap.contrast_std)
            sharp = float(np.mean(np.abs(np.diff(snap.frame_small.mean(axis=2), axis=1))))
            warm_cool = float(snap.warm_cool_ratio)
            motion = float(snap.motion_flux)
            window_title = self.fg_sensor.get_active_window_title()
            capture_ok = True

        score = (contrast - 0.16) * 2.5 + (sharp - 0.04) * 4.0 + motion * 3.0 + (warm_cool - 1.0) * 0.4
        spectrum = _clamp(math.tanh(score), -1.0, 1.0)
        return self._record_refraction(
            referent_id=referent_id,
            spectrum_value=spectrum,
            raw_observables={
                "contrast_std": contrast,
                "edge_sharpness": sharp,
                "warm_cool_ratio": warm_cool,
                "motion_flux": motion,
            },
            chromatic=(0.35 + 0.4 * motion, 0.30 + 0.3 * min(1.0, contrast * 2.0), 0.25),
            evidence={"capture_ok": capture_ok, "window_title": window_title},
            step=step,
        )


def external_world_probe_action(
    raw_referent: Dict[str, Any],
    workspace_root: Optional[Path] = None,
    max_scan_files: int = 28,
) -> Dict[str, Any]:
    """
    Open-world `action_line` invoked automatically when a `CrossNode` hits high tension.
    Performs deeper genuine external inspection rather than internal math tweaking:
      - Inspects real AST top-level docstring span vs executable statement count
      - Scans neighboring workspace files to see if other real artifacts reference/import this stem
      - Captures live retina high-res zoom delta if a screen referent is active
    """
    existing = raw_referent.get("probe_evidence") or {}
    path_val = raw_referent.get("path")
    if path_val:
        p = Path(path_val)
        if not p.exists() or not p.is_file():
            return {"produced_new_evidence": False, "reason": "missing_file"}
        # Avoid falsely claiming brand-new evidence if we already probed this exact path cleanly
        if existing.get("probed_path") == str(p):
            return {"produced_new_evidence": False, "probed_path": str(p)}

        text = p.read_bytes().decode("utf-8", errors="replace")
        docstring_ratio = 0.0
        compiles_ok = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text)
            compiles_ok = True
            total_lines = max(1, len(text.splitlines()))
            doc_lines = 0
            for stmt in tree.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    end_ln = getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno
                    doc_lines += max(1, end_ln - stmt.lineno + 1)
            docstring_ratio = min(1.0, float(doc_lines) / float(total_lines))
        except Exception:
            compiles_ok = False

        root = workspace_root
        if root is None:
            root = p.parent.parent if p.parent.parent.exists() else p.parent

        stem = p.stem
        ref_files: List[str] = []
        scanned = 0
        if len(stem) >= 3 and root.exists():
            for cand in root.rglob("*.py"):
                if scanned >= max_scan_files:
                    break
                if cand == p:
                    continue
                scanned += 1
                try:
                    if cand.stat().st_size > 160_000:
                        continue
                    src = cand.read_text(encoding="utf-8", errors="ignore")
                    if stem in src:
                        ref_files.append(cand.name)
                except Exception:
                    continue

        return {
            "produced_new_evidence": True,
            "probed_path": str(p),
            "compiles_ok": compiles_ok,
            "docstring_ratio": round(docstring_ratio, 4),
            "workspace_ref_count": len(ref_files),
            "referencing_files": ref_files[:6],
            "scanned_neighbors": scanned,
        }

    # Screen referent probe (if applicable)
    if raw_referent.get("capture_screen") and not existing.get("probed_screen"):
        retina = ScreenRetinaSensor()
        zoom = retina.capture_salient_zoom_crop()
        return {
            "produced_new_evidence": bool(zoom.get("ok", False)),
            "probed_screen": True,
            "zoom_sharpness_gain": zoom.get("sharpness_gain", 0.0),
            "detail_revealed": zoom.get("detail_revealed", False),
        }

    return {"produced_new_evidence": False}


def extract_raw_external_observables(raw_referent: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts purely unlabelled raw structural/physical observables from disk or raw text
    (used by `SelfWeavingLoom` to birth new `EmergentExternalLine` prisms when external
    refutation exposes a blind spot in existing lines).
    """
    text, path_obj, byte_len = _read_referent_text(raw_referent)
    if not text and path_obj is None:
        return {}

    if len(text) > 120_000:
        text = text[:120_000]

    lines = text.splitlines()
    n_lines = max(1, len(lines))
    non_empty = [ln for ln in lines if ln.strip()]
    n_non_empty = max(1, len(non_empty))
    char_len = max(1, len(text))

    triple_quote_hits = text.count('"""') + text.count("'''")
    heading_lines = sum(1 for ln in non_empty if ln.lstrip().startswith("# "))
    comment_lines = sum(
        1 for ln in non_empty if ln.lstrip().startswith("#") and not ln.lstrip().startswith("# ")
    )
    colon_lines = sum(1 for ln in non_empty if ln.rstrip().endswith(":"))
    brace_count = sum(1 for ch in text if ch in "{}")
    bracket_count = sum(1 for ch in text if ch in "[]")
    equals_count = text.count("=")
    non_ascii_count = sum(1 for ch in text if ord(ch) > 127)
    digit_count = sum(1 for ch in text if ch.isdigit())
    blank_lines = sum(1 for ln in lines if not ln.strip())
    max_indent = max((len(ln) - len(ln.lstrip(" \t")) for ln in non_empty), default=0)

    return {
        "triple_quote_ratio": round(min(1.0, float(triple_quote_hits * 3) / float(n_non_empty)), 5),
        "heading_hash_ratio": round(float(heading_lines) / float(n_non_empty), 5),
        "colon_line_ratio": round(float(colon_lines) / float(n_non_empty), 5),
        "brace_density": round(min(1.0, float(brace_count) / float(char_len) * 20.0), 5),
        "bracket_density": round(min(1.0, float(bracket_count) / float(char_len) * 20.0), 5),
        "comment_line_ratio": round(float(comment_lines) / float(n_non_empty), 5),
        "equals_density": round(min(1.0, float(equals_count) / float(char_len) * 25.0), 5),
        "non_ascii_ratio": round(float(non_ascii_count) / float(char_len), 5),
        "digit_density": round(min(1.0, float(digit_count) / float(char_len) * 10.0), 5),
        "blank_line_ratio": round(float(blank_lines) / float(n_lines), 5),
        "max_indent_depth": round(min(1.0, float(max_indent) / 24.0), 5),
        "file_size_log": round(min(1.0, math.log1p(max(0, byte_len)) / 12.0), 5),
    }


def discover_uncurated_workspace_files(
    root: Path,
    limit: int = 12,
) -> List[Path]:
    """
    Discovers diverse real files across a directory tree WITHOUT developer labels or `expected` tags.
    Buckets gently by file suffix only so the encounter stream isn't monoculturally one extension.
    """
    if not root.exists():
        return []
    by_suffix: Dict[str, List[Path]] = {}
    ignored_parts = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        if any(part in ignored_parts for part in item.parts):
            continue
        try:
            size = item.stat().st_size
        except OSError:
            continue
        if size == 0 or size > 250_000:
            continue
        sfx = item.suffix.lower() or "<none>"
        if sfx not in {".py", ".md", ".ini", ".txt", ".json", ".toml", ".yaml", ".yml"}:
            continue
        bucket = by_suffix.setdefault(sfx, [])
        if len(bucket) < 6:
            bucket.append(item)
        if sum(len(v) for v in by_suffix.values()) >= limit * 3:
            break

    selected: List[Path] = []
    suffixes = sorted(by_suffix.keys())
    idx = 0
    while len(selected) < limit and any(by_suffix.values()):
        s = suffixes[idx % len(suffixes)]
        if by_suffix[s]:
            selected.append(by_suffix[s].pop(0))
        idx += 1
        if idx > limit * 10:
            break
    return selected
