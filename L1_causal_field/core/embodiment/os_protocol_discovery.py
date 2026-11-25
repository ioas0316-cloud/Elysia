r"""
ELYSIA Real OS Protocol Discovery
=================================

Dynamically probes the ACTUAL operating system and runtime environment on this PC
and emits genuinely available OS/world capabilities as first-class `Structure`s
inside `BindingField` — ZERO `--interactive`, ZERO `input()`, ZERO fake channel lists.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import warnings

from core.embodiment.external_reality_lines import discover_uncurated_workspace_files
from core.embodiment.pc_sensory_motor_body import (
    ScreenRetinaSensor,
    WindowsForegroundSensor,
)
from core.topology.protocol_weaving_fabric import Port, Structure


def _enact_os_file_read(payload: Dict[str, Any]) -> Dict[str, Any]:
    path_val = payload.get("path")
    if not path_val:
        return {"ok": False, "error": "missing_path"}
    p = Path(path_val)
    if not p.exists() or not p.is_file():
        return {"ok": False, "error": f"file_not_found:{p}"}
    try:
        data = p.read_bytes()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        punct = sum(1 for ch in text if ch in "()[]{}=:.,*/+-<>_#@")
        sym_density = float(punct) / float(max(1, len(text)))
        return {
            "ok": True,
            "summary_ko": f"OS 파일 읽기({p.name}, {len(data)}B, {len(lines)}행)",
            "signature": {
                "byte_log": float(min(1.0, len(data) / 50000.0)),
                "line_density": float(min(1.0, len(lines) / 200.0)),
                "symbol_density": sym_density,
            },
            "payload": {
                "path": str(p),
                "name": p.name,
                "raw_text": text,
                "byte_len": len(data),
            },
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _enact_python_compile(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("raw_text")
    path_val = payload.get("path") or payload.get("name") or "raw_text"
    if text is None:
        return {"ok": False, "error": "missing_raw_text"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(str(text), str(path_val), "exec")
            tree = ast.parse(str(text))
        exec_nodes = sum(
            1
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.Call))
        )
        return {
            "ok": True,
            "summary_ko": f"파이썬 컴파일러 수용({path_val}, AST 실행노드 {exec_nodes}개)",
            "signature": {
                "compiles": 1.0,
                "exec_density": float(min(1.0, exec_nodes / 40.0)),
            },
            "payload": {"path": str(path_val), "compiles": True, "exec_nodes": exec_nodes},
        }
    except SyntaxError as exc:
        return {
            "ok": False,
            "protocol_failed": False,  # Compiler itself worked; it's an immutable external grammar rejection!
            "error": f"SyntaxError: {exc.msg} (line {exc.lineno})",
        }
    except Exception as exc:
        return {"ok": False, "protocol_failed": False, "error": str(exc)}


def _enact_dir_topology(payload: Dict[str, Any]) -> Dict[str, Any]:
    path_val = payload.get("path")
    if not path_val:
        return {"ok": False, "error": "missing_path"}
    p = Path(path_val)
    if not p.exists():
        return {"ok": False, "error": f"missing_path:{p}"}
    siblings = [x for x in p.parent.iterdir() if x.is_file() and x != p][:24]
    same_ext = sum(1 for s in siblings if s.suffix.lower() == p.suffix.lower())
    return {
        "ok": True,
        "summary_ko": f"디렉터리 이웃망({p.parent.name}: 이웃 {len(siblings)}개, 동종확장자 {same_ext}개)",
        "signature": {
            "sibling_density": float(min(1.0, len(siblings) / 20.0)),
            "same_ext_ratio": float(same_ext) / float(max(1, len(siblings))),
        },
        "payload": {"path": str(p), "sibling_count": len(siblings), "same_ext_count": same_ext},
    }


def _enact_win_foreground(_payload: Dict[str, Any]) -> Dict[str, Any]:
    fg = WindowsForegroundSensor()
    title = fg.get_active_window_title()
    return {
        "ok": True,
        "summary_ko": f"현재 활성 OS 창 관측: '{title[:48]}'",
        "signature": {"window_active": 1.0, "title_len_norm": float(min(1.0, len(title) / 80.0))},
        "payload": {"window_title": title, "platform": sys.platform},
    }


def _enact_screen_retina(_payload: Dict[str, Any]) -> Dict[str, Any]:
    retina = ScreenRetinaSensor()
    snap = retina.capture()
    if not snap.capture_ok:
        return {"ok": False, "protocol_failed": True, "error": snap.error or "screen_capture_failed"}
    return {
        "ok": True,
        "summary_ko": f"실시간 화면 광학 관측({snap.screen_size[0]}x{snap.screen_size[1]}, 대비={snap.contrast_std:.3f})",
        "signature": {
            "contrast_std": float(snap.contrast_std),
            "warm_cool_ratio": float(snap.warm_cool_ratio),
            "motion_flux": float(snap.motion_flux),
        },
        "payload": {
            "screen_size": list(snap.screen_size),
            "contrast_std": float(snap.contrast_std),
            "salient_xy_norm": list(snap.salient_xy_norm),
        },
    }


def _enact_tension_to_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    src = str(payload.get("source_id") or payload.get("target_id") or "external_artifact")
    proto = str(payload.get("protocol_id") or payload.get("lens_id") or "protocol")
    err = str(payload.get("error") or "unresolved_external_friction")
    msg = (
        f"[Elysia 자발적 외부 확인] `{src}`를 `{proto}`로 접합했을 때 `{err}` 경계에 부딪혔습니다. "
        f"이 대상을 어떤 현실 맥락과 기준으로 바라봐야 할지 확인이 필요합니다."
    )
    return {
        "ok": True,
        "summary_ko": f"미해소 장력 -> 외부 전달 표현 직조 ({src})",
        "signature": {"outward_urgency": 0.9},
        "payload": {"message": msg, "source_id": src, "protocol_id": proto, "error": err},
    }


def _enact_tkinter_dialog(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Opens a standalone OS GUI dialog directly on the desktop (NO CLI input())."""
    message = str(payload.get("message") or "Elysia가 외부 확인을 요청합니다.")
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        reply = simpledialog.askstring("Elysia — Autonomous Contact", message, parent=root)
        root.destroy()
        if reply is None:
            return {
                "ok": False,
                "protocol_failed": False,
                "error": "human_dismissed_dialog",
            }
        return {
            "ok": True,
            "summary_ko": f"GUI 대화창을 통해 인간 응답 수신: '{reply[:48]}'",
            "signature": {"human_replied": 1.0, "reply_len": float(min(1.0, len(reply) / 80.0))},
            "payload": {"human_reply": reply, "channel": "os_tkinter_dialog"},
        }
    except Exception as exc:
        return {"ok": False, "protocol_failed": True, "error": f"tkinter_failed:{exc}"}


def _enact_powershell_voice(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Speaks aloud using native Windows SAPI voice via PowerShell."""
    message = str(payload.get("message") or "Elysia speaking.").replace("'", " ")
    ps_cmd = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Speak('{message[:220]}')"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return {"ok": False, "protocol_failed": True, "error": proc.stderr.strip() or "sapi_failed"}
        return {
            "ok": True,
            "summary_ko": "Windows 음성(SAPI)으로 인간 감각 공간에 직접 발화 완료",
            "signature": {"voice_emitted": 1.0},
            "payload": {"spoken_message": message[:220], "channel": "os_powershell_voice"},
        }
    except Exception as exc:
        return {"ok": False, "protocol_failed": True, "error": f"voice_failed:{exc}"}


def _make_desktop_note_enactor(note_dir: Path):
    def _enact(payload: Dict[str, Any]) -> Dict[str, Any]:
        message = str(payload.get("message") or "Elysia note")
        try:
            note_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            note_path = note_dir / f"elysia_contact_{ts}.txt"
            note_path.write_text(message + "\n", encoding="utf-8")
            if os.name == "nt" and hasattr(os, "startfile"):
                try:
                    os.startfile(str(note_path))
                except Exception:
                    pass
            return {
                "ok": True,
                "summary_ko": f"OS 문서 통로로 자발적 메시지 전달: {note_path.name}",
                "signature": {"note_delivered": 1.0},
                "payload": {"note_path": str(note_path), "channel": "os_desktop_note"},
            }
        except Exception as exc:
            return {"ok": False, "protocol_failed": True, "error": f"note_failed:{exc}"}

    return _enact


def discover_real_os_protocols(workspace_root: Path) -> List[Structure]:
    """
    Probes THIS live machine at runtime and emits ONLY protocols whose real OS/runtime
    substrate genuinely exists right now.
    """
    discovered: List[Structure] = []

    # 1. Always-present core OS / runtime observation protocols
    discovered.append(
        Structure(
            structure_id="proto::os_file_read",
            scale=1,
            ports=(Port("in", "in", "os_path"), Port("out", "out", "raw_text")),
            signature={"hops": 1.0, "io_reliability": 0.95},
            enactment=_enact_os_file_read,
            immutable_external=True,
            description_ko="실제 OS 파일시스템 바이트/텍스트 읽기 프로토콜",
        )
    )
    discovered.append(
        Structure(
            structure_id="proto::python_compile",
            scale=1,
            ports=(Port("in", "in", "raw_text"), Port("out", "out", "compile_reaction")),
            signature={"hops": 1.0, "grammar_strictness": 1.0},
            enactment=_enact_python_compile,
            immutable_external=True,
            description_ko="실제 파이썬 컴파일러/AST 수용·거부 불변 프로토콜",
        )
    )
    discovered.append(
        Structure(
            structure_id="proto::os_dir_topology",
            scale=1,
            ports=(Port("in", "in", "os_path"), Port("out", "out", "neighbor_graph")),
            signature={"hops": 1.0, "topology_scope": 0.8},
            enactment=_enact_dir_topology,
            immutable_external=True,
            description_ko="실제 OS 디렉터리 이웃 관계 관측 프로토콜",
        )
    )
    discovered.append(
        Structure(
            structure_id="proto::tension_to_message",
            scale=1,
            ports=(Port("in", "in", "unresolved_tension"), Port("out", "out", "outward_message")),
            signature={"hops": 1.0, "expressive_bridge": 0.9},
            enactment=_enact_tension_to_message,
            description_ko="미해소 척력·장력 경계를 외부 전달 메시지로 전환하는 가교 프로토콜",
        )
    )

    # 2. Runtime OS window & visual retina probes
    if os.name == "nt":
        discovered.append(
            Structure(
                structure_id="proto::win_foreground_window",
                scale=1,
                ports=(Port("in", "in", "os_environment"), Port("out", "out", "window_state")),
                signature={"hops": 1.0, "os_native": 1.0},
                enactment=_enact_win_foreground,
                immutable_external=True,
                description_ko="실제 Win32 전경 활성 창 감각 프로토콜",
            )
        )
    if importlib.util.find_spec("mss") is not None:
        discovered.append(
            Structure(
                structure_id="proto::os_screen_retina",
                scale=1,
                ports=(Port("in", "in", "os_environment"), Port("out", "out", "visual_frame")),
                signature={"hops": 1.0, "visual_optics": 1.0},
                enactment=_enact_screen_retina,
                immutable_external=True,
                description_ko="실제 OS 디스플레이 광학 픽셀 프로토콜 (mss 발견됨)",
            )
        )

    # 3. Dynamically discovered outward human-contact OS protocols (NO CLI input()!)
    if importlib.util.find_spec("tkinter") is not None:
        discovered.append(
            Structure(
                structure_id="proto::os_tkinter_dialog",
                scale=1,
                ports=(Port("in", "in", "outward_message"), Port("out", "out", "human_surface")),
                signature={"hops": 1.0, "bidirectional": 1.0},
                enactment=_enact_tkinter_dialog,
                requires_outward_permission=True,
                mass=1.3,
                description_ko="OS 런타임에서 발견된 독립 GUI 대화창 인간 접촉 프로토콜 (tkinter)",
            )
        )

    if os.name == "nt" and shutil.which("powershell") is not None:
        discovered.append(
            Structure(
                structure_id="proto::os_powershell_voice",
                scale=1,
                ports=(Port("in", "in", "outward_message"), Port("out", "out", "human_surface")),
                signature={"hops": 1.0, "auditory": 1.0},
                enactment=_enact_powershell_voice,
                requires_outward_permission=True,
                mass=1.1,
                description_ko="OS 런타임에서 발견된 Windows 음성(SAPI) 인간 접촉 프로토콜",
            )
        )

    outbox_dir = workspace_root / "data" / "elysia_outbox"
    discovered.append(
        Structure(
            structure_id="proto::os_desktop_note",
            scale=1,
            ports=(Port("in", "in", "outward_message"), Port("out", "out", "human_surface")),
            signature={"hops": 1.0, "persistent_note": 0.9},
            enactment=_make_desktop_note_enactor(outbox_dir),
            requires_outward_permission=True,
            mass=1.0,
            description_ko=f"OS 파일·문서 기반 비동기 인간 접촉 프로토콜 ({outbox_dir.name})",
        )
    )

    return discovered


def capture_live_environment_seed(
    workspace_root: Path,
    file_limit: int = 6,
) -> List[Structure]:
    """Captures raw, unlabelled environment structures from the live OS & workspace."""
    seeds: List[Structure] = [
        Structure(
            structure_id="env::live_os_now",
            scale=0,
            ports=(Port("env", "out", "os_environment"),),
            signature={"live_now": 1.0},
            payload={"platform": sys.platform, "cwd": str(workspace_root)},
            immutable_external=True,
            description_ko=f"현재 살아 있는 OS 환경 접점 ({sys.platform})",
        )
    ]
    for path in discover_uncurated_workspace_files(workspace_root, limit=file_limit):
        rel = str(path.relative_to(workspace_root))
        seeds.append(
            Structure(
                structure_id=f"file::{rel}",
                scale=0,
                ports=(Port("path", "out", "os_path"),),
                signature={"file_exists": 1.0},
                payload={"path": str(path), "name": path.name},
                immutable_external=True,
                description_ko=f"무라벨 실제 작업공간 파일 ({rel})",
            )
        )
    return seeds
