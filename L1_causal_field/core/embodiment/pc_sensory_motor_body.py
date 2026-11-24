r"""
ELYSIA Universal PC Sensory-Motor Body
======================================

Provides one unified sensory-motor interface over any screen and OS surface:
- `WindowsIdleSensor`: continuous detection of user idle duration via Win32 `GetLastInputInfo`.
- `ScreenRetinaSensor`: primary monitor capture (`mss`) compressed to a 48-D spatial rotor field,
  measuring temporal motion flux, contrast, warm/cool balance, and highest-salience focus.
- `UniversalMotorActuator`: non-site-specific physical actions (`observe_attend`,
  `speak_recombine`, `scroll_explore`, `click_salient`, `type_curiosity`, `open_web_search`)
  strictly gated by `AgencyMode.IDLE_FREE_ROAM` and user idle duration.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import Enum
import os
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import quote
import webbrowser

import numpy as np
from PIL import Image


class AgencyMode(str, Enum):
    OBSERVE_AND_CHAT = "observe_and_chat"
    IDLE_FREE_ROAM = "idle_free_roam"


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("dwTime", ctypes.c_uint32),
    ]


class WindowsIdleSensor:
    """Queries elapsed user inactivity seconds via Windows GetLastInputInfo."""

    def __init__(self, idle_provider: Optional[Callable[[], float]] = None) -> None:
        self._idle_provider = idle_provider

    def get_idle_seconds(self) -> float:
        if self._idle_provider is not None:
            try:
                return max(0.0, float(self._idle_provider()))
            except Exception:
                return 0.0
        if os.name != "nt":
            return 0.0
        try:
            lii = _LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
            ok = ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
            if not ok:
                return 0.0
            tick64 = int(ctypes.windll.kernel32.GetTickCount64())
            elapsed_ms = ((tick64 & 0xFFFFFFFF) - int(lii.dwTime)) & 0xFFFFFFFF
            return float(elapsed_ms) / 1000.0
        except Exception:
            return 0.0


@dataclass
class ScreenFrameSnapshot:
    """Downsampled retina observation carrying spatial-chromatic energy and focus."""

    frame_small: np.ndarray  # (24, 32, 3) float64 in [0, 1]
    visual_vec: np.ndarray  # (48,) float64 in [0, 1]
    motion_flux: float
    contrast_std: float
    warm_cool_ratio: float
    salient_xy_norm: Tuple[float, float]
    screen_size: Tuple[int, int]
    capture_ok: bool = True
    error: str = ""


class ScreenRetinaSensor:
    """Captures the desktop screen via `mss` or injectable test frame provider."""

    def __init__(
        self,
        frame_provider: Optional[Callable[[], np.ndarray]] = None,
        monitor_index: int = 1,
    ) -> None:
        self._frame_provider = frame_provider
        self._monitor_index = max(1, int(monitor_index))
        self._prev_frame: Optional[np.ndarray] = None

    def _to_frame_small(self, arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        np_arr = np.asarray(arr)
        if np_arr.ndim == 2:
            np_arr = np.repeat(np_arr[..., None], 3, axis=2)
        elif np_arr.ndim == 3 and np_arr.shape[2] == 4:
            np_arr = np_arr[..., :3]
        h, w = int(np_arr.shape[0]), int(np_arr.shape[1])
        if np_arr.dtype != np.uint8:
            max_v = float(np_arr.max()) if np_arr.size else 1.0
            scale = 255.0 if max_v <= 1.0 else 1.0
            u8 = np.clip(np_arr.astype(np.float64) * scale, 0.0, 255.0).astype(np.uint8)
        else:
            u8 = np_arr
        if (h, w) != (24, 32):
            pil = Image.fromarray(u8, mode="RGB").resize((32, 24), Image.Resampling.BILINEAR)
            small = np.asarray(pil, dtype=np.float64) / 255.0
        else:
            small = u8.astype(np.float64) / 255.0
        return small, (w, h)

    def _snapshot_from_small(
        self,
        frame_small: np.ndarray,
        screen_size: Tuple[int, int],
        capture_ok: bool = True,
        error: str = "",
    ) -> ScreenFrameSnapshot:
        blocks = frame_small.reshape(4, 6, 4, 8, 3)
        visual_vec = blocks.mean(axis=(1, 3)).reshape(48).astype(np.float64)

        if self._prev_frame is not None and self._prev_frame.shape == frame_small.shape:
            diff_abs = np.abs(frame_small - self._prev_frame)
            motion_flux = float(np.clip(np.mean(diff_abs), 0.0, 1.0))
            cell_motion = diff_abs.reshape(4, 6, 4, 8, 3).mean(axis=(1, 3, 4))
        else:
            motion_flux = 0.0
            cell_motion = np.zeros((4, 4), dtype=np.float64)

        self._prev_frame = frame_small.copy()

        cell_std = blocks.std(axis=(1, 3, 4))
        salience = cell_std + 2.5 * cell_motion
        r_idx, c_idx = divmod(int(np.argmax(salience)), 4)
        salient_xy_norm = (round((c_idx + 0.5) / 4.0, 4), round((r_idx + 0.5) / 4.0, 4))

        contrast_std = round(float(frame_small.std()), 4)
        warm_cool_ratio = round(
            float((frame_small[..., 0].mean() + 0.1) / (frame_small[..., 2].mean() + 0.1)),
            4,
        )
        return ScreenFrameSnapshot(
            frame_small=frame_small,
            visual_vec=visual_vec,
            motion_flux=motion_flux,
            contrast_std=contrast_std,
            warm_cool_ratio=warm_cool_ratio,
            salient_xy_norm=salient_xy_norm,
            screen_size=screen_size,
            capture_ok=capture_ok,
            error=error,
        )

    def capture(self) -> ScreenFrameSnapshot:
        if self._frame_provider is not None:
            raw = self._frame_provider()
            small, size = self._to_frame_small(raw)
            return self._snapshot_from_small(small, size, capture_ok=True)
        try:
            import mss

            with mss.mss() as sct:
                idx = min(self._monitor_index, len(sct.monitors) - 1)
                mon = sct.monitors[idx]
                shot = sct.grab(mon)
                img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                arr = np.asarray(img.resize((32, 24), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
                return self._snapshot_from_small(arr, (int(shot.width), int(shot.height)), capture_ok=True)
        except Exception as exc:
            zeros = np.zeros((24, 32, 3), dtype=np.float64)
            return self._snapshot_from_small(zeros, (1920, 1080), capture_ok=False, error=str(exc))


_PHYSICAL_OS_ACTIONS = {
    "scroll_explore",
    "click_salient",
    "type_curiosity",
    "open_web_search",
}
_INTERNAL_CHAT_ACTIONS = {"observe_attend", "speak_recombine"}
ALL_EMBODIED_ACTIONS = tuple(sorted(_INTERNAL_CHAT_ACTIONS | _PHYSICAL_OS_ACTIONS))


class UniversalMotorActuator:
    """
    Non-site-specific physical OS/browser actuator with strict idle + failsafe gating.
    Never clicks, scrolls, types, or opens pages unless `IDLE_FREE_ROAM` is active
    AND `user_idle_s >= idle_threshold_s`.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = bool(dry_run)
        try:
            import pyautogui

            pyautogui.FAILSAFE = True
        except Exception:
            pass

    def execute(
        self,
        action_kind: str,
        payload: Dict[str, Any],
        agency_mode: AgencyMode,
        user_idle_s: float,
        idle_threshold_s: float,
        screen_size: Tuple[int, int] = (1920, 1080),
    ) -> Dict[str, Any]:
        if action_kind in _INTERNAL_CHAT_ACTIONS:
            return {
                "executed": True,
                "action_kind": action_kind,
                "dry_run": self.dry_run,
                "payload": dict(payload),
            }
        if action_kind not in _PHYSICAL_OS_ACTIONS:
            return {
                "executed": False,
                "action_kind": action_kind,
                "reason": "unknown_action_kind",
            }
        if agency_mode != AgencyMode.IDLE_FREE_ROAM or float(user_idle_s) < float(idle_threshold_s):
            return {
                "executed": False,
                "action_kind": action_kind,
                "reason": "user_active_or_observe_only",
                "user_idle_s": round(float(user_idle_s), 2),
                "idle_threshold_s": float(idle_threshold_s),
            }
        if self.dry_run:
            return {
                "executed": True,
                "action_kind": action_kind,
                "dry_run": True,
                "payload": dict(payload),
            }

        try:
            if action_kind == "open_web_search":
                query = str(payload.get("query") or "causal emergence").strip()
                url = f"https://www.google.com/search?q={quote(query)}"
                webbrowser.open(url)
                return {
                    "executed": True,
                    "action_kind": action_kind,
                    "dry_run": False,
                    "url": url,
                    "query": query,
                }

            import pyautogui

            pyautogui.FAILSAFE = True
            if action_kind == "scroll_explore":
                clicks = int(payload.get("scroll_delta", -320))
                pyautogui.scroll(clicks)
                return {"executed": True, "action_kind": action_kind, "dry_run": False, "scroll_delta": clicks}

            if action_kind == "click_salient":
                xy = payload.get("salient_xy_norm", (0.5, 0.5))
                nx = float(np.clip(float(xy[0]), 0.06, 0.94))
                ny = float(np.clip(float(xy[1]), 0.06, 0.94))
                tx = int(nx * max(1, int(screen_size[0])))
                ty = int(ny * max(1, int(screen_size[1])))
                pyautogui.click(tx, ty)
                return {"executed": True, "action_kind": action_kind, "dry_run": False, "click_xy": [tx, ty]}

            if action_kind == "type_curiosity":
                raw_text = str(payload.get("text") or payload.get("query") or "causal field").strip()
                ascii_text = raw_text.encode("ascii", errors="ignore").decode("ascii").strip() or "causal"
                pyautogui.write(ascii_text[:64], interval=0.02)
                if payload.get("press_enter"):
                    pyautogui.press("enter")
                return {"executed": True, "action_kind": action_kind, "dry_run": False, "typed": ascii_text[:64]}

            return {"executed": False, "action_kind": action_kind, "reason": "unhandled_action"}
        except Exception as exc:
            return {
                "executed": False,
                "action_kind": action_kind,
                "reason": f"actuator_error:{exc}",
            }
