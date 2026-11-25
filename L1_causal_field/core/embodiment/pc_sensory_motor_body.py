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
from pathlib import Path
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


class WindowsForegroundSensor:
    """Senses the active OS application/window title currently in front of Elysia's eye."""

    def __init__(self, title_provider: Optional[Callable[[], str]] = None) -> None:
        self._title_provider = title_provider

    def get_active_window_title(self) -> str:
        if self._title_provider is not None:
            try:
                val = str(self._title_provider() or "").strip()
                return val or "desktop_surface"
            except Exception:
                return "desktop_surface"
        if os.name != "nt":
            return "desktop_surface"
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return "desktop_surface"
            length = int(user32.GetWindowTextLengthW(hwnd))
            if length <= 0:
                return "desktop_surface"
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = str(buf.value or "").strip()
            return title[:120] if title else "desktop_surface"
        except Exception:
            return "desktop_surface"


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
        self._last_raw_image: Optional[Image.Image] = None
        self._last_salient_xy: Tuple[float, float] = (0.5, 0.5)

    def _to_rgb_pil(self, arr: np.ndarray) -> Image.Image:
        """Converts numeric ndarray frames (grayscale/RGBA/float/uint8) cleanly into RGB PIL Image."""
        np_arr = np.asarray(arr)
        if np_arr.ndim == 2:
            np_arr = np.repeat(np_arr[..., None], 3, axis=2)
        elif np_arr.ndim == 3 and np_arr.shape[2] == 4:
            np_arr = np_arr[..., :3]
        if np_arr.dtype != np.uint8:
            max_v = float(np_arr.max()) if np_arr.size else 1.0
            scale = 255.0 if max_v <= 1.0 else 1.0
            u8 = np.clip(np_arr.astype(np.float64) * scale, 0.0, 255.0).astype(np.uint8)
        else:
            u8 = np_arr
        return Image.fromarray(u8, mode="RGB")

    def _to_frame_small(self, arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Image.Image]:
        pil_full = self._to_rgb_pil(arr)
        w, h = pil_full.size
        if (h, w) != (24, 32):
            pil = pil_full.resize((32, 24), Image.Resampling.BILINEAR)
            small = np.asarray(pil, dtype=np.float64) / 255.0
        else:
            small = np.asarray(pil_full, dtype=np.float64) / 255.0
        return small, (w, h), pil_full

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
            small, size, pil_full = self._to_frame_small(raw)
            self._last_raw_image = pil_full
            snap = self._snapshot_from_small(small, size, capture_ok=True)
            self._last_salient_xy = snap.salient_xy_norm
            return snap
        try:
            import mss

            with mss.mss() as sct:
                idx = min(self._monitor_index, len(sct.monitors) - 1)
                mon = sct.monitors[idx]
                shot = sct.grab(mon)
                img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                self._last_raw_image = img
                arr = np.asarray(img.resize((32, 24), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
                snap = self._snapshot_from_small(arr, (int(shot.width), int(shot.height)), capture_ok=True)
                self._last_salient_xy = snap.salient_xy_norm
                return snap
        except Exception as exc:
            self._last_raw_image = None
            zeros = np.zeros((24, 32, 3), dtype=np.float64)
            return self._snapshot_from_small(zeros, (1920, 1080), capture_ok=False, error=str(exc))

    def capture_salient_zoom_crop(
        self,
        center_xy_norm: Optional[Tuple[float, float]] = None,
        zoom_factor: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Performs a real high-resolution zoom crop around `center_xy_norm` (or latest salient focus)
        and measures genuine pixel sharpness/contrast gain relative to the full-screen view.
        """
        if self._last_raw_image is None:
            snap = self.capture()
            if not snap.capture_ok or self._last_raw_image is None:
                return {"ok": False, "error": snap.error or "No live screen image available"}

        full_stats = _measure_rgb_image_sensory(self._last_raw_image)
        w, h = self._last_raw_image.size
        zf = max(1.2, float(zoom_factor))
        cx, cy = center_xy_norm or self._last_salient_xy
        nx = float(np.clip(float(cx), 0.05, 0.95))
        ny = float(np.clip(float(cy), 0.05, 0.95))

        box_w = max(8, int(round(w / zf)))
        box_h = max(8, int(round(h / zf)))
        left = int(np.clip(round(nx * w - box_w / 2.0), 0, max(0, w - box_w)))
        top = int(np.clip(round(ny * h - box_h / 2.0), 0, max(0, h - box_h)))
        right = min(w, left + box_w)
        bottom = min(h, top + box_h)

        cropped = self._last_raw_image.crop((left, top, right, bottom))
        crop_stats = _measure_rgb_image_sensory(cropped)
        sharp_gain = round(float(crop_stats["edge_sharpness"] - full_stats["edge_sharpness"]), 5)
        detail_revealed = bool(
            crop_stats["edge_sharpness"] > full_stats["edge_sharpness"] * 1.05
            or crop_stats["contrast_std"] > full_stats["contrast_std"] * 1.05
        )
        return {
            "ok": True,
            "crop_box": [left, top, right, bottom],
            "zoom_factor": zf,
            "full_edge_sharpness": full_stats["edge_sharpness"],
            "crop_edge_sharpness": crop_stats["edge_sharpness"],
            "sharpness_gain": sharp_gain,
            "full_contrast_std": full_stats["contrast_std"],
            "crop_contrast_std": crop_stats["contrast_std"],
            "crop_warm_cool_ratio": crop_stats["warm_cool_ratio"],
            "crop_mean_rgb": crop_stats["mean_rgb_01"],
            "detail_revealed": detail_revealed,
            "summary_ko": (
                f"초점({left},{top})-({right},{bottom}) 확대 시 경계선명도 "
                f"{full_stats['edge_sharpness']:.4f}->{crop_stats['edge_sharpness']:.4f}"
                f"(변화 {sharp_gain:+.4f}), 대비 {crop_stats['contrast_std']:.4f}로 실제 픽셀 세부 관찰"
            ),
        }


def _measure_rgb_image_sensory(img: Image.Image) -> Dict[str, Any]:
    """Measures real luminance edge sharpness, contrast, RGB balance, and warmth from an RGB image."""
    rgb_im = img.convert("RGB")
    w, h = rgb_im.size
    arr = np.asarray(rgb_im.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    dx = float(np.mean(np.abs(np.diff(luma, axis=1)))) if luma.shape[1] > 1 else 0.0
    dy = float(np.mean(np.abs(np.diff(luma, axis=0)))) if luma.shape[0] > 1 else 0.0
    return {
        "width": int(w),
        "height": int(h),
        "mean_rgb_01": [round(float(arr[..., c].mean()), 4) for c in range(3)],
        "contrast_std": round(float(arr.std()), 5),
        "warm_cool_ratio": round(float((arr[..., 0].mean() + 0.1) / (arr[..., 2].mean() + 0.1)), 4),
        "edge_sharpness": round(float(dx + dy), 5),
    }


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def inspect_local_file_sensory(path: Path | str) -> Dict[str, Any]:
    """Reads an actual local image or text file through the sensory boundary with honest friction on errors."""
    target = Path(path)
    if not target.exists() or not target.is_file():
        return {"ok": False, "kind": "missing", "path": str(target), "error": f"파일을 찾을 수 없음: {target}"}

    try:
        if target.suffix.lower() in _IMAGE_EXTS:
            with Image.open(target) as im:
                stats = _measure_rgb_image_sensory(im)
            return {
                "ok": True,
                "kind": "image",
                "path": str(target),
                "name": target.name,
                **stats,
                "summary_ko": (
                    f"이미지 '{target.name}'({stats['width']}x{stats['height']}px): "
                    f"경계선명도={stats['edge_sharpness']:.4f}, 대비={stats['contrast_std']:.4f}, "
                    f"온기={stats['warm_cool_ratio']:.3f}"
                ),
            }

        raw_txt = target.read_text(encoding="utf-8", errors="replace")[:3000]
        lines = raw_txt.splitlines()
        preview = " ".join(ln.strip() for ln in lines[:4] if ln.strip())[:220]
        return {
            "ok": True,
            "kind": "text",
            "path": str(target),
            "name": target.name,
            "line_count": len(lines),
            "char_count": len(raw_txt),
            "preview": preview,
            "text_body": raw_txt,
            "summary_ko": f"문서 '{target.name}'({len(lines)}줄, {len(raw_txt)}자): \"{preview[:120]}\"",
        }
    except Exception as exc:
        return {"ok": False, "kind": "error", "path": str(target), "error": str(exc)}


def extract_live_sensory_observables(
    frame_snap: ScreenFrameSnapshot,
    *,
    visual_change: float = 0.0,
    unresolved_friction: float = 0.0,
    order_blue: float = 0.5,
    novelty: float = 0.5,
    harmony: float = 0.5,
    idle_ratio: float = 0.0,
) -> Dict[str, float]:
    """
    Extracts genuine continuous sensory observables directly from live monitor RGB
    pixels (`frame_small`) and real OS/motor response — zero hardcoded semantic dicts.
    """
    rgb = np.asarray(frame_snap.frame_small, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        rgb = np.zeros((24, 32, 3), dtype=np.float64)
    r_ch = rgb[..., 0]
    g_ch = rgb[..., 1]
    b_ch = rgb[..., 2]

    luma = 0.2126 * r_ch + 0.7152 * g_ch + 0.0722 * b_ch
    max_rgb = np.max(rgb[..., :3], axis=2)
    min_rgb = np.min(rgb[..., :3], axis=2)

    dx = float(np.mean(np.abs(np.diff(luma, axis=1)))) if luma.shape[1] > 1 else 0.0
    dy = float(np.mean(np.abs(np.diff(luma, axis=0)))) if luma.shape[0] > 1 else 0.0

    r_mean = float(np.mean(r_ch))
    b_mean = float(np.mean(b_ch))

    return {
        "visual.wavelength": round(float(np.clip(r_mean / (r_mean + b_mean + 1e-6), 0.0, 1.0)), 5),
        "visual.luminance": round(float(np.clip(np.mean(luma), 0.0, 1.0)), 5),
        "visual.gloss": round(float(np.mean(max_rgb > 0.85)), 5),
        "visual.weave_density": round(float(np.clip((dx + dy) * 3.0, 0.0, 1.0)), 5),
        "visual.sheerness": round(float(np.clip(1.0 - float(np.mean(max_rgb - min_rgb)) * 1.4, 0.0, 1.0)), 5),
        "tactile.elasticity": round(float(np.clip(float(visual_change) * 5.0, 0.0, 1.0)), 5),
        "tactile.tension": round(float(np.clip(float(unresolved_friction), 0.0, 1.0)), 5),
        "tactile.tear_stress": round(
            float(np.clip(max(0.0, float(frame_snap.motion_flux) - 0.12) * 3.0 + 0.6 * float(frame_snap.contrast_std), 0.0, 1.0)),
            5,
        ),
        "proprioceptive.balance": round(float(np.clip(order_blue, 0.0, 1.0)), 5),
        "proprioceptive.harmony": round(float(np.clip(harmony, 0.0, 1.0)), 5),
        "proprioceptive.novelty": round(float(np.clip(novelty, 0.0, 1.0)), 5),
        "proprioceptive.load_weight": round(float(np.clip(idle_ratio, 0.0, 1.0)), 5),
        "warm_cool_ratio": round(float(frame_snap.warm_cool_ratio), 5),
        "motion_flux": round(float(frame_snap.motion_flux), 5),
        "contrast_std": round(float(frame_snap.contrast_std), 5),
    }


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
