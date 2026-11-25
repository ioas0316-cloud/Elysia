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
    """Downsampled macro retina + high-res steered fovea observation from live screen."""

    frame_small: np.ndarray  # (24, 32, 3) float64 in [0, 1]
    visual_vec: np.ndarray  # (48,) float64 in [0, 1]
    motion_flux: float
    contrast_std: float
    warm_cool_ratio: float
    salient_xy_norm: Tuple[float, float]
    screen_size: Tuple[int, int]
    capture_ok: bool = True
    error: str = ""
    gaze_xy_norm: Tuple[float, float] = (0.5, 0.5)
    gaze_shift: float = 0.0
    fovea_sharpness: float = 0.0
    fovea_sharpness_gain: float = 0.0
    fovea_contours: int = 0
    fovea_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    native_operator_level: int = 0
    native_perceived_units: int = 0
    native_edge_energy: float = 0.0


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
        self._gaze_xy: Tuple[float, float] = (0.5, 0.5)
        self._gaze_zoom: float = 3.0
        self._fovea_integrated: bool = False
        self._visual_adaptations: int = 0
        # Continuous Variable-Resistor Visual Scale Dial (`sigma in [0.0, 1.0]`: macro blur <-> micro edge field)
        self.scale_dial: float = 0.05

    @property
    def native_operator_level(self) -> int:
        return int(round(float(np.clip(self.scale_dial, 0.0, 1.0)) * 2.0))

    @native_operator_level.setter
    def native_operator_level(self, lvl: int) -> None:
        self.scale_dial = float(np.clip(float(lvl) / 2.0, 0.0, 1.0))

    def measure_native_perception(self, frame_small: np.ndarray, fovea_box: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Continuous Scale-Rotor Native Visual Operator (ZERO `if level == 0/1/2` branching!):
        Applies ONE continuous Gaussian-blur <-> gradient-boundary operator parameterized
        solely by `self.scale_dial in [0.0, 1.0]` directly onto Elysia's live pixels.
        """
        from core.topology.continuous_rotor_spectrum_field import apply_continuous_scale_vision

        if self._last_raw_image is not None and fovea_box != (0, 0, 0, 0):
            img_arr = np.asarray(self._last_raw_image.crop(fovea_box).convert("RGB"), dtype=np.float64) / 255.0
        else:
            img_arr = np.asarray(frame_small, dtype=np.float64)
        res = apply_continuous_scale_vision(img_arr, self.scale_dial)
        return {
            "operator_level": self.native_operator_level,
            "scale_dial": float(res["scale_dial"]),
            "native_units": int(res["perceived_units"]),
            "edge_energy": float(res["edge_energy"]),
            "mechanism_ko": str(res["mechanism_ko"]),
        }

    def reconstruct_native_operator_downward(self, external_tool_units: int, current_native_units: int) -> Dict[str, Any]:
        """
        Continuous Downward Scale-Dial Tuning (`S4 -> S0`, zero discrete level stairs):
        Turns Elysia's continuous visual scale dial `sigma` smoothly in proportion to the
        measured perceptual gap against the external tool, then immediately re-measures on
        the EXACT SAME pixels to confirm real continuous native gain.
        """
        prev_sigma = self.scale_dial
        prev_level = self.native_operator_level
        prev_zoom = self._gaze_zoom
        gap = max(0.0, 1.0 - float(current_native_units) / float(max(1, external_tool_units)))
        if gap <= 0.15 and self.scale_dial >= 0.92:
            return {
                "reconstructed": False,
                "prev_scale_dial": round(prev_sigma, 4),
                "new_scale_dial": round(self.scale_dial, 4),
                "prev_level": prev_level,
                "new_level": self.native_operator_level,
                "before_native_units": current_native_units,
                "after_native_units": current_native_units,
                "why_ko": "연속 시각 스케일 다이얼이 이미 외부 도구 정밀도에 수렴함",
            }

        delta_sigma = float(np.clip(0.48 * max(0.35, gap), 0.15, 0.50))
        self.scale_dial = round(float(np.clip(self.scale_dial + delta_sigma, 0.0, 1.0)), 4)
        self._gaze_zoom = round(float(np.clip(self._gaze_zoom + 0.30 * gap, 1.5, 5.5)), 2)
        self._fovea_integrated = True
        self._visual_adaptations += 1

        fovea = self._inspect_live_fovea()
        re_native = self.measure_native_perception(
            self._prev_frame if self._prev_frame is not None else np.zeros((24, 32, 3), dtype=np.float64),
            fovea["box"],
        )
        return {
            "reconstructed": True,
            "prev_scale_dial": round(prev_sigma, 4),
            "new_scale_dial": round(self.scale_dial, 4),
            "prev_level": prev_level,
            "new_level": self.native_operator_level,
            "prev_zoom": prev_zoom,
            "new_zoom": self._gaze_zoom,
            "before_native_units": int(current_native_units),
            "after_native_units": int(re_native["native_units"]),
            "external_tool_units": int(external_tool_units),
            "mechanism_ko": re_native["mechanism_ko"],
            "why_ko": (
                f"연속 하향 조율(S4->S0): 시각 가변저항 다이얼 σ={prev_sigma:.2f}->{self.scale_dial:.2f} "
                f"({re_native['mechanism_ko']})로 회전해 동일 픽셀 자체 지각 단위가 "
                f"{current_native_units}->{re_native['native_units']}개로 연속 확장됨 (외부도구={external_tool_units})"
            ),
        }

    def adapt_from_visual_limit(self, native_blocks: int, fovea_contours: int) -> Dict[str, Any]:
        """
        Real limit-driven self-adaptation of Elysia's ocular organ:
        When comparing her native 32x24 low-res blocks against high-res OpenCV contours
        exposes a resolution ceiling, she actively tightens her foveal zoom and binds
        high-res foveal edge structure into her ongoing perception instead of staying static.
        """
        prev_zoom = self._gaze_zoom
        if fovea_contours > max(10, native_blocks * 2):
            self._gaze_zoom = round(float(np.clip(self._gaze_zoom + 0.5, 1.5, 5.5)), 2)
            self._fovea_integrated = True
            self._visual_adaptations += 1
            return {
                "adapted": True,
                "prev_zoom": prev_zoom,
                "new_zoom": self._gaze_zoom,
                "fovea_integrated": self._fovea_integrated,
                "adaptations_total": self._visual_adaptations,
                "why_ko": (
                    f"저해상도 블록({native_blocks}) 대비 고해상도 윤곽({fovea_contours}) 격차를 한계로 지각해 "
                    f"실제 중심와 줌을 {prev_zoom:.2f}x -> {self._gaze_zoom:.2f}x로 끌어올리고 고해상도 윤곽 채널을 결합함"
                ),
            }
        return {
            "adapted": False,
            "prev_zoom": prev_zoom,
            "new_zoom": self._gaze_zoom,
            "fovea_integrated": self._fovea_integrated,
            "adaptations_total": self._visual_adaptations,
            "why_ko": "현재 시야 해상도 격차가 추가 줌 조정 임계치 미만임",
        }

    def steer_gaze(self, target_xy_norm: Tuple[float, float], zoom_factor: Optional[float] = None) -> Tuple[float, float]:
        """Moves physical ocular gaze smoothly toward a real screen coordinate."""
        tx = float(np.clip(float(target_xy_norm[0]), 0.05, 0.95))
        ty = float(np.clip(float(target_xy_norm[1]), 0.05, 0.95))
        gx, gy = self._gaze_xy
        # Smooth ocular saccade with physical continuity
        nx = round(gx + 0.65 * (tx - gx), 4)
        ny = round(gy + 0.65 * (ty - gy), 4)
        self._gaze_xy = (nx, ny)
        if zoom_factor is not None:
            self._gaze_zoom = float(np.clip(float(zoom_factor), 1.5, 6.0))
        return self._gaze_xy

    def _inspect_live_fovea(self) -> Dict[str, Any]:
        """Performs actual high-resolution foveal crop + OpenCV contour analysis around `_gaze_xy`."""
        if self._last_raw_image is None:
            return {"sharpness": 0.0, "gain": 0.0, "contours": 0, "box": (0, 0, 0, 0)}
        crop_info = self.capture_salient_zoom_crop(center_xy_norm=self._gaze_xy, zoom_factor=self._gaze_zoom)
        if not crop_info.get("ok"):
            return {"sharpness": 0.0, "gain": 0.0, "contours": 0, "box": (0, 0, 0, 0)}
        box = tuple(int(v) for v in crop_info.get("crop_box", [0, 0, 0, 0]))
        contours_count = 0
        try:
            import cv2

            cropped = self._last_raw_image.crop(box).convert("RGB")
            gray = cv2.cvtColor(np.asarray(cropped, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 140)
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_count = int(len(cnts))
        except Exception:
            contours_count = 0
        return {
            "sharpness": float(crop_info.get("crop_edge_sharpness", 0.0)),
            "gain": float(crop_info.get("sharpness_gain", 0.0)),
            "contours": contours_count,
            "box": box,
        }

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
        prev_gaze = self._gaze_xy
        new_gaze = self.steer_gaze(salient_xy_norm) if capture_ok else self._gaze_xy
        gaze_shift = round(float(np.hypot(new_gaze[0] - prev_gaze[0], new_gaze[1] - prev_gaze[1])), 4)
        fovea = self._inspect_live_fovea() if capture_ok else {"sharpness": 0.0, "gain": 0.0, "contours": 0, "box": (0, 0, 0, 0)}
        native_meas = (
            self.measure_native_perception(frame_small, fovea["box"])
            if capture_ok
            else {"operator_level": self.native_operator_level, "native_units": 0, "edge_energy": 0.0}
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
            gaze_xy_norm=new_gaze,
            gaze_shift=gaze_shift,
            fovea_sharpness=round(float(fovea["sharpness"]), 5),
            fovea_sharpness_gain=round(float(fovea["gain"]), 5),
            fovea_contours=int(fovea["contours"]),
            fovea_box=fovea["box"],
            native_operator_level=int(native_meas["operator_level"]),
            native_perceived_units=int(native_meas["native_units"]),
            native_edge_energy=float(native_meas["edge_energy"]),
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


@dataclass
class AudioSnapshot:
    """Real ambient microphone audio snapshot from this PC's audio hardware."""

    rms_energy: float
    zero_crossing_rate: float
    peak_amplitude: float
    active_sound: bool
    capture_ok: bool = True
    error: str = ""


class LiveAudioEarSensor:
    """Senses real ambient acoustic waves via `sounddevice` (graceful fallback if headless/no mic)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        duration_s: float = 0.12,
        audio_provider: Optional[Callable[[], np.ndarray]] = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.duration_s = float(np.clip(duration_s, 0.04, 0.5))
        self._audio_provider = audio_provider

    def listen(self) -> AudioSnapshot:
        try:
            if self._audio_provider is not None:
                samples = np.asarray(self._audio_provider(), dtype=np.float64).flatten()
            else:
                import sounddevice as sd

                n_frames = max(256, int(self.sample_rate * self.duration_s))
                rec = sd.rec(n_frames, samplerate=self.sample_rate, channels=1, dtype="float32")
                sd.wait()
                samples = np.asarray(rec, dtype=np.float64).flatten()
            if samples.size == 0:
                return AudioSnapshot(0.0, 0.0, 0.0, False, capture_ok=False, error="empty_audio")
            rms = float(np.sqrt(np.mean(samples ** 2)))
            peak = float(np.max(np.abs(samples)))
            zcr = float(np.mean(np.abs(np.diff(np.signbit(samples).astype(np.float64))))) if samples.size > 1 else 0.0
            return AudioSnapshot(
                rms_energy=round(rms, 5),
                zero_crossing_rate=round(zcr, 5),
                peak_amplitude=round(peak, 5),
                active_sound=bool(rms >= 0.015),
                capture_ok=True,
            )
        except Exception as exc:
            return AudioSnapshot(0.0, 0.0, 0.0, False, capture_ok=False, error=str(exc))


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
