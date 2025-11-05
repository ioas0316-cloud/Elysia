from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SkyStar:
    x: float  # 0..1
    y: float  # 0..1
    radius: float  # px scale after canvas size applied
    brightness: float  # 0..1


@dataclass
class SkySnapshot:
    width: int = 1280
    height: int = 720
    stars: List[SkyStar] = None
    cloudiness: float = 0.0  # 0..1
    rain_intensity: float = 0.0  # 0..1
    sun_strength: float = 0.3  # 0..1
    rainbow: bool = False


class SkyRenderer:
    """Renders a symbolic night-sky style snapshot mapping internal signals to visuals.

    Purpose: Provide a gentle, beautiful visual reflection of Elysia's inner state.
    Role: Part of Project_Mirror. Accepts a SkySnapshot and produces a PNG.
    """

    def __init__(self) -> None:
        try:
            from PIL import Image, ImageDraw  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Pillow (PIL) is required to render images. Install with 'pip install Pillow'."
            ) from e

    def render(self, snapshot: SkySnapshot, out_path: str | os.PathLike) -> Path:
        from PIL import Image, ImageDraw

        w, h = snapshot.width, snapshot.height
        img = Image.new("RGBA", (w, h), (10, 12, 24, 255))
        draw = ImageDraw.Draw(img, "RGBA")

        # Background vertical gradient (night → dawn by sun_strength)
        self._draw_gradient(draw, w, h, top=(8, 10, 22), bottom=(20 + int(120 * snapshot.sun_strength), 30 + int(90 * snapshot.sun_strength), 60 + int(60 * snapshot.sun_strength)))

        # Sun glow (soft radial from bottom-right)
        if snapshot.sun_strength > 0:
            self._draw_sun(img, cx=int(w * 0.82), cy=int(h * 0.85), base_radius=int(min(w, h) * 0.35), strength=snapshot.sun_strength)

        # Clouds (soft translucent ellipses)
        if snapshot.cloudiness > 0:
            self._draw_clouds(draw, w, h, snapshot.cloudiness)

        # Stars (larger first)
        for star in sorted(snapshot.stars or [], key=lambda s: s.radius, reverse=True):
            self._draw_star(draw, w, h, star)

        # Rain (lines)
        if snapshot.rain_intensity > 0:
            self._draw_rain(draw, w, h, snapshot.rain_intensity)

        # Rainbow (arc)
        if snapshot.rainbow:
            self._draw_rainbow(draw, w, h)

        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        img.save(outp, format="PNG")
        return outp

    # ---------------- helpers ----------------
    def _draw_gradient(self, draw, w: int, h: int, top: Tuple[int, int, int], bottom: Tuple[int, int, int]) -> None:
        for y in range(h):
            t = y / max(1, h - 1)
            r = int(top[0] * (1 - t) + bottom[0] * t)
            g = int(top[1] * (1 - t) + bottom[1] * t)
            b = int(top[2] * (1 - t) + bottom[2] * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b, 255))

    def _draw_star(self, draw, w: int, h: int, s: SkyStar) -> None:
        x = int(s.x * w)
        y = int(s.y * h)
        r = max(1, int(s.radius))
        a = int(180 + 75 * s.brightness)
        # soft glow: outer halo + core
        for i in range(3, 0, -1):
            rr = r + i * 2
            alpha = max(20, int(a * (0.2 * i)))
            draw.ellipse([(x - rr, y - rr), (x + rr, y + rr)], fill=(255, 255, 255, alpha))
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(255, 255, 255, a))

    def _draw_sun(self, img, cx: int, cy: int, base_radius: int, strength: float) -> None:
        from PIL import Image, ImageDraw

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay, "RGBA")
        steps = 12
        for i in range(steps):
            t = i / steps
            r = int(base_radius * (0.2 + 0.8 * t))
            alpha = int(8 + 150 * (1 - t) * strength)
            color = (255, int(220 - 80 * t), int(120 - 100 * t), alpha)
            d.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=color)
        img.alpha_composite(overlay)

    def _draw_clouds(self, draw, w: int, h: int, density: float) -> None:
        n = int(8 + 24 * density)
        for _ in range(n):
            cx = random.randint(int(w * 0.05), int(w * 0.95))
            cy = random.randint(int(h * 0.05), int(h * 0.6))
            rx = random.randint(int(w * 0.06), int(w * 0.18))
            ry = random.randint(int(h * 0.03), int(h * 0.10))
            a = int(30 + 120 * density)
            draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], fill=(220, 230, 240, a))

    def _draw_rain(self, draw, w: int, h: int, intensity: float) -> None:
        n = int(50 + 600 * intensity)
        for _ in range(n):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            l = random.randint(8, 18)
            draw.line([(x, y), (x + 3, y + l)], fill=(180, 210, 255, 140), width=1)

    def _draw_rainbow(self, draw, w: int, h: int) -> None:
        cx, cy = int(w * 0.75), int(h * 0.82)
        outer = int(min(w, h) * 0.7)
        for i, col in enumerate([(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 127, 255), (0, 0, 255), (127, 0, 255)]):
            bb = [cx - outer + i * 8, cy - outer + i * 8, cx + outer - i * 8, cy + outer - i * 8]
            draw.arc(bb, start=200, end=340, fill=col + (180,), width=6)


def build_snapshot_from_telemetry(base_dir: Optional[Path] = None, max_events: int = 5000) -> SkySnapshot:
    """Construct a SkySnapshot by reading recent telemetry events.

    Mapping (heuristic, non-intrusive):
    - echo_updated.top_nodes -> bright, larger stars
    - route.arc(latency)     -> star brightness distribution
    - fs.index/count         -> star count baseline
    - config.warn / errors   -> cloudiness
    - wh.extract/promote     -> rain & rainbow hints
    """
    project_root = Path(__file__).resolve().parents[1]
    telem_base = base_dir or (project_root / "data" / "telemetry")
    # pick latest day dir
    if not telem_base.exists():
        return SkySnapshot(stars=_fallback_stars())
    day_dirs = [d for d in telem_base.iterdir() if d.is_dir() and d.name.isdigit()]
    if not day_dirs:
        return SkySnapshot(stars=_fallback_stars())
    latest = sorted(day_dirs)[-1]
    events_path = latest / "events.jsonl"
    if not events_path.exists():
        return SkySnapshot(stars=_fallback_stars())

    stars: List[SkyStar] = []
    cloudiness = 0.0
    rain = 0.0
    rainbow = False
    sun = 0.35

    top_nodes: List[Tuple[str, float]] = []
    latencies: List[float] = []
    fs_count = 0
    warns = 0
    extracts = 0
    promotes = 0

    # read last N events
    lines = []
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                lines.append(line)
        lines = lines[-max_events:]
    except Exception:
        return SkySnapshot(stars=_fallback_stars())

    for line in lines:
        try:
            evt = json.loads(line)
        except Exception:
            continue
        t = evt.get("event_type")
        p = evt.get("payload", {})
        if t == "echo_updated":
            # collect top nodes as stars
            for node in p.get("top_nodes", [])[:8]:
                e = float(node.get("e", 0.0))
                top_nodes.append((str(node.get("id", "?")), e))
        elif t == "route.arc":
            lat = float(p.get("latency_ms", 0.0))
            latencies.append(lat)
        elif t in ("fs.index",):
            fs_count += int(p.get("count", 0))
        elif t in ("config.warn", "fs.index.error"):
            warns += 1
        elif t in ("wh.extract",):
            extracts += 1
        elif t in ("wh.promote",):
            promotes += 1

    # Stars from top nodes
    rng = random.Random("Elysia")
    for i, (_, energy) in enumerate(top_nodes[:24]):
        stars.append(
            SkyStar(
                x=rng.random() * 0.9 + 0.05,
                y=rng.random() * 0.7 + 0.05,
                radius=3.0 + 6.0 * _sigmoid(energy),
                brightness=0.6 + 0.4 * _sigmoid(energy),
            )
        )

    # Baseline stars scaled by fs_count and routes
    baseline = max(40, min(250, int(20 + 0.02 * fs_count + 0.2 * len(latencies))))
    for _ in range(baseline):
        stars.append(
            SkyStar(
                x=rng.random(),
                y=rng.random(),
                radius=1.0 + 2.5 * rng.random(),
                brightness=0.4 + 0.6 * rng.random(),
            )
        )

    # Clouds & rain
    cloudiness = max(0.0, min(1.0, 0.15 * warns))
    rain = max(0.0, min(1.0, 0.08 * extracts))
    rainbow = promotes > 0 and rain > 0.1

    return SkySnapshot(stars=stars, cloudiness=cloudiness, rain_intensity=rain, sun_strength=sun, rainbow=bool(rainbow))


def _fallback_stars() -> List[SkyStar]:
    rng = random.Random(42)
    return [SkyStar(x=rng.random(), y=rng.random(), radius=1.0 + 2.0 * rng.random(), brightness=0.5 + 0.5 * rng.random()) for _ in range(120)]


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5
