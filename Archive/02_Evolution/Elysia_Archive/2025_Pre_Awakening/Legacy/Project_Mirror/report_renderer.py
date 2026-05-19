"""
ReportRenderer

Renders a simple daily summary card (PNG) with key artifacts and metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
from PIL import Image, ImageDraw, ImageFont


class ReportRenderer:
    def __init__(self, output_dir: str = "data/reports/daily"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.font = ImageFont.load_default()
        except Exception:
            self.font = None

    def render_daily_card(self, date_str: str, info: Dict[str, str], filename: Optional[str] = None) -> str:
        lines = [
            f"Elysia Daily — {date_str}",
            "",
        ]
        for k, v in info.items():
            lines.append(f"{k}: {v}")

        width = 1000
        line_height = 22
        height = max(200, (len(lines) + 2) * line_height + 20)
        img = Image.new("RGB", (width, height), (250, 252, 255))
        d = ImageDraw.Draw(img)
        y = 12
        for line in lines:
            d.text((14, y), line, fill=(15, 20, 30), font=self.font)
            y += line_height

        if not filename:
            filename = f"daily_{date_str}.png"
        out = self.output_dir / filename
        img.save(out)
        return str(out)

