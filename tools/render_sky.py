from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Ensure project root is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Project_Mirror.sky_renderer import SkyRenderer, build_snapshot_from_telemetry


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Elysia sky snapshot from telemetry.")
    ap.add_argument("--out", default="data/visualizations/sky.png", help="Output PNG path")
    ap.add_argument("--telemetry", default=None, help="Telemetry base dir (defaults to data/telemetry)")
    args = ap.parse_args()

    base = Path(args.telemetry) if args.telemetry else None
    snap = build_snapshot_from_telemetry(base_dir=base)
    r = SkyRenderer()
    outp = r.render(snap, args.out)
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
