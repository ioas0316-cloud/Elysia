# [Genesis: 2025-12-02] Purified by Elysia
#!/usr/bin/env python
"""WebSocket bridge that streams persona frames in real time."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.append(str(root))

from scripts.persona_hooks.persona_stream import collect_persona_event, DEFAULT_OUTPUT  # type: ignore


async def persona_stream_handler(websocket, path, *, persona_key: str, interval: float):
    del path  # unused
    print(f"[ws_persona_server] client connected for {persona_key}")
    try:
        while True:
            event = collect_persona_event(persona_key)
            await websocket.send(json.dumps(event, ensure_ascii=False))
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[ws_persona_server] client error: {exc}")
    finally:
        print("[ws_persona_server] client disconnected")


async def main_async(persona_key: str, port: int, interval: float) -> None:
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "The 'websockets' package is required. Install via 'pip install websockets'."
        ) from exc

    server = await websockets.serve(
        lambda ws, path: persona_stream_handler(ws, path, persona_key=persona_key, interval=interval),
        "127.0.0.1",
        port,
    )
    print(f"[ws_persona_server] running on ws://127.0.0.1:{port} (persona={persona_key})")
    try:
        await asyncio.Future()
    finally:
        server.close()
        await server.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve persona frames over WebSocket.")
    parser.add_argument("--persona", required=True, help="Persona key (e.g., elysia.dancer)")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port (default 8765)")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between frames (default 0.2)",
    )
    args = parser.parse_args()

    # ensure log directory exists (mirrors persona_stream behavior)
    Path(DEFAULT_OUTPUT).parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(main_async(args.persona, args.port, args.interval))


if __name__ == "__main__":
    main()