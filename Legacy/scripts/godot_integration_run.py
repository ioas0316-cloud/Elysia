from __future__ import annotations

import os
import sys
import time
from typing import Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Project_Elysia.high_engine.godot_integration import GodotIntegration


def _parse_command(command: str) -> Optional[dict]:
    command = command.strip()
    if not command:
        return None
    if command.startswith("chat:"):
        return {"type": "input", "input_type": "chat", "text": command[len("chat:"):].strip()}
    if command.startswith("vision:"):
        _, payload = command.split(":", 1)
        parts = [p.strip() for p in payload.split("|")]
        desc = parts[0]
        palette = parts[1].split(",") if len(parts) > 1 and parts[1] else []
        brightness = float(parts[2]) if len(parts) > 2 else 0.5
        return {
            "type": "input",
            "input_type": "vision",
            "description": desc,
            "palette": palette,
            "brightness": brightness,
        }
    if command.startswith("exit") or command.startswith("quit"):
        return {"type": "exit"}
    return None


def run_loop():
    integration = GodotIntegration()
    init = integration.get_initial_frame()
    print("Initial frame received. World width:", init.get("world", {}).get("width"))
    try:
        while True:
            frame = integration.next_frame()
            elysia = frame.get("elysia", {})
            world = frame.get("world", {})
            print(f"\nTick {frame.get('tick')} | Mood: {elysia.get('mood')} | Focus: {elysia.get('primary_focus')}")
            print("  Thought:", elysia.get("thought_trail"))
            print("  Meta:", elysia.get("meta_observation"))
            print("  Focus area:", elysia.get("focus_area"))
            print("  World width:", world.get("width"))
            try:
                cmd = input("command (chat:..., vision:desc|red,blue|0.6, exit): ")
            except EOFError:
                print("\n입력이 끊겼습니다. 종료합니다.")
                break
            parsed = _parse_command(cmd)
            if parsed is None:
                continue
            if parsed.get("type") == "exit":
                print("Shutting down integration loop.")
                break
            integration.process_input(parsed)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted; exiting.")


if __name__ == "__main__":
    run_loop()
