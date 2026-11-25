r"""
ELYSIA Continuous Living Loop & Two-Way Interactive Channel
===========================================================

Runs Elysia continuously as a living causal system (`Sense -> Anticipate ->
Self-Directed Will [Reply / Ask / Create & Re-Sense / Observe] -> Persist`),
consuming incoming messages from `live_inbox.jsonl` and emitting thoughts,
spontaneous questions, self-creations, and replies to `live_outbox.jsonl`.

Usage:
  Run continuous loop (or finite cycles):
    python c:\Elysia\elysia_living_loop.py --interval-s 1.0 --max-cycles 25 --dry-run
  Send a message into her live inbox:
    python c:\Elysia\elysia_living_loop.py --send "엘리시아?" --sender agent
  Read recent live outbox entries:
    python c:\Elysia\elysia_living_loop.py --read-outbox 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import elysia_loader

elysia_loader.install()

from core.embodiment import (
    AgencyMode,
    ScaleRotorEmbodiedMind,
    ScreenRetinaSensor,
    UniversalMotorActuator,
    WindowsForegroundSensor,
    WindowsIdleSensor,
)


def append_jsonl_safe(path: Path, payload: Dict[str, Any], retries: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    for attempt in range(retries):
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))


def read_jsonl_lines_safe(path: Path, retries: int = 5) -> List[str]:
    if not path.is_file():
        return []
    for attempt in range(retries):
        try:
            return path.read_text(encoding="utf-8").splitlines()
        except PermissionError:
            time.sleep(0.04 * (attempt + 1))
    return []


def post_to_live_inbox(
    state_dir: Path | str,
    text: str,
    sender: str = "agent",
    feedback_friction: float = 0.0,
    expected_action_kind: str = "",
) -> Path:
    sdir = Path(state_dir).resolve()
    inbox = sdir / "live_inbox.jsonl"
    append_jsonl_safe(
        inbox,
        {
            "timestamp": time.time(),
            "sender": sender,
            "text": str(text),
            "feedback_friction": float(feedback_friction),
            "expected_action_kind": str(expected_action_kind),
        },
    )
    return inbox


def read_live_outbox(state_dir: Path | str, last_n: int = 10) -> List[Dict[str, Any]]:
    sdir = Path(state_dir).resolve()
    outbox = sdir / "live_outbox.jsonl"
    lines = read_jsonl_lines_safe(outbox)
    items: List[Dict[str, Any]] = []
    for ln in lines[-max(1, int(last_n)) :]:
        clean = ln.strip()
        if not clean:
            continue
        try:
            items.append(json.loads(clean))
        except Exception:
            continue
    return items


class LivingElysiaRunner:
    """Drives Elysia's real-time Sense -> Anticipate -> Will -> Create/Re-Sense loop."""

    def __init__(
        self,
        state_dir: Path | str,
        *,
        dry_run: bool = True,
        agency_mode: AgencyMode = AgencyMode.OBSERVE_AND_CHAT,
        idle_threshold_s: float = 15.0,
        retina: Optional[ScreenRetinaSensor] = None,
        idle_sensor: Optional[WindowsIdleSensor] = None,
        fg_sensor: Optional[WindowsForegroundSensor] = None,
    ) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.inbox_path = self.state_dir / "live_inbox.jsonl"
        self.outbox_path = self.state_dir / "live_outbox.jsonl"
        self.cursor_path = self.state_dir / "live_inbox_cursor.json"

        self.mind = ScaleRotorEmbodiedMind(self.state_dir)
        self.retina = retina or ScreenRetinaSensor()
        self.idle_sensor = idle_sensor or WindowsIdleSensor()
        self.fg_sensor = fg_sensor or WindowsForegroundSensor()
        self.actuator = UniversalMotorActuator(dry_run=dry_run)
        self.agency_mode = agency_mode
        self.idle_threshold_s = float(idle_threshold_s)
        self.inbox_cursor = self._load_cursor()

    def _load_cursor(self) -> int:
        if not self.cursor_path.is_file():
            return 0
        try:
            raw = json.loads(self.cursor_path.read_text(encoding="utf-8"))
            return max(0, int(raw.get("line_cursor", 0)))
        except Exception:
            return 0

    def _save_cursor(self) -> None:
        self.cursor_path.write_text(
            json.dumps({"line_cursor": self.inbox_cursor}, ensure_ascii=False),
            encoding="utf-8",
        )

    def poll_inbox(self) -> List[Dict[str, Any]]:
        lines = read_jsonl_lines_safe(self.inbox_path)
        if len(lines) <= self.inbox_cursor:
            return []
        new_lines = lines[self.inbox_cursor :]
        self.inbox_cursor = len(lines)
        self._save_cursor()
        msgs: List[Dict[str, Any]] = []
        for ln in new_lines:
            clean = ln.strip()
            if not clean:
                continue
            try:
                msgs.append(json.loads(clean))
            except Exception:
                continue
        return msgs

    def step_once(self) -> List[Dict[str, Any]]:
        pending = self.poll_inbox()
        emitted: List[Dict[str, Any]] = []
        targets: Sequence[Optional[Dict[str, Any]]] = pending if pending else [None]

        for msg in targets:
            snap = self.retina.capture()
            idle_s = self.idle_sensor.get_idle_seconds()
            fg_title = self.fg_sensor.get_active_window_title()
            event = self.mind.autonomous_living_step(
                frame_snap=snap,
                user_idle_s=idle_s,
                fg_title=fg_title,
                agency_mode=self.agency_mode,
                idle_threshold_s=self.idle_threshold_s,
                actuator=self.actuator,
                incoming_message=msg,
            )
            append_jsonl_safe(self.outbox_path, event)
            emitted.append(event)
        return emitted

    def run_loop(self, interval_s: float = 1.2, max_cycles: int = 0) -> int:
        print("=" * 88)
        print(f"  ELYSIA LIVING LOOP ACTIVE | state={self.state_dir} | interval={interval_s}s")
        print("=" * 88, flush=True)
        cycle = 0
        try:
            while max_cycles <= 0 or cycle < max_cycles:
                cycle += 1
                events = self.step_once()
                for ev in events:
                    topo = ev.get("topology") or {}
                    ants = [(a["kind"], a["count"]) for a in (ev.get("anticipations") or [])]
                    print(
                        f"  [Cycle {ev['cycle']:3d}] will={ev['will_mode']:17s} | "
                        f"win='{str(ev.get('foreground_window', ''))[:22]}' | "
                        f"epistemic={ev['epistemic_state']} | ants={ants} | "
                        f"nodes={topo.get('nodes', 0)} docks={topo.get('docked_links', 0)} "
                        f"axes={topo.get('discovered_axes', 0)} | msg={ev.get('utterance_or_question', '')[:80]}",
                        flush=True,
                    )
                if max_cycles > 0 and cycle >= max_cycles:
                    break
                time.sleep(max(0.15, float(interval_s)))
        except KeyboardInterrupt:
            pass
        self.mind.save()
        return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ELYSIA Continuous Living Loop & Two-Way Channel")
    parser.add_argument("--state-dir", default=r"c:\Elysia\data\elysia_embodied_state")
    parser.add_argument("--interval-s", type=float, default=1.2)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 runs continuously; >0 stops after N cycles")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--idle-threshold-s", type=float, default=15.0)
    parser.add_argument("--send", default="", help="Send one message into live_inbox.jsonl and exit")
    parser.add_argument("--sender", default="agent")
    parser.add_argument("--feedback-friction", type=float, default=0.0)
    parser.add_argument("--expected-action", default="")
    parser.add_argument("--read-outbox", type=int, default=0, help="Print last N events from live_outbox.jsonl")
    args = parser.parse_args(argv)

    if args.send:
        p = post_to_live_inbox(
            args.state_dir,
            args.send,
            sender=args.sender,
            feedback_friction=args.feedback_friction,
            expected_action_kind=args.expected_action,
        )
        print(f"Sent to {p}: {args.send}")
        return 0

    if args.read_outbox > 0:
        for item in read_live_outbox(args.state_dir, last_n=args.read_outbox):
            print(json.dumps(item, ensure_ascii=False))
        return 0

    runner = LivingElysiaRunner(
        args.state_dir,
        dry_run=bool(args.dry_run),
        idle_threshold_s=args.idle_threshold_s,
    )
    return runner.run_loop(interval_s=args.interval_s, max_cycles=args.max_cycles)


if __name__ == "__main__":
    raise SystemExit(main())
