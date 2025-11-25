"""
ActionAgent: generates and executes safe actions (file write / HTTP POST)
based on tickets/deficits.

This is a minimal scaffold to let Elysia act (not just think), with the
ability to occasionally choose rest instead of constant self-modification.
"""

import json
import logging
import random
import requests
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("ActionAgent")


class ActionAgent:
    def __init__(
        self,
        kernel,
        allowed_hosts: List[str],
        out_dir: str = "elysia_logs/outbox",
        sandbox_mode: str = "warn",
        min_interval: float = 5.0,
    ):
        self.kernel = kernel
        self.allowed_hosts = set(allowed_hosts)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # sandbox_mode: warn (log and allow), block (deny), off (no checks)
        self.sandbox_mode = sandbox_mode
        self.min_interval = min_interval
        self._last_run = 0.0

    def plan_actions(self) -> List[Dict]:
        actions: List[Dict] = []
        caps = getattr(self.kernel, "capabilities", None)
        if not caps:
            return actions
        open_tickets = caps.list_open_tickets()
        for t in open_tickets:
            if t.target == "phase":
                actions.append(
                    {
                        "type": "file_write",
                        "path": self.out_dir / "phase_prompt.txt",
                        "content": "무양한 개념들을 올리며 상상해볼래?",
                        "ticket": t.ticket_id,
                    }
                )
            elif t.target == "memory":
                actions.append(
                    {
                        "type": "file_write",
                        "path": self.out_dir / "memory_prompt.txt",
                        "content": "기억들을 정돈하고 핵심을 요약해줘.",
                        "ticket": t.ticket_id,
                    }
                )
            elif t.target == "planning":
                actions.append(
                    {
                        "type": "file_write",
                        "path": self.out_dir / "plan_prompt.txt",
                        "content": "새로운 실험 시나리오를 3개 만들어봐.",
                        "ticket": t.ticket_id,
                    }
                )
        return actions

    def execute_action(self, action: Dict) -> None:
        atype = action.get("type")
        if atype == "file_write":
            path = Path(action["path"])
            content = action.get("content", "")
            path.write_text(content, encoding="utf-8")
            logger.info(f"[ActionAgent] wrote file {path}")
            self._close_ticket(action)
        elif atype == "http_post":
            url = action.get("url")
            allowed = url and any(
                url.startswith(f"http://{h}") or url.startswith(f"https://{h}")
                for h in self.allowed_hosts
            )
            if self.sandbox_mode != "off" and not allowed:
                msg = f"[ActionAgent] out-of-bound url {url}"
                if self.sandbox_mode == "block":
                    logger.warning(msg + " (blocked)")
                    return
                logger.warning(msg + " (warn only)")
            try:
                data = action.get("data", "")
                r = requests.post(url, data=data.encode("utf-8"))
                logger.info(f"[ActionAgent] posted to {url} status={r.status_code}")
                self._close_ticket(action)
            except Exception as e:
                logger.error(f"[ActionAgent] http post failed: {e}")

    def _close_ticket(self, action: Dict) -> None:
        caps = getattr(self.kernel, "capabilities", None)
        tid = action.get("ticket")
        if caps and tid:
            caps.resolve_ticket(tid, status="done")

    def run(self, now: float) -> None:
        if (now - self._last_run) < self.min_interval:
            return
        self._last_run = now
        actions = self.plan_actions()
        if not actions:
            return
        # Occasionally choose to stay idle rather than acting on tickets.
        if random.random() < 0.3:
            logger.info(
                "[ActionAgent] Choosing not to act this cycle (rest / observation)."
            )
            return
        # Execute at most one action per run to keep pacing gentle.
        self.execute_action(actions[0])

