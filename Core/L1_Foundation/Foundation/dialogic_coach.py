"""
DialogicCoach

Purpose: Support truly dialogic, mixed initiative conversation by
- reflecting (active listening),
- asking clarifying/value questions,
- inviting mutual knowing (Elysia also shares briefly),
- and selecting a gentle follow up move.

This module is API free and heuristic by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DialogMove:
    kind: str  # 'reflect' | 'clarify' | 'deepen' | 'preference' | 'value'
    text: str


class DialogicCoach:
    def __init__(self):
        pass

    def suggest_followup(self, user_message: str, context: Dict) -> Optional[DialogMove]:
        msg = (user_message or "").strip()
        echo = context.get("echo", {}) or {}
        identity = context.get("identity", {}) or {}

        # If the message is short/ambiguous, prefer clarification
        if len(msg) <= 3 or msg.endswith("?"):
            return DialogMove(
                kind="clarify",
                text="                     .                        ?"
            )

        # If there is rich echo (many active concepts), invite focus
        if len(echo) >= 6:
            return DialogMove(
                kind="deepen",
                text="                  .                      ?"
            )

        # Value oriented invitation sometimes
        if any(k in msg for k in ["  ", "  ", "  ", "  ", "  "]):
            return DialogMove(
                kind="value",
                text="                            ?"
            )

        # Default: brief reflection + open question
        reflection = self._reflect(msg)
        return DialogMove(
            kind="reflect",
            text=f"               : {reflection}                ?"
        )

    def _reflect(self, msg: str) -> str:
        # Minimal pragmatic reflection: keep first sentence, soften
        s = msg.splitlines()[0]
        if len(s) > 120:
            s = s[:120] + " "
        return f"'{s}'         "
