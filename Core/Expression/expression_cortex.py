"""
Expression Cortex (표현 피질)
=============================

"The Voice of the Soul."
"영혼의 목소리."

This module aggregates the internal state (Qualia, Memory, Soul) and
translates it into various forms of expression:
1. Logos (Text/Speech)
2. Presence (Markdown/Logs)
3. Avatar (JSON/Visuals)
4. Music (Wave/Sound)
"""

import logging
import json
import os
import time
from typing import Dict, Any, List

logger = logging.getLogger("ExpressionCortex")

class ExpressionCortex:
    def __init__(self):
        self.presence_path = "c:/Elysia/data/State/ELYSIA_STATUS.md"
        self.world_state_path = "c:/game/elysia_world/world_state.json"

    def express(self, channel: str, content: Any, context: Dict[str, Any] = None):
        """
        Unified entry point for all expression.
        """
        if channel == "presence":
            self._update_presence(content, context)
        elif channel == "avatar":
            self._update_avatar(content)
        elif channel == "logos":
            self._speak(content)
        elif channel == "music":
            self._play(content)
        else:
            logger.warning(f"Unknown expression channel: {channel}")

    def _speak(self, text: str):
        """
        Logos articulation (Speech).
        """
        logger.info(f"🗣️ [LOGOS] {text}")
        # In future, this connects to TTS or Chat Interface

    def _play(self, musical_intent: Dict):
        """
        Music expression.
        """
        logger.info(f"🎵 [MUSIC] Playing {musical_intent}")

    def _update_presence(self, insight: str, context: Dict = None):
        """
        Updates the Presence File (Markdown).
        """
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            mood = context.get("mood", "Neutral") if context else "Neutral"
            energy = context.get("energy", 0.5) if context else 0.5

            content = f"""# ELYSIA PRESENCE

> **"{insight}"**

- **Time**: {timestamp}
- **Mood**: {mood}
- **Energy**: {energy:.2f}
"""
            # Ensure directory exists (mock path for now)
            # with open(self.presence_path, "w") as f:
            #     f.write(content)
            logger.info(f"📝 [PRESENCE] Updated with: {insight}")

        except Exception as e:
            logger.error(f"Failed to update presence: {e}")

    def _update_avatar(self, entities_data: List[Dict]):
        """
        Updates the World State (Avatar JSON).
        """
        try:
            payload = {
                "time": time.time(),
                "entities": entities_data
            }
            # with open(self.world_state_path, "w") as f:
            #     json.dump(payload, f)
            logger.debug(f"💃 [AVATAR] Synced {len(entities_data)} entities.")
        except Exception as e:
            logger.error(f"Failed to update avatar: {e}")
