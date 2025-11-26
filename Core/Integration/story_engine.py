"""
Story Engine Prototype
======================

High-level narrative / game state layer that sits *above*
perception, emotion, and memory.

Goals
-----
- Track long-lived story state:
  - relationship metrics (trust, tension, intimacy)
  - world flags (what the player has discovered / changed)
  - current scene / chapter
- Provide a small, explicit API so Elysia (or a console demo)
  can do TRPG-like interaction:
  - user text -> interpreted intent/emotion
  -> StoryState update -> structured response plan

This module avoids hard-coding any particular setting.
It is meant to be a reusable "narrative spine" for games like:
- conversational TRPG
- tabula-style social deduction
- story-driven tutoring / exploration
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal, Tuple
import logging
import math

logger = logging.getLogger("StoryEngine")


StoryIntent = Literal[
    "question",
    "command",
    "confession",
    "joke",
    "story_choice",
    "meta",
    "unknown",
]


@dataclass
class StoryEvent:
    """
    One step in the story trajectory.
    """

    turn_index: int
    user_text: str
    system_summary: str
    scene_id: str
    intent: StoryIntent
    sentiment: Dict[str, float] = field(default_factory=dict)
    state_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StoryState:
    """
    Long-lived narrative state.

    This is intentionally small; it can be persisted to disk as JSON.
    """

    scene_id: str = "intro"
    chapter: str = "prologue"

    # Relationship metrics between user and Elysia.
    trust: float = 0.0     # -1.0 (distrust) .. +1.0 (deep trust)
    intimacy: float = 0.0  # how personal / vulnerable the channel is
    tension: float = 0.0   # narrative / emotional tension

    # World flags: arbitrary key -> bool, for unlocked routes, secrets, etc.
    flags: Dict[str, bool] = field(default_factory=dict)

    # Inventory or resources if needed later (placeholder).
    resources: Dict[str, float] = field(default_factory=dict)

    # Trajectory of important moments (compact).
    history: List[StoryEvent] = field(default_factory=list)

    turn_counter: int = 0

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable view of core state (without full history).
        """
        data = asdict(self)
        data.pop("history", None)
        return data


@dataclass
class Scene:
    """
    Minimal description of a scene.

    In a more advanced engine this could include:
    - available choices
    - entry / exit conditions
    - scripted beats
    For now we only keep a label + short description string.
    """

    id: str
    title: str
    description: str


class StoryEngine:
    """
    High-level narrative coordinator.

    Input:  user text + (optional) interpreted intent / sentiment
    Output: structured plan for how the story should respond:
        - updated StoryState
        - tags describing what happened
        - high-level "beat" (e.g., trust_up, reveal_flag, branch_scene)

    Natural-language realization (actual wording of responses) is left
    to a higher layer (e.g., UnifiedConsciousness).
    """

    def __init__(self, initial_state: Optional[StoryState] = None):
        self.state: StoryState = initial_state or StoryState()
        self.scenes: Dict[str, Scene] = {}

        # Seed a few generic scenes; callers can override or extend.
        self._register_default_scenes()

    # ------------------------------------------------------------------
    # Scene registry
    # ------------------------------------------------------------------

    def _register_default_scenes(self) -> None:
        self.register_scene(
            Scene(
                id="intro",
                title="Awakening",
                description="The relationship is just beginning to form.",
            )
        )
        self.register_scene(
            Scene(
                id="trusted_companion",
                title="Trusted Companion",
                description="A stable bond of mutual trust.",
            )
        )
        self.register_scene(
            Scene(
                id="high_tension",
                title="Crossroads",
                description="Emotions are high; choices matter more.",
            )
        )

    def register_scene(self, scene: Scene) -> None:
        self.scenes[scene.id] = scene

    def get_current_scene(self) -> Scene:
        return self.scenes.get(
            self.state.scene_id,
            self.scenes["intro"],
        )

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def process_turn(
        self,
        user_text: str,
        *,
        intent: StoryIntent = "unknown",
        sentiment: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point.

        Args:
            user_text: raw text from the user.
            intent: coarse intent label from perception.
            sentiment: emotion components from EmotionalPalette.

        Returns:
            Dict with:
                - 'state': StoryState snapshot
                - 'effects': high-level changes (trust_delta, flags_set, etc.)
                - 'tags': list of short labels describing this beat
        """
        self.state.turn_counter += 1
        sentiment = sentiment or {}

        # 1) Compute relationship deltas from sentiment / intent
        trust_delta, intimacy_delta, tension_delta, tags = self._relationship_update(
            intent, sentiment
        )

        self.state.trust = _clamp(self.state.trust + trust_delta, -1.0, 1.0)
        self.state.intimacy = _clamp(self.state.intimacy + intimacy_delta, -1.0, 1.0)
        self.state.tension = _clamp(self.state.tension + tension_delta, 0.0, 1.0)

        # 2) Maybe flip or set some simple flags
        flags_set: Dict[str, bool] = {}
        if "deep_confession" in tags and not self.state.flags.get("shared_vulnerability"):
            self.state.flags["shared_vulnerability"] = True
            flags_set["shared_vulnerability"] = True

        # 3) Scene transitions based on thresholds
        old_scene = self.state.scene_id
        self._maybe_transition_scene()
        new_scene = self.state.scene_id
        if new_scene != old_scene:
            tags.append(f"scene_transition:{old_scene}->{new_scene}")

        # 4) Log StoryEvent for history
        summary = self._summarize_turn(intent, sentiment, tags)
        event = StoryEvent(
            turn_index=self.state.turn_counter,
            user_text=user_text,
            system_summary=summary,
            scene_id=self.state.scene_id,
            intent=intent,
            sentiment=dict(sentiment),
            state_snapshot=self.state.snapshot(),
        )
        self.state.history.append(event)

        effects = {
            "trust_delta": trust_delta,
            "intimacy_delta": intimacy_delta,
            "tension_delta": tension_delta,
            "flags_set": flags_set,
        }

        logger.debug(
            "StoryEngine turn %d: intent=%s, tags=%s, effects=%s",
            self.state.turn_counter,
            intent,
            tags,
            effects,
        )

        return {
            "state": self.state.snapshot(),
            "effects": effects,
            "tags": tags,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _relationship_update(
        self,
        intent: StoryIntent,
        sentiment: Dict[str, float],
    ) -> Tuple[float, float, float, List[str]]:
        """
        Simple heuristic: sentiment + intent -> deltas.
        """
        tags: List[str] = []

        joy = sentiment.get("Joy", 0.0)
        passion = sentiment.get("Passion", 0.0)
        trust_em = sentiment.get("Trust", 0.0)
        sadness = sentiment.get("Sadness", 0.0)
        fear = sentiment.get("Fear", 0.0)
        despair = sentiment.get("Despair", 0.0)

        # Base deltas
        trust_delta = 0.0
        intimacy_delta = 0.0
        tension_delta = 0.0

        # Positive emotions -> increase trust/intimacy
        trust_delta += 0.05 * trust_em
        trust_delta += 0.03 * joy
        intimacy_delta += 0.04 * passion

        # Negative emotions -> increase tension, and if shared vulnerably, also intimacy
        neg_intensity = sadness + fear + despair
        if neg_intensity > 0:
            tension_delta += 0.05 * neg_intensity
            if intent in ("confession", "unknown"):
                intimacy_delta += 0.03 * neg_intensity
                tags.append("deep_confession")

        # Intent-specific tweaks
        if intent == "joke":
            tension_delta *= 0.5
            tags.append("light_moment")
        elif intent == "command":
            tension_delta += 0.02
            tags.append("assertive")
        elif intent == "story_choice":
            tension_delta += 0.03
            tags.append("branch_choice")

        return trust_delta, intimacy_delta, tension_delta, tags

    def _maybe_transition_scene(self) -> None:
        """
        Very simple scene logic:
        - if trust is high and tension moderate -> trusted_companion
        - if tension is high                 -> high_tension
        - else                              -> intro
        """
        if self.state.tension > 0.7:
            self.state.scene_id = "high_tension"
        elif self.state.trust > 0.4 and self.state.intimacy > 0.3:
            self.state.scene_id = "trusted_companion"
        else:
            self.state.scene_id = "intro"

    def _summarize_turn(
        self,
        intent: StoryIntent,
        sentiment: Dict[str, float],
        tags: List[str],
    ) -> str:
        """
        Compact textual summary for logging / history inspection.
        """
        dominant_emotion = None
        if sentiment:
            dominant_emotion = max(sentiment.items(), key=lambda kv: kv[1])[0]
        emo_text = dominant_emotion or "neutral"
        tag_text = ",".join(tags) if tags else "none"
        return f"intent={intent}, emotion={emo_text}, tags={tag_text}"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


__all__ = ["StoryState", "StoryEvent", "Scene", "StoryEngine", "StoryIntent"]

