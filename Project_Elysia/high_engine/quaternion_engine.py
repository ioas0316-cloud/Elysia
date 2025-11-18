from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from Project_Elysia.core_memory import CoreMemory


@dataclass
class QuaternionOrientation:
    """
    Simple consciousness-orientation quaternion.

    Interpretation (Codex §21 style):
    - x, y: behavior plane (world / interaction axes)
    - z   : intention / law axis (Z-axis)
    - w   : meta / reflection axis
    """

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def normalized(self) -> "QuaternionOrientation":
        mag_sq = self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
        if mag_sq <= 0.0:
            return QuaternionOrientation()
        mag = mag_sq ** 0.5
        return QuaternionOrientation(
            w=self.w / mag,
            x=self.x / mag,
            y=self.y / mag,
            z=self.z / mag,
        )

    def lerp_toward(self, target: "QuaternionOrientation", alpha: float) -> "QuaternionOrientation":
        """
        Very small step toward another orientation.
        This is not a full SLERP; we keep it simple and normalize.
        """
        alpha = max(0.0, min(1.0, alpha))
        inv = 1.0 - alpha
        blended = QuaternionOrientation(
            w=self.w * inv + target.w * alpha,
            x=self.x * inv + target.x * alpha,
            y=self.y * inv + target.y * alpha,
            z=self.z * inv + target.z * alpha,
        )
        return blended.normalized()


class QuaternionConsciousnessEngine:
    """
    Minimal quaternion engine for Elysia's consciousness orientation.

    Role:
      - Keep a running orientation of where attention/interpretation sits
        between behavior (XY), intention/law (Z), and meta-reflection (W).
      - Allow higher layers to update this orientation given:
        - recent law_alignment (7-law scores),
        - recent dialogue intent bundle,
        - optional CoreMemory-based self-view.

    Design goals:
      - Small, inspectable, no side effects on its own.
      - Produces orientation telemetry that other modules can read.
    """

    def __init__(self, core_memory: Optional[CoreMemory] = None) -> None:
        self.core_memory = core_memory
        self._orientation = QuaternionOrientation().normalized()

    @property
    def orientation(self) -> QuaternionOrientation:
        return self._orientation

    def orientation_as_dict(self) -> Dict[str, float]:
        return self._orientation.as_dict()

    def reset(self) -> None:
        self._orientation = QuaternionOrientation().normalized()

    def update_from_turn(
        self,
        law_alignment: Optional[Dict[str, Any]] = None,
        intent_bundle: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update orientation based on a single dialogue turn.

        Heuristics (small, safe steps):
          - Reflection-heavy turns pull toward W (meta).
          - Strong law scores (life/creation/truth/…) pull toward Z (intention).
          - Command / action-oriented intents pull toward XY (behavior).
        """
        if law_alignment is None and intent_bundle is None:
            return

        current = self._orientation
        target = QuaternionOrientation(
            w=current.w,
            x=current.x,
            y=current.y,
            z=current.z,
        )

        # --- Law-based influence (Z and W) ---
        scores = (law_alignment or {}).get("scores") or {}
        if isinstance(scores, dict) and scores:
            reflection = float(scores.get("reflection", 0.0))
            truth = float(scores.get("truth", 0.0))
            love = float(scores.get("love", 0.0))
            liberation = float(scores.get("liberation", 0.0))

            # Intention / law axis (Z): truth / love / liberation weight.
            z_push = max(0.0, truth + love + liberation)
            if z_push > 0.0:
                target.z += z_push

            # Meta / reflection axis (W): reflection weight.
            if reflection > 0.0:
                target.w += reflection

        # --- Intent-based influence (XY) ---
        if isinstance(intent_bundle, dict):
            intent_type = (intent_bundle.get("intent_type") or "").lower()
            style = (intent_bundle.get("style") or "").lower()

            # External / action / planning pulls toward +X,+Y (outer world, behavior).
            if intent_type in ("command", "propose_action", "plan", "respond"):
                target.x += 0.5
                target.y += 0.5

            # Internal / dream-like processing pulls toward -X (inner world).
            if intent_type in ("dream", "inner_dream", "rumination"):
                target.x -= 0.5

            # Strongly introspective styles pull slightly toward W (meta / reflection).
            if "reflect" in style or "introspective" in style:
                target.w += 0.3

        # Small step toward target.
        self._orientation = current.lerp_toward(target, alpha=0.2)
