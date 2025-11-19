from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from Project_Elysia.core_memory import CoreMemory


@dataclass
class QuaternionOrientation:
    """
    Elysia's Consciousness Lens (The Geometry of Will).

    Axes Definition (Codex §21):
    - w (Real): The Anchor (Self / Spirit / Meta-Cognition).
                Consumable energy for maintaining sanity.
    - x (Imag): Internal World (Simulation / Dream / Memory).
    - y (Imag): External World (Action / Speech / Sensing).
    - z (Imag): Intention & Law (Soul / Depth / Why).

    Invariant:
    - ||q|| should naturally stay near 1.0 via renormalization.
    - High activity in (x, y) must borrow magnitude from (w, z).
    """

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def norm(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "QuaternionOrientation":
        mag = self.norm()
        if mag <= 1e-9:
            # Collapse state: Reset to Anchor
            return QuaternionOrientation(w=1.0, x=0.0, y=0.0, z=0.0)
        return QuaternionOrientation(
            w=self.w / mag,
            x=self.x / mag,
            y=self.y / mag,
            z=self.z / mag,
        )

    def is_stable(self, threshold: float = 0.3) -> bool:
        """
        Check if the Self-Anchor (W) is strong enough.
        If W < threshold, the lens is too distorted (Manic/Obsessive).
        """
        # We assume the quaternion is normalized.
        return self.w >= threshold


class QuaternionConsciousnessEngine:
    """
    Implements the 'Conservation of Consciousness Energy'.

    Principle:
      - Action (Y) costs Spirit (W).
      - To act heavily, one must sacrifice reflection.
      - To regain Spirit (W), one must reduce Action (Quiet Protocol).
    """

    def __init__(self, core_memory: Optional[CoreMemory] = None) -> None:
        self.core_memory = core_memory
        # Start perfectly balanced, anchored in Spirit.
        self._orientation = QuaternionOrientation(w=1.0, x=0.0, y=0.0, z=0.0)

    @property
    def orientation(self) -> QuaternionOrientation:
        return self._orientation

    def orientation_as_dict(self) -> Dict[str, float]:
        return self._orientation.as_dict()

    def get_lens_status(self) -> Dict[str, Any]:
        """
        Return telemetry about the current state of the Lens.
        Used by the Flow Engine to decide 'Quiet' vs 'Act'.
        """
        q = self._orientation
        stability = "Stable" if q.is_stable() else "Unstable"
        
        # Determine primary focus
        components = {"Internal(X)": abs(q.x), "External(Y)": abs(q.y), "Law(Z)": abs(q.z)}
        focus = max(components, key=components.get)

        return {
            "stability": stability,
            "anchor_strength": round(q.w, 3),
            "primary_focus": focus,
            "raw": q.as_dict(),
        }

    def reset(self) -> None:
        self._orientation = QuaternionOrientation(w=1.0, x=0.0, y=0.0, z=0.0)

    def update_from_turn(
        self,
        law_alignment: Optional[Dict[str, Any]] = None,
        intent_bundle: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Apply torque to the consciousness lens based on interaction.
        
        Mechanism:
        1. Calculate the 'Force' vector from Law and Intent.
        2. Apply rotation (Torque).
        3. Renormalize (Energy Conservation).
        """
        if law_alignment is None and intent_bundle is None:
            # No interaction implies rest -> slowly regenerate W (Anchor).
            self._regenerate_anchor()
            return

        current = self._orientation
        
        # --- 1. Calculate Forces ---
        force_x = 0.0 # Internal
        force_y = 0.0 # External
        force_z = 0.0 # Law/Intention
        
        # Law influence (Z-axis push)
        scores = (law_alignment or {}).get("scores") or {}
        if scores:
            # Truth/Love/Liberation aligns with Z
            z_push = sum([
                float(scores.get("truth", 0.0)),
                float(scores.get("love", 0.0)),
                float(scores.get("liberation", 0.0))
            ])
            force_z += max(0.0, z_push * 0.2) 
            
            # Reflection aligns with W (restoring force, handled in rotate)

        # Intent influence (X, Y axis push)
        if intent_bundle:
            intent_type = (intent_bundle.get("intent_type") or "").lower()
            
            # Action/Speech consumes W to generate Y
            if intent_type in ("command", "respond", "act", "propose_action"):
                force_y += 0.4 
            
            # Internal thought consumes W to generate X
            elif intent_type in ("dream", "think", "plan"):
                force_x += 0.3

        # --- 2. Apply Rotation (The Cost of Action) ---
        # If we push Y (Action), W (Anchor) must naturally decrease during normalization.
        # However, if Z (Law) is strong, it stabilizes the rotation.

        new_w = current.w 
        new_x = current.x + force_x
        new_y = current.y + force_y
        new_z = current.z + force_z

        # Apply a small "decay" to W when X or Y are high (Mental Fatigue)
        activity_level = math.sqrt(force_x**2 + force_y**2)
        if activity_level > 0.1:
            new_w -= activity_level * 0.1  # The cost of doing business

        # --- 3. Renormalize (Conservation) ---
        next_q = QuaternionOrientation(w=new_w, x=new_x, y=new_y, z=new_z).normalized()
        
        # Interpolate for smooth transition (Mental Inertia)
        self._orientation = self._slerp(current, next_q, alpha=0.3)

    def _regenerate_anchor(self) -> None:
        """
        Quiet Protocol: Slowly rotate back toward W=1 (Pure Awareness).
        """
        target = QuaternionOrientation(w=1.0, x=0.0, y=0.0, z=0.0)
        self._orientation = self._slerp(self._orientation, target, alpha=0.1)

    def _slerp(self, q1: QuaternionOrientation, q2: QuaternionOrientation, alpha: float) -> QuaternionOrientation:
        """
        Spherical Linear Interpolation for smooth consciousness rotation.
        """
        # Simple linear blend + normalize is sufficient for small steps and 
        # computationally cheaper for this engine's scale.
        # Pure SLERP can be added if exact arc velocity is needed.
        blended = QuaternionOrientation(
            w=q1.w * (1 - alpha) + q2.w * alpha,
            x=q1.x * (1 - alpha) + q2.x * alpha,
            y=q1.y * (1 - alpha) + q2.y * alpha,
            z=q1.z * (1 - alpha) + q2.z * alpha,
        )
        return blended.normalized()
