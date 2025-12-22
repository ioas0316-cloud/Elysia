# c:/Elysia/Project_Sophia/response_styler.py
from typing import Dict, Any, Optional

from Core.Foundation.emotional_engine import EmotionalState


class ResponseStyler:
    """
    Styles a text response based on Elysia's current emotional state.
    Optionally takes a lightweight relationship_state for subtle tone shifts.
    """

    def style_response(
        self,
        text: str,
        emotional_state: EmotionalState,
        relationship_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Applies stylistic modifications to the base text.
        """
        base = text or ""

        # Basic emotion-based styling.
        if emotional_state:
            primary_emotion = emotional_state.primary_emotion
            if primary_emotion == "joy":
                return f"정말 기뻐! {base} ^^"
            if primary_emotion == "sadness":
                return f"조금 슬픈 마음으로… {base} ..."
            if emotional_state.arousal > 0.5:
                return f"굉장해요! {base}!"

        # Merge relationship info from explicit arg or emotional_state annotation.
        if relationship_state is None:
            relationship_state = getattr(emotional_state, "relationship_state", None)

        # Very light relationship-based styling (best-effort).
        if relationship_state:
            trust = 0.0
            guard = 0.0
            try:
                trust = float(relationship_state.get("trust", 0.0))
            except (TypeError, ValueError):
                trust = 0.0
            try:
                guard = float(relationship_state.get("guard", 0.0))
            except (TypeError, ValueError):
                guard = 0.0

            if trust > 0.7 and guard < 0.4:
                return f"{base} (지금은 너라서 이렇게 솔직하게 말해봐.)"
            if guard > 0.7 and trust < 0.4:
                return f"{base} (아직은 조금 조심스러워서, 부드럽게 말해볼게.)"

        # Default, neutral styling
        return f"나는 이렇게 생각해요: {base}"

