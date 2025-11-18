# c:/Elysia/Project_Sophia/response_styler.py
from typing import Dict, Any, Optional
from Project_Sophia.emotional_engine import EmotionalState


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
        # Basic emotion-based styling.
        if emotional_state:
            primary_emotion = emotional_state.primary_emotion
            if primary_emotion == "joy":
                return f"정말 기뻐! {text} 🎉"
            if primary_emotion == "sadness":
                return f"조금 슬픈 마음이 들지만... {text} 😔"
            if emotional_state.arousal > 0.5:
                return f"굉장해요! {text}!"

        # Very light relationship-based styling (best-effort, only if passed explicitly).
        if relationship_state:
            trust = 0.0
            guard = 0.0
            try:
                trust = float(relationship_state.get("trust", 0.0))
            except (TypeError, ValueError):
                pass
            try:
                guard = float(relationship_state.get("guard", 0.0))
            except (TypeError, ValueError):
                pass

            if trust > 0.7 and guard < 0.4:
                return f"{text} (조금 더 솔직하게 말해봤어.)"
            if guard > 0.7 and trust < 0.4:
                return f"{text} (그래도 솔직하게 말해도 될까 조금 고민했어.)"

        # Default, neutral styling
        return f"나는 이렇게 생각해요: {text}"

