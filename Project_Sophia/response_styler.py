# c:/Elysia/Project_Sophia/response_styler.py
from typing import Dict, Any
from Project_Sophia.emotional_engine import EmotionalState

class ResponseStyler:
    """
    Styles a text response based on Elysia's current emotional state.
    """
    def style_response(self, text: str, emotional_state: EmotionalState) -> str:
        """
        Applies stylistic modifications to the base text.
        """
        # This is a basic implementation. More complex logic can be added later.
        if emotional_state:
            primary_emotion = emotional_state.primary_emotion
            if primary_emotion == 'joy':
                return f"정말 기뻐요! {text} 🎉"
            elif primary_emotion == 'sadness':
                return f"조금 슬픈 마음이 들지만... {text} 😔"
            elif emotional_state.arousal > 0.5:
                return f"굉장해요! {text}!"

        # Default, neutral styling
        return f"저는 이렇게 생각해요: {text}"
