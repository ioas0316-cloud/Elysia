from Project_Sophia.core_memory import EmotionalState

class ResponseStyler:
    """
    Adjusts the tone and vocabulary of text responses based on Elysia's current mood.
    """
    def style_response(self, text: str, emotional_state: EmotionalState) -> str:
        """
        Applies stylistic changes to the response text based on the emotional state.
        """
        if emotional_state.primary_emotion == 'joy' and emotional_state.arousal > 0.5:
            return f"{text} ㅋㅋㅋ"

        # Add more styling rules for other emotions here in the future

        return text
