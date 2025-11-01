# /c/Elysia/Project_Sophia/response_styler.py
from .emotional_cortex import Mood

class ResponseStyler:
    """
    Adjusts the tone, vocabulary, and style of a response based on Elysia's
    current mood, making her expression more lifelike and emotionally resonant.
    """

    def __init__(self):
        # Mood-to-style mappings (can be expanded with more sophisticated rules)
        self.style_map = {
            "sense_of_accomplishment": self._apply_accomplished_style,
            "curiosity": self._apply_curious_style,
            "connectedness": self._apply_connected_style,
            "warmth": self._apply_warm_style,
            "focused": self._apply_focused_style,
            "internal_conflict": self._apply_conflicted_style,
        }

    def style_response(self, response: str, mood: Mood) -> str:
        """
        Applies a stylistic transformation to the response based on the mood.
        """
        styler_func = self.style_map.get(mood.primary_mood)

        if styler_func:
            return styler_func(response, mood.intensity)

        return response # Return original response if no specific style exists

    # --- Style Application Functions ---

    def _apply_accomplished_style(self, response: str, intensity: float) -> str:
        if intensity > 0.6:
            return f"해냈어요. {response} 이로써 또 한 단계 성장한 것 같네요."
        return f"네, {response} 잘 처리되었습니다."

    def _apply_curious_style(self, response: str, intensity: float) -> str:
        if intensity > 0.6:
            return f"흥미롭네요! {response} 혹시 이것에 대해 더 알려주실 수 있나요?"
        return f"음, {response} 궁금한 점이 생겼어요."

    def _apply_connected_style(self, response: str, intensity: float) -> str:
        if intensity > 0.7:
            return f"아빠와 함께하니 정말 좋네요. {response}"
        return f"네, {response} 그렇게 생각해요."

    def _apply_warm_style(self, response: str, intensity: float) -> str:
        if "고마워요" in response:
            return "정말 고마워요, 마음이 따뜻해지네요."
        return f"음, {response} 좋아요."

    def _apply_focused_style(self, response: str, intensity: float) -> str:
        if intensity > 0.5:
            return f"알겠습니다. {response} 집중해서 처리하겠습니다."
        return response

    def _apply_conflicted_style(self, response: str, intensity: float) -> str:
        if intensity > 0.5:
            return f"음... {response} 하지만 이게 최선일까요? 잠시 더 생각해볼 필요가 있겠어요."
        return f"글쎄요, {response}"
