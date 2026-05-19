from typing import Dict, Any, Optional


class StanceManager:
    """
    Chooses a conversational stance based on user signal and self model.
    Default is 'companion'. If user signals repetition/rigidity, switch to 'improv'.
    """

    REPETITION_HINTS = [
        '반복', '같은 말', '지루', '틀에 박힌', '상투', '템플릿', '뻔한'
    ]

    def __init__(self, self_model: Any):
        self.self_model = self_model

    def decide(self, user_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        text = (user_text or '').lower()
        stance = 'companion'
        if any(hint in user_text for hint in self.REPETITION_HINTS):
            stance = 'improv'
        return {
            'name': stance,
            'tone': self.self_model.preferred_tone,
            'anchors': self.self_model.as_lens_anchors()
        }

