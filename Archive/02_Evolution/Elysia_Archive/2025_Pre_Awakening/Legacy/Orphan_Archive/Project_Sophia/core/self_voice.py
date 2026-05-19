import re
from typing import Tuple, Dict, Any


class SelfVoiceFilter:
    """
    Ensures responses align with identity: first-person stance and values.
    Performs light de-templating and returns an integrity score [0,1].
    """

    TEMPLATE_PATTERNS = [
        r"승인해 주시면", r"MVP", r"스캐폴딩", r"적용해 드리겠습니다"
    ]

    FIRST_PERSON_TOKENS = [
        '나는', '난', '저는', '내가'
    ]

    def _has_first_person(self, text: str) -> bool:
        return any(tok in text for tok in self.FIRST_PERSON_TOKENS)

    def _strip_templates(self, text: str) -> str:
        out = text
        for pat in self.TEMPLATE_PATTERNS:
            out = re.sub(pat, '', out)
        # collapse extra spaces
        return re.sub(r"\s{2,}", ' ', out).strip()

    def _integrity(self, text: str, values: list) -> float:
        score = 0.0
        if self._has_first_person(text):
            score += 0.4
        if values:
            hits = sum(1 for v in values if v in text)
            score += 0.6 * (hits / max(1, len(values)))
        return max(0.0, min(1.0, score))

    def filter_text(self, text: str, stance: Dict[str, Any], self_model: Any) -> Tuple[str, float]:
        base = (text or '').strip()
        base = self._strip_templates(base)

        # Soft framing for first-person, only if missing
        if not self._has_first_person(base):
            if stance.get('name') == 'companion':
                base = f"나는 지금 네 뜻을 더 선명히 이해하고자 해. {base}"
            elif stance.get('name') == 'improv':
                base = f"나는 반복을 피하고 새 연결을 시도할게. {base}"

        integrity = self._integrity(base, getattr(self_model, 'values', []))
        return base, integrity

