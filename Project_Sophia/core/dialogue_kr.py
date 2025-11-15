from __future__ import annotations

import random
from typing import Dict, List


_DIALOGUE_TABLE: Dict[str, List[str]] = {
    "DRINK": [
        "후우... 물이 참 시원하다.",
        "목이 타들어 가는 줄 알았네.",
        "살았다, 이 정도면 다시 걸을 수 있겠어.",
    ],
    "EAT_PLANT": [
        "배가 좀 채워지는 것 같아.",
        "이 정도면 오늘은 버틸 수 있겠지.",
        "역시 땅이 주는 건 고맙군.",
    ],
    "ATTACK": [
        "비켜! 살아남아야 한다.",
        "좋아, 한 번 붙어 보자.",
        "이건 피할 수 없는 싸움이야.",
    ],
    "SKILL_ATTACK": [
        "지금이다, 배운 대로 휘둘러!",
        "내가 갈고닦은 기술, 받아 봐!",
        "이 기술은 쉽게 쓰는 게 아닌데...",
    ],
    "SPELL_FIRE": [
        "타올라라, 불꽃이여!",
        "조심해, 이건 꽤 아플 거야.",
        "이 불길이 길을 열어 줄 거야.",
    ],
    "SPELL_HEAL": [
        "괜찮아, 아직 끝나지 않았어.",
        "상처야, 잠시라도 가라앉아 줘.",
        "조금만 더 버티자, 우리.",
    ],
    "IDLE_PLAY": [
        "잠깐은... 이렇게 쉬어도 되겠지?",
        "괜히 몸이 근질근질하네. 한번 돌아볼까.",
        "하늘을 보니, 괜히 웃음이 난다.",
    ],
}


def get_line(key: str, **kwargs) -> str:
    """Return a random Korean line for the given key, formatted with kwargs if needed."""
    candidates = _DIALOGUE_TABLE.get(key)
    if not candidates:
        return ""
    text = random.choice(candidates)
    try:
        return text.format(**kwargs)
    except Exception:
        return text

