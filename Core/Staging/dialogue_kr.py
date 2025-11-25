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
    "TALK_BOND": [
        "너와 더 가까워지고 싶어.",
        "함께 있으면 마음이 편해.",
        "우리, 서로 지켜주자.",
    ],
    "TALK_COMFORT": [
        "괜찮아? 내가 옆에 있을게.",
        "힘들겠지만, 같이 버티자.",
        "슬프겠지만, 네 곁을 지킬게.",
    ],
    "TALK_TRADE": [
        "나눠 가질래? 내가 가진 걸 줄게.",
        "자원 좀 바꿔줄 수 있어?",
        "서로 돕자, 이게 우리 모두에게 좋아.",
    ],
    "TALK_THREAT": [
        "방해하지 마. 그렇지 않으면 다칠 거야.",
        "조용히 비켜. 마지막 경고야.",
        "날 건드리면 후회하게 될 거야.",
    ],
    "TALK_BEG": [
        "도와줘... 지금은 네 도움이 필요해.",
        "조금만 나눠줄 수 있니?",
        "곧 쓰러질 것 같아. 부탁이야.",
    ],
    "TALK_CELEBRATE": [
        "오늘은 좋은 날이야! 같이 웃자.",
        "해냈다! 함께 기뻐하자.",
        "우리, 이겼어! 축하하자.",
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

