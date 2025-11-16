from __future__ import annotations

import re
from typing import Dict, List


def _s(x: str) -> str:
    return (x or '').strip()


def _conceptify(x: str) -> str:
    x = _s(x)
    if not x:
        return x
    if ':' in x:
        return x
    return f'concept:{x}'


def interpret(text: str) -> List[Dict[str, Dict]]:
    """
    Very small Korean intent → nano message mapping (safe patterns only).
    Returns [{'verb': str, 'slots': dict}, ...] or empty list on no match.
    """
    t = _s(text)
    if not t:
        return []
    # Normalize spaces
    tsp = re.sub(r"\s+", " ", t)
    out: List[Dict[str, Dict]] = []

    # link: "A를 B와 연결", "A를 B에 연결", "A와 B를 연결"
    m = re.match(r"^(.+?)(?:를|을)\s+(.+?)(?:와|과|에)\s+연결$", tsp)
    if m:
        a, b = _conceptify(m.group(1)), _conceptify(m.group(2))
        out.append({'verb': 'link', 'slots': {'subject': a, 'object': b, 'rel': 'related_to'}})
        return out
    m = re.match(r"^(.+?)(?:와|과)\s+(.+?)(?:를|을)\s*연결$", tsp)
    if m:
        a, b = _conceptify(m.group(1)), _conceptify(m.group(2))
        out.append({'verb': 'link', 'slots': {'subject': a, 'object': b, 'rel': 'related_to'}})
        return out

    # verify: "A와 B를 검증/확인"
    m = re.match(r"^(.+?)(?:와|과)\s+(.+?)(?:를|을)\s*(검증|확인)$", tsp)
    if m:
        a, b = _conceptify(m.group(1)), _conceptify(m.group(2))
        out.append({'verb': 'verify', 'slots': {'subject': a, 'object': b, 'rel': 'related_to'}})
        return out

    # summarize: "X 요약", "X 정리", "X 요약해(줘)"
    m = re.match(r"^(.+?)\s*(요약|정리)(해줘|해)?$", tsp)
    if m:
        x = _conceptify(m.group(1))
        out.append({'verb': 'summarize', 'slots': {'target': x}})
        return out

    # compose: "A와 B 조합/결합"
    m = re.match(r"^(.+?)(?:와|과|랑|하고)\s+(.+?)\s*(조합|결합)$", tsp)
    if m:
        a, b = _conceptify(m.group(1)), _conceptify(m.group(2))
        out.append({'verb': 'compose', 'slots': {'a': a, 'b': b}})
        return out

    # explain: "X 설명: 텍스트", "X를 설명: 텍스트"
    m = re.match(r"^(.+?)(?:를|을)?\s*설명\s*:\s*(.+)$", tsp)
    if m:
        x, txt = _conceptify(m.group(1)), _s(m.group(2))
        out.append({'verb': 'explain', 'slots': {'target': x, 'text': txt}})
        return out

    return out

