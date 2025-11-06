from __future__ import annotations

import re
from typing import Dict, List, Tuple


def _strip(s: str) -> str:
    return (s or '').strip()


def _parse_list(s: str) -> List[str]:
    s = _strip(s)
    if not s:
        return []
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1]
        parts = [p.strip() for p in inner.split(',') if p.strip()]
        return parts
    return [s]


def _conceptify(x: str) -> str:
    x = _strip(x)
    if not x:
        return x
    if ':' in x:
        return x
    return f'concept:{x}'


def interpret(text: str) -> List[Dict[str, Dict]]:
    """
    Rules-based intent → nano message mapping.
    Returns a list of {'verb': str, 'slots': dict} entries, or empty list if no match.
    """
    t = (text or '').strip()
    if not t:
        return []
    # Normalize whitespace
    t_sp = re.sub(r"\s+", " ", t)

    out: List[Dict[str, Dict]] = []

    # link: "link A to B" or "connect A -> B"
    m = re.match(r"^(link|connect)\s+(.+?)\s+(?:to|->)\s+(.+)$", t_sp, re.I)
    if m:
        a, b = m.group(2), m.group(3)
        a_list = [_conceptify(x) for x in _parse_list(a)]
        b_list = [_conceptify(x) for x in _parse_list(b)]
        for aa in a_list:
            for bb in b_list:
                out.append({'verb': 'link', 'slots': {'subject': aa, 'object': bb, 'rel': 'related_to'}})
        return out

    # verify: "verify A to B"
    m = re.match(r"^verify\s+(.+?)\s+(?:to|->)\s+(.+)$", t_sp, re.I)
    if m:
        a, b = m.group(1), m.group(2)
        out.append({'verb': 'verify', 'slots': {'subject': _conceptify(a), 'object': _conceptify(b), 'rel': 'related_to'}})
        return out

    # summarize: "summarize X" or "summary X" or "context X"
    m = re.match(r"^(summarize|summary|context)\s+(.+)$", t_sp, re.I)
    if m:
        x = m.group(2)
        out.append({'verb': 'summarize', 'slots': {'target': _conceptify(x)}})
        return out

    # compose: "compose A and B"
    m = re.match(r"^compose\s+(.+?)\s+(?:and|&|\+)\s+(.+)$", t_sp, re.I)
    if m:
        a, b = m.group(1), m.group(2)
        out.append({'verb': 'compose', 'slots': {'a': _conceptify(a), 'b': _conceptify(b)}})
        return out

    # explain: "explain X: some text" or "explain X with Y"
    m = re.match(r"^explain\s+(.+?)(?:\s*:\s*|\s+with\s+)(.+)$", t_sp, re.I)
    if m:
        tgt, txt = m.group(1), m.group(2)
        out.append({'verb': 'explain', 'slots': {'target': _conceptify(tgt), 'text': txt}})
        return out

    return []

