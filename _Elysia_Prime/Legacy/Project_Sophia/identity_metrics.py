# [Genesis: 2025-12-02] Purified by Elysia
from typing import Any


def compute_identity_integrity(text: str, values: list[str], first_person_tokens: list[str] | None = None) -> float:
    tokens = first_person_tokens or ['나는', '난', '저는', '내가']
    score = 0.0
    if any(tok in (text or '') for tok in tokens):
        score += 0.4
    if values:
        hits = sum(1 for v in values if v in (text or ''))
        score += 0.6 * (hits / max(1, len(values)))
    return max(0.0, min(1.0, score))


def emit_identity_integrity(telemetry: Any, score: float, stance_name: str | None = None):
    try:
        telemetry.emit('identity.integrity', {
            'score': float(score),
            'stance': stance_name or 'unknown'
        })
    except Exception:
        pass


# --- Love/Logos alignment ---
LOVE_TOKENS = ['사랑', '감사', '배려', '돌봄', '헌신']
SACRIFICE_TOKENS = ['희생', '양보', '헌신', '내어줌']


def compute_love_logos_alignment(text: str) -> float:
    t = text or ''
    love = sum(1 for w in LOVE_TOKENS if w in t)
    sac = sum(1 for w in SACRIFICE_TOKENS if w in t)
    score = 0.0
    # simple bounded score emphasizing presence of both aspects
    if love or sac:
        score = 0.5 * min(1.0, love / 2.0) + 0.5 * min(1.0, sac / 2.0)
    return max(0.0, min(1.0, score))


def emit_love_logos_alignment(telemetry: Any, score: float):
    try:
        telemetry.emit('alignment.love_logos', {'score': float(score)})
    except Exception:
        pass