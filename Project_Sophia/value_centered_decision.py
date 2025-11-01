import re
import random

class VCD:
    """Value-Centered Decision helper.

    Simple implementation:
    - core_value: string (e.g., 'love')
    - score_action(candidate, context, emotion, memory): returns float score
    - suggest_action(candidates, context, emotion, memory): returns best candidate
    """
    def __init__(self, core_value='love'):
        self.core_value = core_value.lower() if core_value else 'love'
        # tunable weights
        self.alpha = 1.0  # value alignment
        self.beta = 0.7   # context fit
        self.gamma = 1.5  # freshness / novelty
        self.recent_history = []  # recent outputs for repetition penalty

    def set_core_value(self, v):
        self.core_value = v.lower()

    def value_alignment(self, text):
        """Simple keyword-based alignment to core value."""
        if not text:
            return 0.0
        keywords = {
            'love': ['사랑', '고마', '연결', '감사', '따뜻', '관계', '돌봄'],
            # add more mappings for other core values as needed
        }
        ks = keywords.get(self.core_value, [])
        s = 0
        for k in ks:
            if k in text:
                s += 1
        # normalize
        return min(1.0, s / max(1, len(ks)))

    def context_fit(self, text, context):
        """Naive context fit: fraction of context tokens present in text."""
        if not context:
            return 0.5
        ctx = ' '.join(context).lower()
        text_l = text.lower()
        tokens = set(re.findall(r"\w+", ctx))
        if not tokens:
            return 0.5
        found = sum(1 for t in tokens if t in text_l)
        return min(1.0, found / len(tokens))

    def freshness(self, text):
        """Penalize repetition: if recently used, lower score."""
        penalty = 0.0
        history_len = len(self.recent_history[-10:])
        for i, prev in enumerate(reversed(self.recent_history[-10:])):
            if prev == text:
                # Penalize more for more recent occurrences (i=0 is most recent)
                penalty += 0.9 * (history_len - i)
        return max(0.0, 1.0 - penalty)

    def score_action(self, candidate, context=None, emotion='neutral', memory=None):
        va = self.value_alignment(candidate)
        cf = self.context_fit(candidate, context)
        fr = self.freshness(candidate)
        score = self.alpha * va + self.beta * cf + self.gamma * fr
        # small random tie-breaker
        score += random.random() * 0.01
        return score

    def suggest_action(self, candidates, context=None, emotion='neutral', memory=None):
        if not candidates:
            return None
        scored = [(self.score_action(c, context, emotion, memory), c) for c in candidates]
        scored.sort(reverse=True)
        best = scored[0][1]
        # record chosen candidate
        self.recent_history.append(best)
        return best
