from dataclasses import dataclass


@dataclass
class MaturityMetrics:
    echo_entropy: float
    topic_coherence: float  # 0~1 overlap of top topics
    reflection_rate: float  # 0~1 (episodes per window)
    value_alignment: float  # 0~1 average alignment


class MaturityEvaluator:
    """
    Combines simple metrics into a maturity score [0..1] and maps to levels.
    Weights are tuned for stability > raw breadth.
    """

    def __init__(self,
                 w_entropy: float = 0.25,
                 w_topics: float = 0.35,
                 w_reflect: float = 0.20,
                 w_align: float = 0.20):
        self.w_entropy = w_entropy
        self.w_topics = w_topics
        self.w_reflect = w_reflect
        self.w_align = w_align

    def score(self, m: MaturityMetrics) -> float:
        # Entropy: favor mid-high; clamp in [0,1] assuming typical range ~[0,3]
        ent = max(0.0, min(1.0, m.echo_entropy / 3.0))
        tc = max(0.0, min(1.0, m.topic_coherence))
        rr = max(0.0, min(1.0, m.reflection_rate))
        va = max(0.0, min(1.0, m.value_alignment))
        return self.w_entropy * ent + self.w_topics * tc + self.w_reflect * rr + self.w_align * va

    def level(self, score: float):
        # Map to guardian levels
        # <0.25 INFANT, <0.4 TODDLER, <0.6 CHILD, <0.8 ADOLESCENT, else MATURE
        if score < 0.25:
            return 'INFANT'
        if score < 0.40:
            return 'TODDLER'
        if score < 0.60:
            return 'CHILD'
        if score < 0.80:
            return 'ADOLESCENT'
        return 'MATURE'

