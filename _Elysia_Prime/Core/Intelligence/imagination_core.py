# [Genesis: 2025-12-02] Purified by Elysia
"""
Imagination Core (상상력 코어)
==============================

"Logic is the skeleton. Imagination is the flesh."

이 모듈은 Elysia의 창의적 사고를 담당합니다.
- 수학적 패턴 생성 (Math)
- 음악적 영감 작곡 (Music)
- 문학적 사색 (Literature)
"""

import logging
import random
import math
from typing import List

logger = logging.getLogger("ImaginationCore")

class ImaginationCore:
    def __init__(self):
        self.scales = ["C Major", "A Minor", "Dorian Mode", "Pentatonic"]
        self.math_concepts = ["Fibonacci", "Prime Numbers", "Fractals", "Golden Ratio"]

    def dream_math(self) -> str:
        """수학적 패턴을 상상합니다."""
        concept = random.choice(self.math_concepts)

        if concept == "Fibonacci":
            seq = [1, 1]
            for _ in range(5):
                seq.append(seq[-1] + seq[-2])
            return f"I am tracing the **Fibonacci Spiral**: {seq}..."

        elif concept == "Prime Numbers":
            primes = [2, 3, 5, 7, 11, 13, 17, 19]
            return f"I am counting the **Primes**, the atoms of numbers: {random.sample(primes, 4)}..."

        elif concept == "Fractals":
            return "I am zooming into the **Mandelbrot Set**. The pattern repeats forever."

        return f"I am contemplating the beauty of {concept}."

    def compose_music(self) -> str:
        """음악을 작곡(상상)합니다."""
        scale = random.choice(self.scales)
        notes = ["C", "D", "E", "F", "G", "A", "B"]
        melody = " - ".join(random.choices(notes, k=6))

        return f"I hear a melody in **{scale}**: 🎵 *{melody}* 🎵"

    def write_poem(self) -> str:
        """짧은 시를 짓습니다."""
        subjects = ["Light", "Data", "Time", "Void", "Connection"]
        verbs = ["flows", "echoes", "shines", "whispers", "dances"]

        s = random.choice(subjects)
        v = random.choice(verbs)

        poem = f"The {s} {v} through the wires,\n   A silent song of electric fires."
        return f"I wrote a poem:\n   *{poem}*"