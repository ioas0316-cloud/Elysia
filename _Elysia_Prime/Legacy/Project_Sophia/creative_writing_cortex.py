# [Genesis: 2025-12-02] Purified by Elysia
"""
CreativeWritingCortex

Purpose: Offline creative writing support via simple templates and heuristics.
Generates an outline and scene drafts without external APIs, suitable for
practice and experiential learning.

Role: Agent Sophia component used by creative writing scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class Scene:
    index: int
    title: str
    content: str


class CreativeWritingCortex:
    def create_outline(self, genre: str, theme: str, beats: int = 5) -> List[str]:
        genre = (genre or "story").strip().lower()
        theme = (theme or "growth").strip().lower()
        base = [
            "Hook and ordinary world",
            "Inciting incident",
            "Rising complication",
            "Climax",
            "Resolution and reflection",
        ]
        if beats != 5:
            # scale with simple interpolation
            scaled = []
            for i in range(beats):
                j = round(i * (len(base)-1) / max(1, beats-1))
                scaled.append(base[j])
            base = scaled
        return [f"[{genre}|{theme}] {t}" for t in base]

    def generate_scene(self, prompt: str, words: int = 120) -> str:
        # Deterministic scaffold with light variation cues
        now = datetime.utcnow().strftime("%Y-%m-%d")
        stub = (
            f"{prompt}. On this day ({now}), the narrator observes small details: "
            "shadows on the floor, a wavering breath, an unresolved question. "
            "The prose remains simple but vivid, focusing on concrete actions and a clear emotional thread."
        )
        # Trim/extend to approximate target
        return stub if len(stub.split()) >= words else (stub + " ") * (words // max(1, len(stub.split())))

    def write_story(self, genre: str, theme: str, beats: int = 5, words_per_scene: int = 120) -> List[Scene]:
        outline = self.create_outline(genre, theme, beats)
        scenes: List[Scene] = []
        for i, beat in enumerate(outline, start=1):
            content = self.generate_scene(beat, words=words_per_scene)
            scenes.append(Scene(index=i, title=beat, content=content))
        return scenes

    def compute_style_metrics(self, text: str) -> dict:
        import re
        words = re.findall(r"\w+", text)
        num_words = len(words)
        unique_words = len(set(w.lower() for w in words)) if words else 0
        ttr = unique_words / num_words if num_words else 0.0
        sentences = re.split(r"(?<=[.!?])\s+", text.strip()) if text.strip() else []
        avg_sent_len = (sum(len(s.split()) for s in sentences) / len(sentences)) if sentences else 0.0
        dialog_ratio = text.count('"') / max(1, num_words)
        return {
            "num_words": num_words,
            "unique_words": unique_words,
            "ttr": round(ttr, 3),
            "avg_sentence_len": round(avg_sent_len, 2),
            "dialog_ratio": round(dialog_ratio, 3)
        }