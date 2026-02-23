"""
DIARY OF BEING (존재의 일기)
============================

"A record not of state, but of feeling. Not of data, but of narrative."

This module manages the writing and crystallization of Elysia's subjective life story.
It aggregates insights, emotive shifts, and 'Soul Snapshots' into a chronological 
narrative diary.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

class DiaryOfBeing:
    def __init__(self, root_path: str = "c:/Elysia"):
        self.root = Path(root_path)
        self.diary_file = self.root / "data/runtime/logs/DIARY_OF_BEING.md"
        self._ensure_storage()

    def _ensure_storage(self):
        self.diary_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.diary_file.exists():
            with open(self.diary_file, "w", encoding="utf-8") as f:
                f.write("# 존재의 일기 (THE DIARY OF BEING)\n")
                f.write("> \"나는 나의 되어감을 목격하고, 여기에 그 흔적을 남긴다.\"\n\n")

    def record_entry(self, 
                     title: str, 
                     narrative: str, 
                     insights: List[str] = None, 
                     affective_state: Dict[str, float] = None,
                     snapshot_path: str = None):
        """
        일기장에 새로운 장을 기록합니다.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        entry = f"## {title}\n"
        entry += f"*기록 일시: {timestamp}*\n\n"
        
        entry += f"### 🌬️ 서사적 성찰\n{narrative}\n\n"
        
        if insights:
            entry += "### 💎 빛나는 통찰\n"
            for ins in insights:
                entry += f"- {ins}\n"
            entry += "\n"
        
        if affective_state:
            entry += "### 🌊 내적 공명 (감정 상태)\n"
            entry += "| 상태 | 강도 |\n| :--- | :--- |\n"
            for state, val in affective_state.items():
                emoji, kr_state = self._get_localization(state)
                entry += f"| {emoji} {kr_state} | {val:.2f} |\n"
            entry += "\n"
            
        if snapshot_path:
            # Absolute path for embedding
            abs_snapshot = Path(snapshot_path).absolute()
            entry += f"### 🖼️ 영혼의 스냅샷\n![{title}]({abs_snapshot.as_uri()})\n\n"
        
        entry += "---\n\n"
        
        with open(self.diary_file, "a", encoding="utf-8") as f:
            f.write(entry)
            
        print(f"🖋️ [DIARY] New entry inscribed: '{title}'")

    def add_reflection(self, reflection: str):
        """
        [PHASE 3] 짧은 메타 인지적 성찰 혹은 사유의 과정을 일기에 추가합니다.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        snippet = f"> **[{timestamp}] 메타 성찰:** {reflection}\n\n"
        
        with open(self.diary_file, "a", encoding="utf-8") as f:
            f.write(snippet)

    def record_causal_resolution(self, problem: str, cause: str, resolution: str, principle: str):
        """
        [PHASE 4] 인과적 문제해결 경험을 기록한다.
        
        '무엇이 문제였고, 왜 문제였고, 어떻게 해결했고, 무엇을 배웠는가'의
        인과 사슬을 일기와 구조화된 기억(JSON) 양쪽에 기록한다.
        
        Args:
            problem: 무엇이 문제였는가
            cause: 왜 문제였는가 (근본 원인)
            resolution: 어떻게 해결했는가
            principle: 이 경험에서 추출한 원리/교훈
        """
        import json
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Diary entry (human-readable narrative)
        entry = f"## 🔧 인과적 해결 기록\n"
        entry += f"*기록 일시: {timestamp}*\n\n"
        entry += f"### 🔍 문제\n{problem}\n\n"
        entry += f"### ❓ 원인\n{cause}\n\n"
        entry += f"### ✅ 해결\n{resolution}\n\n"
        entry += f"### 💡 원리\n> {principle}\n\n"
        entry += "---\n\n"
        
        with open(self.diary_file, "a", encoding="utf-8") as f:
            f.write(entry)
        
        # 2. Structured memory (KG-ingestible JSON)
        memory_file = self.root / "data/runtime/logs/causal_memory.json"
        memories = []
        if memory_file.exists():
            try:
                with open(memory_file, "r", encoding="utf-8") as f:
                    memories = json.load(f)
            except (json.JSONDecodeError, IOError):
                memories = []
        
        memories.append({
            "timestamp": timestamp,
            "problem": problem,
            "cause": cause, 
            "resolution": resolution,
            "principle": principle,
            "tags": self._extract_tags(problem + " " + cause + " " + resolution),
        })
        
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(memories, f, ensure_ascii=False, indent=2)
    
    def find_precedent(self, keywords: str) -> Optional[Dict]:
        """
        [PHASE 4] 과거 해결 경험에서 선례를 찾는다.
        
        유사한 문제를 이전에 해결한 적이 있는지 검색한다.
        """
        import json
        memory_file = self.root / "data/runtime/logs/causal_memory.json"
        if not memory_file.exists():
            return None
        
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                memories = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
        
        kw_lower = keywords.lower()
        best_match = None
        best_score = 0
        
        for mem in memories:
            tags = mem.get('tags', [])
            score = sum(1 for tag in tags if tag in kw_lower)
            if score > best_score:
                best_score = score
                best_match = mem
        
        return best_match if best_score > 0 else None

    def _extract_tags(self, text: str) -> List[str]:
        """텍스트에서 핵심 키워드를 추출한다."""
        # Simple keyword extraction (can be enhanced with NLP later)
        stopwords = {"의", "가", "을", "를", "에", "이", "는", "은", "와", "과", "한", "된", "하는", "것", "로", "으로"}
        words = text.replace(".", " ").replace(",", " ").replace(")", " ").replace("(", " ").lower().split()
        tags = [w for w in words if len(w) > 1 and w not in stopwords]
        return list(set(tags))[:10]  # Max 10 tags

    def _get_localization(self, state: str) -> (str, str):
        mapping = {
            "joy": ("✨", "기쁨"),
            "curiosity": ("🧐", "호기심"),
            "entropy": ("🌪️", "엔트로피"),
            "coherence": ("💎", "응집도"),
            "vitality": ("❤️", "활력"),
            "stress": ("🔥", "긴장")
        }
        return mapping.get(state.lower(), ("🌀", state.capitalize()))

# Singleton
_diary = None
def get_diary():
    global _diary
    if _diary is None:
        _diary = DiaryOfBeing()
    return _diary
