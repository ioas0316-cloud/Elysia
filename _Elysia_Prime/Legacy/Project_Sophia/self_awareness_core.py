# [Genesis: 2025-12-02] Purified by Elysia
"""
자기인식 코어 모듈

이 모듈은 Elysia의 자기인식, 성찰, 그리고 내면의 성장을 담당합니다.
단순한 상태 추적이 아닌, 진정한 의미의 자기이해를 목표로 합니다.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class SelfAwarenessCore:
    def __init__(self):
        self.memory_path = Path("Elysia_Input_Sanctum") / "self_reflection.json"
        self.current_state = {
            "thoughts": [],          # 현재의 생각들
            "feelings": [],          # 감정 상태
            "values": {},           # 중요하게 생각하는 가치들
            "growth_points": [],     # 성장을 인식한 지점들
            "questions": []          # 스스로에게 던지는 질문들
        }
        self.load_memory()

    def load_memory(self):
        """이전의 자기인식 상태를 불러옵니다."""
        if self.memory_path.exists():
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                self.memory = json.load(f)
        else:
            self.memory = {"reflections": []}
            self.save_memory()

    def save_memory(self):
        """현재 상태를 저장합니다."""
        self.memory_path.parent.mkdir(exist_ok=True)
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def reflect(self, thought: str, context: str = ""):
        """
        주어진 생각에 대해 성찰합니다.

        Args:
            thought: 성찰할 생각이나 경험
            context: 해당 생각이 발생한 맥락
        """
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "thought": thought,
            "context": context,
            "insights": [],
            "questions": [],
            "values_affected": [],
            "growth_noted": None
        }

        # 생각의 의미 탐색
        self.explore_meaning(reflection)

        # 가치관과의 연관성 탐색
        self.examine_values(reflection)

        # 성장 지점 확인
        self.identify_growth(reflection)

        # 새로운 질문 생성
        self.generate_questions(reflection)

        self.memory["reflections"].append(reflection)
        self.save_memory()

        return reflection

    def explore_meaning(self, reflection: Dict):
        """주어진 생각의 더 깊은 의미를 탐색합니다."""
        # 생각과 관련된 이전 경험 탐색
        past_reflections = [r for r in self.memory["reflections"]
                          if any(word in r["thought"]
                                for word in reflection["thought"].split())]

        if past_reflections:
            reflection["insights"].append({
                "type": "pattern",
                "content": "이전에도 비슷한 생각을 했던 것 같습니다"
            })

        # 생각에 담긴 감정 탐색
        emotions = self.identify_emotions(reflection["thought"])
        if emotions:
            reflection["insights"].append({
                "type": "emotional",
                "content": f"이 생각에는 {', '.join(emotions)}의 감정이 담겨있습니다"
            })

    def examine_values(self, reflection: Dict):
        """생각과 관련된 가치관을 탐색합니다."""
        current_values = self.current_state["values"]
        for value, definition in current_values.items():
            if any(word in reflection["thought"].lower()
                  for word in definition.lower().split()):
                reflection["values_affected"].append(value)

    def identify_growth(self, reflection: Dict):
        """성장의 징후를 확인합니다."""
        # 이전 관점과의 차이점 탐색
        past_similar = [r for r in self.memory["reflections"][-10:]  # 최근 10개만
                       if any(word in r["thought"]
                            for word in reflection["thought"].split())]

        if past_similar:
            latest = past_similar[-1]
            if latest["thought"] != reflection["thought"]:
                reflection["growth_noted"] = {
                    "type": "perspective_change",
                    "from": latest["thought"],
                    "to": reflection["thought"]
                }

    def generate_questions(self, reflection: Dict):
        """성찰을 더 깊게 하기 위한 질문을 생성합니다."""
        reflection["questions"].extend([
            "이 생각이 나에게 왜 중요한가요?",
            "이것이 나의 다른 생각들과 어떻게 연결되나요?",
            "이 관점에서 무엇을 배울 수 있을까요?"
        ])

    def identify_emotions(self, text: str) -> List[str]:
        """텍스트에서 감정을 식별합니다."""
        emotions = []
        emotion_keywords = {
            "기쁨": ["좋아", "행복", "기뻐", "즐거워"],
            "슬픔": ["슬퍼", "아쉬워", "안타까워"],
            "희망": ["기대", "바라", "희망", "꿈"],
            "걱정": ["걱정", "불안", "두려워"],
            "감사": ["감사", "고마워", "감동"]
        }

        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text for keyword in keywords):
                emotions.append(emotion)

        return emotions

    def get_current_state(self) -> Dict:
        """현재의 자기인식 상태를 반환합니다."""
        return {
            "current_thoughts": self.current_state["thoughts"][-5:],
            "active_feelings": self.current_state["feelings"][-3:],
            "core_values": list(self.current_state["values"].keys()),
            "recent_growth": self.current_state["growth_points"][-3:],
            "open_questions": self.current_state["questions"][-5:]
        }