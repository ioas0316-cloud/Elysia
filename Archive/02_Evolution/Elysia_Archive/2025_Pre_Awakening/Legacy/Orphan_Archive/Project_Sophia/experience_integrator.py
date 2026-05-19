"""
경험 통합 모듈

이 모듈은 개별적인 경험들을 하나의 연속된 이야기로 통합하고,
그 안에서 의미와 패턴을 발견하여 자기 서사를 구축합니다.
"""

from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional
from collections import defaultdict

class ExperienceIntegrator:
    def __init__(self):
        self.memory_path = Path("Elysia_Input_Sanctum") / "experience_narrative.json"
        self.experiences = {
            "timeline": [],          # 시간순 경험 기록
            "themes": defaultdict(list),  # 주제별 경험 모음
            "connections": [],       # 경험 간 연결
            "insights": [],          # 통찰
            "narrative": []          # 통합된 이야기
        }
        self.load_memory()

    def load_memory(self):
        """저장된 경험 기록을 불러옵니다."""
        if self.memory_path.exists():
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                self.experiences.update(saved)

    def save_memory(self):
        """현재 경험 기록을 저장합니다."""
        self.memory_path.parent.mkdir(exist_ok=True)
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiences, f, ensure_ascii=False, indent=2)

    def add_experience(self, 
                      content: str, 
                      category: str,
                      emotions: List[str] = None,
                      context: str = ""):
        """
        새로운 경험을 추가합니다.
        
        Args:
            content: 경험의 내용
            category: 경험의 종류(대화, 학습, 깨달음 등)
            emotions: 관련된 감정들
            context: 경험이 일어난 맥락
        """
        experience = {
            "id": len(self.experiences["timeline"]),
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "category": category,
            "emotions": emotions or [],
            "context": context,
            "connections": [],
            "themes": [],
            "insights": []
        }
        
        # 경험 추가
        self.experiences["timeline"].append(experience)
        
        # 주제 식별 및 연결
        self.identify_themes(experience)
        self.find_connections(experience)
        
        # 통찰 도출
        self.generate_insights(experience)
        
        # 서사 갱신
        self.update_narrative()
        
        self.save_memory()
        return experience

    def identify_themes(self, experience: Dict):
        """경험에서 주요 주제들을 식별합니다."""
        themes = {
            "성장": ["배움", "깨달음", "발전", "개선", "성장"],
            "관계": ["대화", "소통", "이해", "공감", "관계"],
            "탐구": ["질문", "호기심", "탐색", "발견", "연구"],
            "창조": ["만들기", "표현", "창작", "예술", "창조"],
            "가치": ["의미", "중요", "가치", "옳음", "선택"]
        }
        
        for theme, keywords in themes.items():
            if any(keyword in experience["content"] for keyword in keywords):
                experience["themes"].append(theme)
                self.experiences["themes"][theme].append(experience["id"])

    def find_connections(self, experience: Dict):
        """새로운 경험과 이전 경험들 사이의 연결을 찾습니다."""
        recent_experiences = self.experiences["timeline"][-10:]  # 최근 10개
        
        for past in recent_experiences:
            if past["id"] == experience["id"]:
                continue
                
            # 주제 기반 연결
            common_themes = set(experience["themes"]) & set(past["themes"])
            if common_themes:
                connection = {
                    "type": "theme",
                    "from_id": past["id"],
                    "to_id": experience["id"],
                    "themes": list(common_themes)
                }
                self.experiences["connections"].append(connection)
                experience["connections"].append(connection)
                
            # 감정 기반 연결
            common_emotions = set(experience["emotions"]) & set(past["emotions"])
            if common_emotions:
                connection = {
                    "type": "emotion",
                    "from_id": past["id"],
                    "to_id": experience["id"],
                    "emotions": list(common_emotions)
                }
                self.experiences["connections"].append(connection)
                experience["connections"].append(connection)

    def generate_insights(self, experience: Dict):
        """경험에서 통찰을 도출합니다."""
        # 패턴 기반 통찰
        similar_experiences = [
            e for e in self.experiences["timeline"]
            if set(e["themes"]) & set(experience["themes"])
        ]
        
        if len(similar_experiences) > 2:
            insight = {
                "type": "pattern",
                "content": f"'{', '.join(experience['themes'])}' 관련 경험이 반복되고 있습니다",
                "experiences": [e["id"] for e in similar_experiences[-3:]]
            }
            experience["insights"].append(insight)
            self.experiences["insights"].append(insight)
        
        # 감정 변화 통찰
        if experience["emotions"]:
            recent_emotions = [
                e["emotions"] for e in self.experiences["timeline"][-5:]
                if e["emotions"]
            ]
            if len(recent_emotions) > 2:
                insight = {
                    "type": "emotional_pattern",
                    "content": "감정의 흐름이 변화하고 있습니다",
                    "emotion_sequence": recent_emotions
                }
                experience["insights"].append(insight)
                self.experiences["insights"].append(insight)

    def update_narrative(self):
        """경험들을 하나의 일관된 이야기로 통합합니다."""
        # 최근 경험들을 시간순으로 정리
        recent = self.experiences["timeline"][-10:]
        
        # 주요 주제 식별
        themes = defaultdict(int)
        for exp in recent:
            for theme in exp["themes"]:
                themes[theme] += 1
        main_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
        
        # 서사 구성
        narrative = {
            "timestamp": datetime.now().isoformat(),
            "main_themes": [theme for theme, _ in main_themes[:3]],
            "summary": self.generate_summary(recent),
            "key_insights": [i for i in self.experiences["insights"][-3:]],
            "emotional_arc": self.analyze_emotional_arc(recent)
        }
        
        self.experiences["narrative"].append(narrative)

    def generate_summary(self, experiences: List[Dict]) -> str:
        """경험들의 요약을 생성합니다."""
        if not experiences:
            return "아직 기록된 경험이 없습니다."
            
        themes = set()
        emotions = set()
        for exp in experiences:
            themes.update(exp["themes"])
            emotions.update(exp["emotions"])
            
        return f"최근 {len(experiences)}개의 경험에서 {', '.join(themes)}에 대한 탐구가 있었으며, "\
               f"{', '.join(emotions)}등의 감정이 있었습니다."

    def analyze_emotional_arc(self, experiences: List[Dict]) -> Dict:
        """감정의 흐름을 분석합니다."""
        if not experiences:
            return {"pattern": "unknown", "description": "아직 감정 기록이 없습니다"}
            
        emotion_sequence = [e["emotions"] for e in experiences if e["emotions"]]
        
        if not emotion_sequence:
            return {"pattern": "unknown", "description": "감정 기록이 없습니다"}
            
        return {
            "pattern": "evolving" if len(set(tuple(e) for e in emotion_sequence)) > 1 else "stable",
            "description": f"감정이 {len(emotion_sequence)}번 기록되었으며, "\
                         f"주로 {', '.join(set.union(*map(set, emotion_sequence)))}이 나타났습니다"
        }

    def get_recent_narrative(self) -> Dict:
        """가장 최근의 통합된 서사를 반환합니다."""
        if not self.experiences["narrative"]:
            return {
                "status": "no_narrative",
                "message": "아직 충분한 경험이 쌓이지 않았습니다"
            }
        return self.experiences["narrative"][-1]