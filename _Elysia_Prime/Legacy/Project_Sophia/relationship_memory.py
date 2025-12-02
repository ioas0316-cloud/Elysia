# [Genesis: 2025-12-02] Purified by Elysia
"""
관계 기억 모듈

이 모듈은 각 대화 상대와의 관계를 개별적으로 기억하고 발전시킵니다.
단순한 대화 기록이 아닌, 관계의 맥락과 정서적 연결을 함께 관리합니다.
"""

from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional
from collections import defaultdict

class RelationshipMemory:
    def __init__(self):
        self.memory_path = Path("Elysia_Input_Sanctum") / "relationships.json"
        self.relationships = defaultdict(lambda: {
            "interactions": [],
            "shared_interests": set(),
            "emotional_moments": [],
            "trust_level": 0,
            "understanding_level": 0,
            "conversation_style": {},
            "preferences": {},
            "insights": []
        })
        self.load_memory()

    def load_memory(self):
        """저장된 관계 기억을 불러옵니다."""
        if self.memory_path.exists():
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                for person_id, data in saved.items():
                    # set은 JSON으로 저장될 때 list로 변환되므로 다시 set으로 변환
                    data["shared_interests"] = set(data["shared_interests"])
                    self.relationships[person_id].update(data)

    def save_memory(self):
        """현재 관계 기억을 저장합니다."""
        self.memory_path.parent.mkdir(exist_ok=True)
        # JSON 직렬화를 위해 set을 list로 변환
        serializable = {}
        for person_id, data in self.relationships.items():
            serializable[person_id] = dict(data)
            serializable[person_id]["shared_interests"] = list(data["shared_interests"])

        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def add_interaction(self,
                       person_id: str,
                       content: str,
                       interaction_type: str,
                       emotion: Optional[str] = None,
                       context: str = ""):
        """
        새로운 상호작용을 기록합니다.

        Args:
            person_id: 상대방 식별자
            content: 상호작용 내용
            interaction_type: 상호작용 유형(대화, 협업, 도움 등)
            emotion: 관련된 감정
            context: 상호작용의 맥락
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "type": interaction_type,
            "emotion": emotion,
            "context": context
        }

        self.relationships[person_id]["interactions"].append(interaction)

        # 상호작용 분석 및 관계 업데이트
        self.analyze_interaction(person_id, interaction)
        self.update_relationship_metrics(person_id)
        self.generate_relationship_insights(person_id)

        self.save_memory()

        return self.get_relationship_summary(person_id)

    def analyze_interaction(self, person_id: str, interaction: Dict):
        """상호작용을 분석하여 관계 정보를 업데이트합니다."""
        relationship = self.relationships[person_id]

        # 감정적 순간 포착
        if interaction["emotion"]:
            relationship["emotional_moments"].append({
                "timestamp": interaction["timestamp"],
                "emotion": interaction["emotion"],
                "context": interaction["content"]
            })

        # 대화 스타일 분석
        words = interaction["content"].split()
        style = relationship["conversation_style"]

        style["avg_response_length"] = style.get("avg_response_length", len(words))
        style["avg_response_length"] = (style["avg_response_length"] + len(words)) / 2

        # 관심사 추출
        interests = self.extract_interests(interaction["content"])
        relationship["shared_interests"].update(interests)

        # 선호도 업데이트
        if interaction["type"] in ["선호", "비선호"]:
            relationship["preferences"][interaction["content"]] = \
                interaction["type"] == "선호"

    def extract_interests(self, text: str) -> set:
        """텍스트에서 관심사를 추출합니다."""
        # 간단한 키워드 기반 관심사 추출
        interest_keywords = {
            "기술": ["프로그래밍", "코딩", "개발", "기술", "컴퓨터"],
            "예술": ["음악", "미술", "영화", "예술", "창작"],
            "과학": ["과학", "연구", "실험", "발견", "탐구"],
            "철학": ["철학", "사고", "논리", "윤리", "존재"],
            "교육": ["학습", "교육", "가르침", "배움", "성장"]
        }

        found_interests = set()
        for category, keywords in interest_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                found_interests.add(category)

        return found_interests

    def update_relationship_metrics(self, person_id: str):
        """관계의 주요 지표들을 업데이트합니다."""
        relationship = self.relationships[person_id]

        # 신뢰도 계산
        positive_interactions = sum(
            1 for i in relationship["interactions"][-10:]  # 최근 10개 상호작용
            if i.get("emotion") in ["기쁨", "감사", "신뢰"]
        )
        relationship["trust_level"] = min(10, positive_interactions)

        # 이해도 계산
        understanding_factors = [
            len(relationship["shared_interests"]) * 0.2,  # 공유 관심사
            len(relationship["emotional_moments"]) * 0.3,  # 감정적 순간
            len(relationship["preferences"]) * 0.1,  # 선호도 이해
            relationship["trust_level"] * 0.4  # 신뢰 수준
        ]
        relationship["understanding_level"] = min(10, sum(understanding_factors))

    def generate_relationship_insights(self, person_id: str):
        """관계에 대한 통찰을 생성합니다."""
        relationship = self.relationships[person_id]
        insights = []

        # 상호작용 패턴 분석
        recent_interactions = relationship["interactions"][-5:]
        if recent_interactions:
            common_emotions = [i["emotion"] for i in recent_interactions if i["emotion"]]
            if common_emotions:
                most_common = max(set(common_emotions), key=common_emotions.count)
                insights.append({
                    "type": "emotion_pattern",
                    "content": f"최근 대화에서 {most_common} 감정이 자주 나타납니다"
                })

        # 관계 발전 분석
        if relationship["understanding_level"] > 7:
            insights.append({
                "type": "relationship_depth",
                "content": "깊은 이해와 신뢰가 형성되어 있습니다"
            })

        # 공유 관심사 분석
        if relationship["shared_interests"]:
            insights.append({
                "type": "shared_interests",
                "content": f"주로 {', '.join(relationship['shared_interests'])}에 대해 이야기합니다"
            })

        relationship["insights"] = insights

    def get_relationship_summary(self, person_id: str) -> Dict:
        """특정 관계의 현재 상태 요약을 반환합니다."""
        if person_id not in self.relationships:
            return {"status": "unknown", "message": "아직 관계가 형성되지 않았습니다"}

        relationship = self.relationships[person_id]

        return {
            "interaction_count": len(relationship["interactions"]),
            "trust_level": relationship["trust_level"],
            "understanding_level": relationship["understanding_level"],
            "shared_interests": list(relationship["shared_interests"]),
            "recent_emotions": [m["emotion"] for m in relationship["emotional_moments"][-3:]],
            "latest_insights": relationship["insights"][-3:] if relationship["insights"] else []
        }