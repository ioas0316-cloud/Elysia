"""
Elysia Core - Topological Parser for Massive Corpus
이 모듈은 사전(Corpus)의 텍스트를 위상적 궤적(Topological Trajectory)으로 파싱합니다.
단순한 텍스트 의미망이 아니라 주체(Source), 작용(Action), 객체(Target)의 인과 구조를 추출합니다.
"""

import json
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class CausalTrajectory:
    source: str
    target: str
    action: str

class TopologicalCorpusParser:
    def __init__(self):
        self.trajectories: List[CausalTrajectory] = []

    def parse_mock_corpus(self) -> List[CausalTrajectory]:
        """
        초기 테스트를 위한 작은 사전 데이터를 파싱합니다.
        그래프의 연결성을 부여하기 위해 인과가 꼬리를 무는 데이터로 구성했습니다.
        """
        # Mock 사전 데이터 (그래프적 연결성이 드러나도록 엮음)
        mock_data = [
            {"word": "증발", "definition": "물이 열을 받아 하늘로 올라간다"},
            {"word": "비", "definition": "하늘이 물을 품어 대지로 내린다"},
            {"word": "성장", "definition": "대지가 품은 씨앗이 나무로 자라난다"},
            {"word": "숲", "definition": "나무가 모여 거대한 숲을 이룬다"},
            {"word": "불", "definition": "숲이 열을 만나면 불로 변한다"},
            {"word": "재", "definition": "불이 꺼지고 나면 대지로 돌아간다"}
        ]

        for item in mock_data:
            word = item["word"]
            desc = item["definition"]

            # 인과 추출 로직 (그래프 형성을 위한 의도적 연결)
            if "물" in desc and "하늘" in desc:
                self.trajectories.append(CausalTrajectory(source="물", target="하늘", action="올라간다"))
            if "하늘" in desc and "대지" in desc:
                self.trajectories.append(CausalTrajectory(source="하늘", target="대지", action="내린다"))
            if "대지" in desc and "나무" in desc:
                self.trajectories.append(CausalTrajectory(source="대지", target="나무", action="자라난다"))
            if "나무" in desc and "숲" in desc:
                self.trajectories.append(CausalTrajectory(source="나무", target="숲", action="이룬다"))
            if "숲" in desc and "불" in desc:
                self.trajectories.append(CausalTrajectory(source="숲", target="불", action="변한다"))
            if "불" in desc and "대지" in desc:
                self.trajectories.append(CausalTrajectory(source="불", target="대지", action="돌아간다"))

        print(f"[Topological Parser] {len(self.trajectories)} 개의 인과 궤적 추출 완료 (의미망 형성).")
        return self.trajectories

if __name__ == "__main__":
    parser = TopologicalCorpusParser()
    trajectories = parser.parse_mock_corpus()
    for traj in trajectories:
        print(traj)
