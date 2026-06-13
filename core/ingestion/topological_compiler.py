"""
Elysia Core - Topological Compiler (Tension Compilation)
추출된 인과 궤적(Causal Trajectory)을 MVA 대지가 호흡할 수 있는
5차원 텐션 벡터(math, lang, spatial, temporal, light_mass)로 변환(컴파일)합니다.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from core.ingestion.topological_parser import CausalTrajectory

@dataclass
class TensionVector:
    math: float
    lang: float
    spatial: float
    temporal: float
    light_mass: float

    def __repr__(self):
        return f"Tension(math={self.math:.2f}, lang={self.lang:.2f}, space={self.spatial:.2f}, time={self.temporal:.2f}, light={self.light_mass:.2f})"

class TopologicalCompiler:
    def __init__(self):
        # 액션(동사)별 장력 매핑 룰 (Phase 19를 위한 하드코딩된 예시, 향후 동적 학습)
        self.action_tension_map = {
            "올라간다": TensionVector(math=0.1, lang=0.2, spatial=0.8, temporal=0.5, light_mass=0.3),
            "자라난다": TensionVector(math=0.2, lang=0.3, spatial=0.5, temporal=0.9, light_mass=0.6),
            "밝힌다": TensionVector(math=0.8, lang=0.1, spatial=0.9, temporal=0.2, light_mass=1.0),
            "끌어당긴다": TensionVector(math=0.9, lang=0.1, spatial=0.7, temporal=0.4, light_mass=0.8),
            "원한다": TensionVector(math=0.0, lang=0.9, spatial=0.1, temporal=0.8, light_mass=0.2) # 생물학적 텐션/결핍
        }

    def compile(self, trajectories: List[CausalTrajectory]) -> List[Tuple[CausalTrajectory, TensionVector]]:
        compiled_data = []
        for traj in trajectories:
            tension = self.action_tension_map.get(traj.action, TensionVector(0,0,0,0,0))
            compiled_data.append((traj, tension))

        print(f"[Topological Compiler] {len(compiled_data)} 개의 궤적을 5차원 텐션 벡터로 컴파일 완료.")
        return compiled_data

if __name__ == "__main__":
    from core.ingestion.topological_parser import TopologicalCorpusParser
    parser = TopologicalCorpusParser()
    trajectories = parser.parse_mock_corpus()
    compiler = TopologicalCompiler()
    compiled = compiler.compile(trajectories)
    for traj, tension in compiled:
        print(f"{traj.source} -> {traj.target} [{traj.action}] ===> {tension}")
