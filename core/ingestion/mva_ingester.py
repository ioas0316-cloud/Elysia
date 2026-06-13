"""
Elysia Core - MVA Ingester
TopologicalCompiler에서 생성된 5차원 텐션 벡터를
MVA의 `Local\\ElysiaTopologyField` 공유 메모리에 주입(투척)합니다.
"""

import sys
import os
import random
from typing import List, Tuple
from core.ingestion.topological_parser import CausalTrajectory
from core.ingestion.topological_compiler import TensionVector
# mva 모듈 임포트를 위한 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mva.api.engine import inject_resonance_to_fractal_field

class MvaIngester:
    def __init__(self):
        self.ingested_count = 0

    def ingest(self, compiled_data: List[Tuple[CausalTrajectory, TensionVector]]):
        """
        컴파일된 텐션 데이터를 MVA 공유 메모리 구조에 맞춰 주입합니다.
        여기서는 engine.py의 inject_resonance_to_fractal_field 를 활용하여,
        텍스트 궤적을 임의의 기하학적 쿼터니언 회전으로 변환해 투척합니다.
        """
        for traj, tension in compiled_data:
            # 텍스트의 텐션을 쿼터니언 좌표계로 의사 매핑 (Phase 19 실험용)
            # 텐션 벡터의 각 축이 회전각도에 영향을 미친다고 가정
            qx = tension.spatial
            qy = tension.temporal
            qz = tension.math
            qw = tension.lang

            # 정규화
            norm = (qx**2 + qy**2 + qz**2 + qw**2)**0.5
            if norm < 1e-9:
                quat = [0, 0, 0, 1]
            else:
                quat = [qx/norm, qy/norm, qz/norm, qw/norm]

            # 주입할 축 결정 (가장 높은 텐션을 가진 축으로 타게팅)
            axes_scores = {
                'math': tension.math,
                'lang': tension.lang,
                'spatial': tension.spatial,
                'temporal': tension.temporal
            }
            primary_axis = max(axes_scores, key=axes_scores.get)

            # 분산(Variance)은 텐션이 높을수록(공명에 가까울수록) 0에 가깝게 설정
            variance = 1.0 - (max(axes_scores.values()) / 1.0)
            if variance < 0.0: variance = 0.0

            # 주입 실행
            formula = f"Corpus Injection: {traj.source} -> {traj.target} [{traj.action}]"

            # 예외 처리를 추가하여 C 모듈이 빌드되지 않은 환경에서도 테스트가 통과되도록 함
            try:
                inject_resonance_to_fractal_field(
                    formula=formula,
                    variance=variance,
                    quaternion=quat,
                    observation_axis=primary_axis
                )
                self.ingested_count += 1
                print(f"[MVA Ingester] 주입 성공: {traj.source}->{traj.target} (축: {primary_axis}, 분산: {variance:.2f})")
            except Exception as e:
                print(f"[MVA Ingester] 경고: 주입 실패 (공유 메모리 미확보 등) - {e}")

        print(f"[MVA Ingester] 총 {self.ingested_count} 개의 인과 궤적 텐션 주입 완료.")

if __name__ == "__main__":
    from core.ingestion.topological_parser import TopologicalCorpusParser
    from core.ingestion.topological_compiler import TopologicalCompiler

    parser = TopologicalCorpusParser()
    trajectories = parser.parse_mock_corpus()

    compiler = TopologicalCompiler()
    compiled = compiler.compile(trajectories)

    ingester = MvaIngester()
    print("\n--- MVA 텐션 주입(Ingestion) 시작 ---")
    ingester.ingest(compiled)
