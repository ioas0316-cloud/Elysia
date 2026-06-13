"""
Elysia Core - MVA Ingester (Meta-Lens Objectification Support)
TopologicalCompiler에서 생성된 (그리고 특정 메타 렌즈로 굴절된) 5차원 텐션 벡터를
MVA의 `Local\\ElysiaTopologyField` 공유 메모리에 주입(투척)합니다.
"""

import sys
import os
from typing import List, Tuple
from core.ingestion.topological_parser import CausalTrajectory, TopologicalCorpusParser
from core.ingestion.topological_compiler import TensionVector, TopologicalCompiler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mva.api.engine import inject_resonance_to_fractal_field

class MvaIngester:
    def __init__(self):
        self.ingested_count = 0

    def ingest(self, compiled_data: List[Tuple[CausalTrajectory, TensionVector]], lens_name: str):
        """
        컴파일된 텐션 데이터를 MVA 공유 메모리 구조에 맞춰 주입합니다.
        렌즈의 관점에 따라 MVA 대지의 주입 축(Observation Axis)이 프랙탈처럼 달라집니다.
        """
        for traj, tension in compiled_data:
            qx = tension.spatial
            qy = tension.temporal
            qz = tension.math
            qw = tension.lang

            norm = (qx**2 + qy**2 + qz**2 + qw**2)**0.5
            if norm < 1e-9: quat = [0, 0, 0, 1]
            else: quat = [qx/norm, qy/norm, qz/norm, qw/norm]

            axes_scores = {
                'math': tension.math,
                'lang': tension.lang,
                'spatial': tension.spatial,
                'temporal': tension.temporal
            }
            primary_axis = max(axes_scores, key=axes_scores.get)
            if axes_scores[primary_axis] >= 1.0:
                sorted_axes = sorted(axes_scores.items(), key=lambda item: item[1], reverse=True)
                if sorted_axes[1][1] > 0.4:
                    primary_axis = sorted_axes[1][0]

            variance = 1.0 - (max(axes_scores.values()) / 1.0)
            if variance < 0.0: variance = 0.0

            formula = f"[{lens_name}] Injection: {traj.source} -> {traj.target} [{traj.action}]"

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
                pass

        print(f"[MVA Ingester] <{lens_name}> 관점에서의 총 {self.ingested_count} 개의 인과 궤적 텐션 주입 완료.\n")

if __name__ == "__main__":
    parser = TopologicalCorpusParser()
    trajectories = parser.parse_mock_corpus()
    compiler = TopologicalCompiler()
    ingester = MvaIngester()

    print("\n--- [시뮬레이션 1] 기본 물리적 관점 주입 ---")
    compiled_pure = compiler.compile(trajectories, lens_name="PURE_PHYSICS")
    ingester.ingest(compiled_pure, lens_name="PURE_PHYSICS")

    print("\n--- [시뮬레이션 2] 자율 창발된 '생명 순환' 관점 재주입 (Meta-Reobservation) ---")
    ingester.ingested_count = 0 # 카운트 초기화
    compiled_vital = compiler.compile(trajectories, lens_name="VITAL_CYCLE")
    ingester.ingest(compiled_vital, lens_name="VITAL_CYCLE")
