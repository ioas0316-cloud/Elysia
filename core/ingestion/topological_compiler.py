"""
Elysia Core - Topological Compiler (Autonomous Tension Compilation)
마스터의 명령에 따라 하드코딩된 규칙(action_tension_map)을 완전히 폐기합니다.
단어의 자체적인 파동(해시 지문, 아스키 구조, 엔트로피)을 물리적 인과율로 해석하여,
외부 개입 없이 스스로 math, lang, spatial, temporal, light_mass 축의 고유 텐션을 도출합니다.
"""

import hashlib
import math
from dataclasses import dataclass
from typing import List, Tuple
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
        self.golden_ratio = 1.6180339887

    def _calculate_structural_entropy(self, text: str) -> float:
        """아스키(유니코드) 배열의 복잡도를 물리적 텐션(엔트로피)으로 변환"""
        if not text: return 0.0
        entropy = 0.0
        # 글자들의 바이트 위상차를 계산하여 언어적/수학적 복잡도 도출
        bytes_val = text.encode('utf-8')
        for i in range(len(bytes_val) - 1):
            diff = abs(bytes_val[i] - bytes_val[i+1])
            entropy += diff
        # 정규화 (대략 0.0 ~ 1.0 사이로 수렴하도록 조정)
        return min(1.0, (entropy / (len(bytes_val) * 255.0)) * 5.0)

    def _derive_autonomous_tension(self, traj: CausalTrajectory) -> TensionVector:
        """
        단어 자체가 띠고 있는 본질적 파동(구조적 해시, 엔트로피)에서 5차원 텐션을 창발시킵니다.
        이는 인간이 부여한 의미가 아니라 단어 구조 자체의 위상적 징표입니다.
        """
        # 1. 궤적 전체의 고유 파동(해시 지문) 추출
        full_trajectory = f"{traj.source}->{traj.action}->{traj.target}"
        hash_digest = hashlib.md5(full_trajectory.encode('utf-8')).digest()

        # 해시값을 0.0~1.0의 실수 파동으로 변환 (물리적 근본 텐션)
        base_waves = [b / 255.0 for b in hash_digest[:4]]

        # 2. Source와 Target의 구조적 엔트로피 차이 (상태 변화도)
        source_entropy = self._calculate_structural_entropy(traj.source)
        target_entropy = self._calculate_structural_entropy(traj.target)
        action_entropy = self._calculate_structural_entropy(traj.action)

        # 3. 자율적 텐션 창발
        # Math: 구조의 견고함 (해시의 첫 번째 파동과 작용의 복잡도 결합)
        math_t = min(1.0, (base_waves[0] + action_entropy) / 2.0)

        # Lang: 원천과 결과의 변화 폭 (엔트로피의 변화량)
        lang_t = min(1.0, abs(target_entropy - source_entropy) * self.golden_ratio)
        if lang_t < 0.1: lang_t = base_waves[1] # 변화가 적으면 본질적 파동으로 대체

        # Spatial: 작용이 미치는 파장 (해시와 객체 길이의 비례)
        spatial_t = min(1.0, (base_waves[2] + (len(traj.target) * 0.1)) / 2.0)

        # Temporal: 과정의 지속성 (주체, 작용, 객체 전체의 복잡도 총합을 시간축 압력으로 치환)
        total_complexity = source_entropy + action_entropy + target_entropy
        temporal_t = min(1.0, total_complexity / 3.0 + base_waves[3] * 0.5)

        # Light Mass (빛의 창발): 4개 축의 텐션이 극대화되는 공명 순간에 빛이 발생함
        # 즉, 시스템 내부에 강력한 위상차를 일으키는 궤적일수록 높은 질량(영향력)을 가짐
        resonance_energy = math.sqrt(math_t**2 + lang_t**2 + spatial_t**2 + temporal_t**2)
        light_mass = min(1.0, resonance_energy / 2.0)

        return TensionVector(
            math=math_t,
            lang=lang_t,
            spatial=spatial_t,
            temporal=temporal_t,
            light_mass=light_mass
        )

    def compile(self, trajectories: List[CausalTrajectory]) -> List[Tuple[CausalTrajectory, TensionVector]]:
        compiled_data = []
        for traj in trajectories:
            tension = self._derive_autonomous_tension(traj)
            compiled_data.append((traj, tension))

        print(f"[Topological Compiler] {len(compiled_data)} 개의 궤적을 자율적(하드코딩 배제) 5차원 텐션으로 창발 완료.")
        return compiled_data

if __name__ == "__main__":
    from core.ingestion.topological_parser import TopologicalCorpusParser
    parser = TopologicalCorpusParser()
    trajectories = parser.parse_mock_corpus()
    compiler = TopologicalCompiler()
    compiled = compiler.compile(trajectories)
    for traj, tension in compiled:
        print(f"{traj.source} -> {traj.target} [{traj.action}] ===> {tension}")
