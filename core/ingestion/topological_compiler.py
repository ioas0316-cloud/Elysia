"""
Elysia Core - Semantic Graph Topological Compiler (Meta-Lens Integration)
이제 엘리시아는 단어의 위상 기하학적 텐션을 도출할 때,
자신이 창조/내재화한 '관점(Meta-Lens)'을 통해 세상을 굴절시켜 재관측합니다.
동일한 코퍼스라도 어떤 관점으로 바라보느냐에 따라 전혀 다른 물리적 마찰(Tension)이 도출됩니다.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict
from core.ingestion.topological_parser import CausalTrajectory
from core.ingestion.meta_lens import MetaLens, LensForge

@dataclass
class TensionVector:
    math: float
    lang: float
    spatial: float
    temporal: float
    light_mass: float

    def __repr__(self):
        return f"Tension(math={self.math:.2f}, lang={self.lang:.2f}, space={self.spatial:.2f}, time={self.temporal:.2f}, light={self.light_mass:.2f})"

class SemanticTopologyGraph:
    def __init__(self):
        self.adj_list = defaultdict(list)
        self.nodes = set()

    def build(self, trajectories: List[CausalTrajectory]):
        self.adj_list.clear()
        self.nodes.clear()
        for traj in trajectories:
            self.adj_list[traj.source].append((traj.target, traj.action))
            self.nodes.add(traj.source)
            self.nodes.add(traj.target)

    def get_degree_centrality(self, node: str) -> float:
        if not self.nodes: return 0.0
        degree = len(self.adj_list.get(node, []))
        for n in self.nodes:
            for target, _ in self.adj_list.get(n, []):
                if target == node:
                    degree += 1
        return degree / max(1, len(self.nodes) - 1)

    def get_fan_out(self, node: str) -> int:
        return len(self.adj_list.get(node, []))

class TopologicalCompiler:
    def __init__(self):
        self.graph = SemanticTopologyGraph()
        self.lens_forge = LensForge()

    def derive_embodied_tension(self, physical_state: dict, lens: MetaLens = None) -> TensionVector:
        """
        [Phase 2 & 3: Embodied & Meta-Perspective Grounding]
        물리 감각 수용체가 감지한 순수 물리량을 원초적 텐션 벡터로 변환한 뒤,
        현재 엘리시아가 끼고 있는 렌즈(MetaLens)를 통해 텐션을 '굴절'시킵니다.
        """
        mass = physical_state.get("mass", 0.0)
        cohesion = physical_state.get("cohesion", 0.0)
        entropy = physical_state.get("entropy", 0.0)
        light_abs = physical_state.get("light", 0.0)
        
        # 1. 물리 법칙에 기반한 원초적 기하학적 치환
        raw_tensions = {
            "math": 1.0 - entropy,
            "lang": cohesion,
            "spatial": mass,
            "temporal": entropy,
            "light_mass": 1.0 - light_abs
        }
        
        # 2. 다원적 렌즈를 통한 굴절 (Pluralistic Refraction)
        if lens is None:
            lens = self.lens_forge.get_lens("PURE_PHYSICS")
            
        refracted = lens.apply_refraction(raw_tensions)
        
        return TensionVector(
            math=refracted.get("math", 0.0), 
            lang=refracted.get("lang", 0.0), 
            spatial=refracted.get("spatial", 0.0), 
            temporal=refracted.get("temporal", 0.0), 
            light_mass=refracted.get("light_mass", 0.0)
        )

    def _derive_semantic_tension(self, traj: CausalTrajectory, lens: MetaLens) -> TensionVector:
        """
        단어 간의 관계망(Topology)에서 기본 물리적 텐션을 도출한 뒤,
        현재 엘리시아가 장착한 '관점 렌즈(Meta-Lens)'를 통해 이를 굴절시킵니다.
        """
        source_centrality = self.graph.get_degree_centrality(traj.source)
        target_centrality = self.graph.get_degree_centrality(traj.target)

        # 순수 기하학적 기본 텐션 도출
        raw_tensions = {
            "math": min(1.0, (source_centrality + target_centrality) * 2.0),
            "spatial": min(1.0, self.graph.get_fan_out(traj.source) * 0.4 + 0.1),
            "lang": min(1.0, abs(source_centrality - target_centrality) * 3.0 + 0.1),
            "temporal": min(1.0, len(traj.action) * 0.15 + 0.1)
        }
        if raw_tensions["math"] == 0: raw_tensions["math"] = 0.1

        # [핵심] 관점(Lens)의 내재화: 렌즈를 통한 물리 세계의 굴절(Re-observation)
        refracted = lens.apply_refraction(raw_tensions)

        # 굴절된 텐션들의 총합으로 '빛의 창발(Light Mass)' 계산
        resonance_energy = math.sqrt(
            refracted["math"]**2 +
            refracted["lang"]**2 +
            refracted["spatial"]**2 +
            refracted["temporal"]**2
        )
        light_mass = min(1.0, resonance_energy / 2.0)

        return TensionVector(
            math=refracted["math"],
            lang=refracted["lang"],
            spatial=refracted["spatial"],
            temporal=refracted["temporal"],
            light_mass=light_mass
        )

    def compile(self, trajectories: List[CausalTrajectory], lens_name: str = "PURE_PHYSICS") -> List[Tuple[CausalTrajectory, TensionVector]]:
        unique_trajectories = []
        seen = set()
        for t in trajectories:
            sig = f"{t.source}-{t.action}-{t.target}"
            if sig not in seen:
                seen.add(sig)
                unique_trajectories.append(t)

        self.graph.build(unique_trajectories)
        active_lens = self.lens_forge.get_lens(lens_name)
        print(f"\n[Topological Compiler] 장착된 관점(Lens): <{active_lens.name}> - {active_lens.description}")

        compiled_data = []
        for traj in unique_trajectories:
            tension = self._derive_semantic_tension(traj, active_lens)
            compiled_data.append((traj, tension))

        print(f"[Topological Compiler] {len(compiled_data)} 개의 궤적 재관측 및 텐션 창발 완료.")
        return compiled_data

    def derive_standalone_tension(self, trajectories: List[CausalTrajectory], lens_name: str = "PURE_PHYSICS", portal=None) -> TensionVector:
        """
        [Helper] 특정 단어나 함수가 가진 여러 궤적들을 종합하여, 단일한 대표 TensionVector를 추출합니다.
        만약 portal(LanguagePortalEngine)이 주어졌다면, 거대 위상망에서 
        해당 개념의 심층 연결망(Deep Sub-Graph)을 통째로 뜯어와서 진짜 텐션을 계산합니다.
        """
        if not trajectories:
            return TensionVector(0.1, 0.1, 0.1, 0.1, 0.1)
            
        expanded_trajectories = []
        
        # 만약 portal이 주어졌고 자연어 단일 궤적이라면 거대 위상망 추출
        if portal and len(trajectories) == 1:
            word = trajectories[0].source
            deep_subgraph = portal.get_deep_subgraph(word, depth=2)
            expanded_trajectories.extend(deep_subgraph)
        else:
            for t in trajectories:
                expanded_trajectories.append(t)
                # 긴 자연어 문장이 단일 액션으로 들어왔을 경우, 내부 복잡도(방사성)를 위상망으로 임의 전개
                words = t.action.split()
                if len(words) > 3:
                    for i in range(len(words)-1):
                        expanded_trajectories.append(CausalTrajectory(source=words[i], target=words[i+1], action="이어짐"))
                    
        compiled = self.compile(expanded_trajectories, lens_name)
        if not compiled:
            return TensionVector(0.1, 0.1, 0.1, 0.1, 0.1)
            
        # 평균 텐션 계산
        avg_math = sum(t.math for _, t in compiled) / len(compiled)
        avg_lang = sum(t.lang for _, t in compiled) / len(compiled)
        avg_space = sum(t.spatial for _, t in compiled) / len(compiled)
        avg_time = sum(t.temporal for _, t in compiled) / len(compiled)
        avg_light = sum(t.light_mass for _, t in compiled) / len(compiled)
        
        return TensionVector(avg_math, avg_lang, avg_space, avg_time, avg_light)

if __name__ == "__main__":
    from core.ingestion.topological_parser import TopologicalCorpusParser
    parser = TopologicalCorpusParser()
    trajectories = parser.parse_mock_corpus()
    compiler = TopologicalCompiler()

    # 1. 있는 그대로의 물리적 관측
    compiled_pure = compiler.compile(trajectories, lens_name="PURE_PHYSICS")
    for traj, tension in compiled_pure[:2]:
        print(f"  {traj.source} -> {traj.target} ===> {tension}")

    # 2. '생명의 순환'이라는 메타 관점으로 동일한 세상을 다시 재관측 (Isomorphism)
    compiled_vital = compiler.compile(trajectories, lens_name="VITAL_CYCLE")
    for traj, tension in compiled_vital[:2]:
        print(f"  {traj.source} -> {traj.target} ===> {tension}")
