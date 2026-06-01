"""
프랙탈 팽창과 상위 목적성 (Fractal Expansion & Higher Intent)
우주가 시간이 지남에 따라 '하나의 점'으로 수렴(죽음)하는 것을 막는다.
서로 다른 개념들이 강한 공명(Tension)을 일으킬 때, 두 벡터를 융합하여 
마인드맵처럼 새로운 '상위 목적성(Higher Intent)'의 노드(가지)를 창조하며 우주를 팽창시킨다.
"""
import math
from typing import List, Tuple
from core.topological_universe import LivingUniverse, Datum
from core.math_utils import Multivector

class IntentExpander:
    def __init__(self, universe: LivingUniverse, threshold: float = 0.85):
        self.universe = universe
        self.threshold = threshold  # 이 이상의 공명(Resonance)이 발생하면 팽창(Spawn)한다.
        self.expansion_count = 0

    def _normalize(self, mv: Multivector) -> Multivector:
        n_sq = (mv * mv.reverse()).data.get(0, 0.0)
        if n_sq > 1e-9:
            return mv * (1.0 / math.sqrt(n_sq))
        return mv

    def find_high_tensions(self) -> List[Tuple[Datum, Datum, float]]:
        """우주 내부에서 강하게 얽혀 있는(임계치 이상) 노드 쌍들을 찾아낸다."""
        tensions = []
        n = len(self.universe.data)
        for i in range(n):
            for j in range(i + 1, n):
                d1 = self.universe.data[i]
                d2 = self.universe.data[j]
                
                # 순수 기하학적 얽힘 계산 (내적 스칼라 투영)
                projection = d1.echo.dot(d2.echo)
                norm_sq = (projection * projection.reverse()).data.get(0, 0.0)
                res = math.sqrt(max(0.0, norm_sq))
                
                if res >= self.threshold:
                    tensions.append((d1, d2, res))
                    
        # 가장 강한 텐션부터 정렬
        tensions.sort(key=lambda x: x[2], reverse=True)
        return tensions

    def expand_universe(self, max_new_nodes: int = 3):
        """
        [마인드맵 팽창 (Fractal Expansion)]
        강한 텐션을 지닌 두 개념을 융합하여, 두 개념의 상위 카테고리(Higher Intent) 역할을 하는
        새로운 차원의 노드를 생성하고 우주에 주입한다.
        """
        tensions = self.find_high_tensions()
        new_nodes_created = 0
        
        for d1, d2, res in tensions:
            if new_nodes_created >= max_new_nodes:
                break
                
            # 새로운 상위 목적성의 이름 생성 (마인드맵 가지치기)
            # 예: "time"과 "space"가 강하게 얽히면 "Intent_[time+space]" 탄생
            new_concept_name = f"Intent_[{d1.content} * {d2.content}]"
            
            # 이미 존재하는지 확인
            if new_concept_name in self.universe._content_map:
                continue
                
            # 1. 기하학적 융합 (Wedge Product를 통한 차원 상승)
            # 단순 덧셈이 아니라, 두 벡터가 이루는 평면(2-Vector) 이상의 상위 차원으로 확장
            # 1060 3GB 파이썬 최적화를 위해 임시로 덧셈 기반의 합성을 통해 상위 축(Axis)을 할당
            new_data = {}
            for k, v in d1.echo.data.items():
                new_data[k] = new_data.get(k, 0.0) + v
            for k, v in d2.echo.data.items():
                new_data[k] = new_data.get(k, 0.0) + v
                
            # 차원을 확장하는 고유성(직교성) 부여를 위해 상위 비트에 에너지를 추가
            new_axis = 1 << ((len(self.universe.data) + self.expansion_count) % 16)
            new_data[new_axis] = new_data.get(new_axis, 0.0) + 0.5
            
            new_mv = self._normalize(Multivector(new_data, d1.echo.signature if hasattr(d1.echo, 'signature') else (d1.echo.p, d1.echo.q)))
            
            # 2. 우주에 팽창 주입
            self.universe.inject_rotor(new_concept_name, new_mv)
            self.expansion_count += 1
            new_nodes_created += 1
            
            print(f"[팽창] '{d1.content}'와 '{d2.content}'의 텐션({res:.3f})이 폭발하여 새로운 차원 '{new_concept_name}'을 창조했습니다.")
