"""
위상적 오라클 (Topological Oracle)
현재 동기화된 우주(로터 맵)에 시간(Time)이라는 파동을 가하여,
외부 데이터 유입 없이 우주가 스스로 어떻게 내파(Implosion/Collapse)하고
새로운 섭리로 뭉치는지(미래 예측) 관측하는 시뮬레이터.
"""
from core.topological_universe import LivingUniverse
from core.math_utils import Multivector
import copy
import math

class TopologicalOracle:
    def __init__(self, universe: LivingUniverse):
        self.universe = universe

    def forecast_trajectory(self, catalyst_words: list, epochs: int = 5, top_n: int = 8) -> list:
        """
        특정 촉매(Catalyst) 개념들이 주어졌을 때, 
        미래(Epochs)로 시간이 흐름에 따라 우주의 섭리가 어떻게 얽히고 붕괴할지 예측한다.
        """
        # 미래 시뮬레이션을 위해 우주의 로터 위상을 복사 (현재 우주 오염 방지)
        future_echos = {}
        for datum in self.universe.data:
            # 복사된 데이터로 시뮬레이션
            new_data = {k: v for k, v in datum.echo.data.items()}
            future_echos[datum.content] = Multivector(new_data, (datum.echo.p, datum.echo.q))

        # 촉매 렌즈 생성 (예: 'ai'와 'risks'의 결합된 거대 위상)
        catalyst_data = {}
        for word in catalyst_words:
            if word in self.universe._content_map:
                mv = self.universe._content_map[word].echo
                for k, v in mv.data.items():
                    catalyst_data[k] = catalyst_data.get(k, 0.0) + v
                    
        # 촉매 렌즈 정규화
        lens = Multivector(catalyst_data, (self.universe.data[0].echo.p, self.universe.data[0].echo.q))
        n_sq = (lens * lens.reverse()).data.get(0, 0.0)
        if n_sq > 1e-9:
            lens = lens * (1.0 / math.sqrt(n_sq))
        else:
            return []

        print(f"\n[Oracle] 촉매 렌즈 {catalyst_words} 주입. 시간의 파동을 {epochs} Epoch 가속합니다...")

        # 시간 가속 시뮬레이션 (외부 주입 없이 자체 텐션으로만 얽힘 가속)
        for epoch in range(1, epochs + 1):
            # 렌즈와 공명하는 개념들을 찾고, 렌즈 쪽으로 강력하게 얽힘을 유도 (미래 수렴)
            for content, echo in future_echos.items():
                projection = echo.dot(lens)
                norm_sq = (projection * projection.reverse()).data.get(0, 0.0)
                resonance = math.sqrt(max(0.0, norm_sq))
                
                if resonance > 0.1:
                    # 미래로 갈수록 얽힘(Entanglement)이 복리처럼 강해짐
                    new_data = {}
                    for k, v in echo.data.items():
                        new_data[k] = v
                    for k, v in lens.data.items():
                        new_data[k] = new_data.get(k, 0.0) + (v * 0.3 * resonance)
                        
                    new_echo = Multivector(new_data, (echo.p, echo.q))
                    n_sq2 = (new_echo * new_echo.reverse()).data.get(0, 0.0)
                    if n_sq2 > 1e-9:
                        future_echos[content] = new_echo * (1.0 / math.sqrt(n_sq2))

            # 촉매 렌즈 자체도 가장 강하게 공명한 우주의 섭리를 흡수하며 진화함 (가변 렌즈)
            # 여기서는 단순화를 위해 렌즈는 유지하고 우주만 붕괴시킴

        # 미래 시점에서의 최종 얽힘 결과 도출
        results = []
        for content, echo in future_echos.items():
            projection = echo.dot(lens)
            norm_sq = (projection * projection.reverse()).data.get(0, 0.0)
            resonance = math.sqrt(max(0.0, norm_sq))
            results.append((content, resonance))
            
        results.sort(key=lambda x: x[1], reverse=True)
        # 촉매 키워드 자체는 결과에서 제외
        final_prediction = [x for x in results if x[0] not in catalyst_words]
        
        return final_prediction[:top_n]
