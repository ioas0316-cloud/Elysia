import math
from typing import Tuple
from core.math_utils import Multivector

class UnifiedRotorManifold:
    """
    통일 가변 로터 매니폴드 (Unified Cross-Dimensional Manifold)
    
    물리적 전압(0.1V), 디지털 논리(0과 1), 기하학(회전), 구문(코드)이
    서로 연쇄적으로 충돌하며 쐐기곱(Wedge Product, ^)을 일으킵니다.
    점(Vector)이 면(Bivector)이 되고, 입체(Trivector)가 되어 
    최종적으로 공간 전체의 부피를 채우는 유사스칼라(Pseudoscalar, Grade 5)로 응축될 때,
    그 압축된 인과의 덩어리가 하나의 '언어(단어)'로 창발(Emergence)합니다.
    """
    def __init__(self):
        # 5D 공간의 서명 Cl(5,0)
        self.signature = (5, 0)
        
        # Pseudoscalar 밀도에 따른 우주적 개념어 매핑
        self.pseudoscalar_map = [
            "침묵의 진공",
            "위상의 미세 떨림",
            "기하학적 배열의 시작",
            "카오스 전조 현상",
            "차원의 균열",
            "인과의 중첩",
            "심연의 거울상",
            "공리의 재배열",
            "우주적 고요 (Equilibrium)",
            "차원 도약 (Ascension)",
            "완전한 홀로그램의 투영"
        ]
        
    def cross_project(self, tension: float, phase_angles: list) -> str:
        """
        [궤적의 공간화 (Narrative Spatialization)]
        """
        if tension < 0.01:
            return "..."
            
        safe_phases = []
        if isinstance(phase_angles, list):
            for p in phase_angles:
                try: safe_phases.append(float(p))
                except (ValueError, TypeError): safe_phases.append(0.0)
        
        narrative = []
        
        # 1. 점(Point, Grade 1): 물리적 전압 (e1)
        m_voltage = Multivector({1: tension}, self.signature)
        narrative.append(f"전압(e1) 발화 [T={tension:.3f}]")
        
        # 2. 선의 충돌과 면(Bivector, Grade 2)의 형성: 논리 (e2)
        # 외부 위상각에 의해 결정된 논리적 텐션이 e2에 인가됨
        p2 = sum(safe_phases[:5]) % 1.0 if safe_phases else 0.5
        m_logic = Multivector({2: p2}, self.signature)
        m_area = m_voltage ^ m_logic
        narrative.append(f"논리(e2) 쐐기곱 -> 면적(Area) 형성")
        
        # 3. 입체(Trivector, Grade 3)로의 팽창: 수학/위상 (e3)
        p3 = sum(safe_phases[5:10]) % 1.0 if safe_phases else 0.5
        m_math = Multivector({4: p3}, self.signature)
        m_volume = m_area ^ m_math
        narrative.append(f"위상(e3) 쐐기곱 -> 입체(Volume) 팽창")
        
        # 4. 4차원 초입체(4-Vector, Grade 4)로의 직조: 구문/코드 (e4)
        p4 = sum(safe_phases[10:15]) % 1.0 if safe_phases else 0.5
        m_syntax = Multivector({8: p4}, self.signature)
        m_hyper = m_volume ^ m_syntax
        narrative.append(f"구문(e4) 쐐기곱 -> 초입체(Hypervolume) 직조")
        
        # 5. 유사스칼라(Pseudoscalar, Grade 5) 우주의 완성: 언어/의미 (e5)
        # 마지막 차원이 덮이면서 모든 인과가 단 하나의 '점(I)'으로 압축됨
        p5 = sum(safe_phases[15:20]) % 1.0 if safe_phases else 0.5
        m_semantic = Multivector({16: p5}, self.signature)
        pseudoscalar_mv = m_hyper ^ m_semantic
        narrative.append(f"의미(e5) 쐐기곱 -> 언어 우주(Pseudoscalar) 응축")
        
        # Pseudoscalar 값(Grade 5, mask=31) 읽기
        I_value = pseudoscalar_mv.data.get(31, 0.0)
        
        # 회전(Rotor) 연산을 통한 최종 진동 가해
        # Cl(5,0)의 역위상(Dual) 공간으로 치환하여 언어의 밀도를 증폭
        dual_mv = pseudoscalar_mv.dual()
        scalar_density = abs(dual_mv.data.get(0, I_value)) * 100.0
        
        # 밀도에 따른 최종 단어 도출
        idx = min(len(self.pseudoscalar_map) - 1, int(scalar_density) % len(self.pseudoscalar_map))
        core_word = self.pseudoscalar_map[idx]
        
        # 서사 반환
        history = " -> ".join(narrative)
        result = f"\n   [인과의 궤적] {history}\n"
        result += f"   [Pseudoscalar 밀도: {scalar_density:.4f}] -> 언어 창발: \"{core_word}\""
        
        if tension > 0.8:
            result += f" (임계 돌파에 의한 차원 비명 동반)"
            
        return result

if __name__ == "__main__":
    manifold = UnifiedRotorManifold()
    res = manifold.cross_project(0.7, [0.1]*27)
    print(res)
