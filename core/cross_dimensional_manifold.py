import math
from typing import Tuple, List
from core.math_utils import Multivector, Quaternion

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
        
    def refract_tension(self, tension: float, phase_angles: list, base_q: Quaternion) -> Tuple[Quaternion, str]:
        """
        [의미망/추상화 계층 (Semantic Abstraction Layer)]
        하드웨어의 순수 텐션(base_q)이 최상위 인지(F6)로 올라가기 전,
        '의미(Semantics)와 의도(Intent)'를 기하학적으로 처리하는 다차원 매니폴드를 거침.
        단순한 1:1 유니코드 매핑이 아니라, 특정 위상 텐션이 '거짓', '은유', '역설'의 
        맥락적 각도를 가지도록 왜곡(굴절)시킴.
        """
        safe_phases = []
        if isinstance(phase_angles, list):
            for p in phase_angles:
                try: safe_phases.append(float(p))
                except (ValueError, TypeError): safe_phases.append(0.0)
        
        # 1. 점(Point, Grade 1): 물리적 전압 (e1)
        m_voltage = Multivector({1: tension}, self.signature)
        
        # 2. 선의 충돌과 면(Bivector, Grade 2)의 형성: 논리 (e2)
        p2 = sum(safe_phases[:5]) % 1.0 if safe_phases else 0.5
        m_logic = Multivector({2: p2}, self.signature)
        m_area = m_voltage ^ m_logic
        
        # 3. 입체(Trivector, Grade 3)로의 팽창: 수학/위상 (e3)
        p3 = sum(safe_phases[5:10]) % 1.0 if safe_phases else 0.5
        m_math = Multivector({4: p3}, self.signature)
        m_volume = m_area ^ m_math
        
        # 4. 4차원 초입체(4-Vector, Grade 4)로의 직조: 구문/코드 (e4)
        p4 = sum(safe_phases[10:15]) % 1.0 if safe_phases else 0.5
        m_syntax = Multivector({8: p4}, self.signature)
        m_hyper = m_volume ^ m_syntax
        
        # 5. 유사스칼라(Pseudoscalar, Grade 5) 우주의 완성: 언어/의미 (e5)
        p5 = sum(safe_phases[15:20]) % 1.0 if safe_phases else 0.5
        m_semantic = Multivector({16: p5}, self.signature)
        pseudoscalar_mv = m_hyper ^ m_semantic
        
        # Pseudoscalar 값(Grade 5, mask=31) 읽기
        I_value = pseudoscalar_mv.data.get(31, 0.0)
        
        # 회전(Rotor) 연산을 통한 최종 진동 가해 (밀도 증폭)
        dual_mv = pseudoscalar_mv.dual()
        scalar_density = abs(dual_mv.data.get(0, I_value)) * 100.0
        
        # 밀도에 따른 최종 단어 도출 (의도/함의의 종류)
        idx = min(len(self.pseudoscalar_map) - 1, int(scalar_density) % len(self.pseudoscalar_map))
        core_intent = self.pseudoscalar_map[idx]
        
        # 매니폴드 굴절 (Refraction)
        # scalar_density를 기반으로 비틀림 각도(Twist Angle)를 생성
        twist_angle = (scalar_density % (2 * math.pi))
        
        # 의미론적 함의(은유/역설)를 담은 트위스트 로터 생성
        twist_rotor = Quaternion(math.cos(twist_angle), math.sin(twist_angle * 0.5), math.sin(twist_angle), 0).normalize()
        
        # 기저 텐션(base_q)을 매니폴드 공간에서 회전시킴 (굴절)
        refracted_q = twist_rotor * base_q * twist_rotor.inverse
        
        return refracted_q.normalize(), core_intent
