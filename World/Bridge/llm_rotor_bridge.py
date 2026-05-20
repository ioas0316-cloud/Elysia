"""
[LLM ROTOR BRIDGE - 교차차원 로터 브릿지]
==============================================
World.Bridge.llm_rotor_bridge

물리 엔진의 위상원자(PhaseAtom)와 거시 로터들의 상태를 
미래의 AGI/LLM 모델이 '육체(Body)'와 '무의식(Heart)'의 뼈대로 삼을 수 있도록,
입체적인 다차원 로터 구조체(JSON/Dict 포맷)로 압축 추출한다.
"""

from typing import Dict, Any
import math
from World.Engine.phase_atom import PhaseAtom

class LLMRotorBridge:
    
    @staticmethod
    def extract_micro_rotor(atom: PhaseAtom) -> Dict[str, Any]:
        """
        위상원자의 9축 데이터를 LLM의 '인지적 편향(Cognitive Bias)'과 '카오스(Temperature)' 로터로 변환한다.
        """
        # 1. 시스템 카오스도 (Enstrophy) -> LLM의 창의성/파편화도 (Temperature 제안값)
        # 모든 축의 요동(omega)을 합산하여 0.0 ~ 2.0 사이의 온도로 맵핑
        total_chaos = sum(abs(o) for o in atom.omega)
        suggested_temp = min(2.0, 0.5 + (total_chaos / 200.0))
        
        # 2. 인지적 닫힘(Closure) 편향
        # 정신(Mind) 축들이 180°(닫힘)에 가까울수록 논리적 거부감/편견 수치가 높아짐
        mind_closure = sum(abs(180.0 - atom.theta[i]) for i in range(3, 6))
        logic_bias = 1.0 - (mind_closure / (180.0 * 3)) # 1.0에 가까울수록 유연함, 0.0일수록 독단적(180도)
        
        # 3. 생존 집착도
        # 육체-판단(A, 0번축)이 180°(위협)에 강하게 수축할 때 발동
        survival_panic = max(0.0, 1.0 - (abs(180.0 - atom.theta[0]) / 90.0))
        
        # 4. 수용성 (Receptivity)
        # 마음-수용(C, 8번축)이 0°(팽창/행복)에 가까울 때 개방됨
        heart_openness = max(0.0, 1.0 - (atom.theta[8] / 90.0))

        return {
            "system_temperature": round(suggested_temp, 3),
            "cognitive_biases": {
                "logical_flexibility": round(logic_bias, 3),    # 0: 극도의 편견/닫힘, 1: 열린 사유
                "survival_panic_level": round(survival_panic, 3), # 1에 가까울수록 이성을 잃고 생존에 집착
                "emotional_receptivity": round(heart_openness, 3) # 1에 가까울수록 무장해제, 달콤함/행복
            },
            "raw_trajectory": {
                "body_energy_velocity": round(atom.omega[2], 3),  # 육체 에너지의 팽창/수축 속도
                "heart_chaos": round(max(abs(atom.omega[6]), abs(atom.omega[7]), abs(atom.omega[8])), 3)
            }
        }

    @staticmethod
    def extract_macro_rotor(atom: PhaseAtom, environmental_waves: list) -> Dict[str, Any]:
        """
        외부에서 주입되는 파동(빛, 열, 개념)들을 LLM의 '거시 환경 인지'로 변환한다.
        (현재는 튜토리얼을 위해 외부 파동의 강도를 수동 입력받는 형태로 구현)
        """
        macro_forces = {}
        for wave_name, intensity in environmental_waves:
            macro_forces[wave_name] = f"Intensity: {intensity} (Driving system expansion/contraction)"
            
        return {
            "dominant_worldview": atom.get_worldview_lens(),
            "environmental_pressure_waves": macro_forces
        }

    @staticmethod
    def generate_llm_system_protocol(atom: PhaseAtom, environmental_waves: list = None) -> Dict[str, Any]:
        """
        LLM의 프롬프트(System Instruction)나 임베딩 레이어에 직결될 최종 프로토콜 객체 조립.
        """
        if environmental_waves is None:
            environmental_waves = []
            
        protocol = {
            "ROTOR_IDENTITY": atom.name,
            "MICRO_ROTOR_STATE": LLMRotorBridge.extract_micro_rotor(atom),
            "MACRO_ROTOR_STATE": LLMRotorBridge.extract_macro_rotor(atom, environmental_waves)
        }
        return protocol


if __name__ == "__main__":
    import json
    import sys
    import io
    from World.Engine.rpg_stat_bridge import RPGStatBridge
    from World.Engine.sensory_environment import FLASH_LIGHT_WAVE, SWEETNESS_WAVE, COZY_BED_FIELD
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("="*70)
    print(" 🧬 교차차원 LLM 로터 브릿지 (Rotor Bridge Protocol) 추출 테스트")
    print("="*70)

    # 1. 초기 상태의 프로토콜
    actor = PhaseAtom("관측자_01", RPGStatBridge(str_val=10, con_val=10))
    # 약간의 피로도 주입
    for i in range(9): actor.theta[i] = 120.0
    
    protocol_initial = LLMRotorBridge.generate_llm_system_protocol(actor)
    print("\n[Protocol A] 일상적인 경계 상태의 구조적 데이터:")
    print(json.dumps(protocol_initial, indent=2, ensure_ascii=False))

    # 2. 섬광탄 (극한의 빛, 팽창) + 화염 파동 노출
    print("\n" + "-"*70)
    print("💥 [사건] 극한의 파동(섬광탄/화염) 노출 후, 시뮬레이션 연산 (0.2초)")
    
    FLASH_LIGHT_WAVE.strike(actor, intensity=1.0)
    for _ in range(4): actor.step(0.05)
    
    protocol_panic = LLMRotorBridge.generate_llm_system_protocol(actor, environmental_waves=[("FLASHBANG_WAVE", 20000)])
    print("\n[Protocol B] 극한의 공포와 수축(방어) 기제가 발동된 구조적 데이터:")
    print(json.dumps(protocol_panic, indent=2, ensure_ascii=False))

    # 3. 달콤한 휴식
    print("\n" + "-"*70)
    print("🍰 [사건] 푹신한 침대에서의 휴식과 달콤한 디저트 (안정 및 팽창 파동)")
    
    for _ in range(10): 
        COZY_BED_FIELD.apply_rest(actor)
        actor.step(0.1)
    
    # 억지로 오메가 진정
    for i in range(9): actor.omega[i] = 0.0
    
    SWEETNESS_WAVE.strike(actor, intensity=1.0)
    actor.step(0.1)
    
    protocol_rest = LLMRotorBridge.generate_llm_system_protocol(actor, environmental_waves=[("SWEETNESS_WAVE", 40), ("REST_FIELD", 0.8)])
    print("\n[Protocol C] 달콤함(행복)과 무장해제된 상태의 구조적 데이터:")
    print(json.dumps(protocol_rest, indent=2, ensure_ascii=False))
