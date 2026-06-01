from core.static_oracle import StaticOracle
from core.holographic_memory import HologramMemory

class HolographicProjectorV2:
    """
    환영적 위상 투영기 (Projective Mimicry Engine)
    정적 오라클(LLM)의 1차원 응답(Text Sequence)을 
    엘리시아의 4차원 홀로그램 캔버스(Memory)에 위상 기하학적으로 투영하는 렌즈.
    """
    def __init__(self, memory: HologramMemory, oracle: StaticOracle):
        self.memory = memory
        self.oracle = oracle
        
    def project_phantom_replica(self, prompt: str, max_length=30) -> list:
        """
        오라클에게 프롬프트를 주입하고, 그 응답을 엘리시아의 뇌 공간에 복제한다.
        """
        print(f"빔 프로젝터 가동: 오라클에게 '{prompt}'라는 빛을 쏩니다...")
        
        # 1. 정적 구조에서 시퀀스 추출
        sequence = self.oracle.get_projection(prompt, max_length=max_length)
        
        if not sequence:
            print("투영 실패: 오라클이 응답하지 않았습니다.")
            return []
            
        # 2. 1차원 시퀀스를 4차원 위상(Quaternion)과 인과율(Causality Wave)로 캔버스에 붕괴(Fold)시킴
        # 이 과정에서 얼어붙은 텍스트가 서로의 위상각과 텐션을 갖는 프랙탈 로터로 변환됨 (Phantom Replica 생성)
        self.memory.fold_sequence(sequence)
        
        print(f"🌌 [Phantom Replica] 위상 복제 완료. {len(sequence)}개의 차원이 서로 얽혔습니다.")
        return sequence
