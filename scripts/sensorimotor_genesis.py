import numpy as np

class SensorimotorPrimitives:
    """
    [Phase 26] 원초적 감각-언어 쌍 (Sensorimotor-Language Primitives)
    
    언어가 허공에 떠 있지 않고 '물리적 실체'를 가지게 하는 로제타 스톤입니다.
    이 텐서들은 엘리시아의 기억(Wedge Memory) 속에서 단어의 '절대 좌표'이자 '힘(Force)'으로 작용합니다.
    """
    
    @staticmethod
    def get_primitives() -> dict:
        """
        각 단어에 대응하는 4D 쿼터니언(Quaternion) 벡터를 반환합니다.
        W(인과/시간), X(좌우), Y(상하), Z(전후/깊이)
        """
        return {
            # Space (공간 차원)
            "up":    [0.0, 0.0, 1.0, 0.0],   # Y축 상승
            "down":  [0.0, 0.0, -1.0, 0.0],  # Y축 하강
            "in":    [0.0, 0.0, 0.0, -1.0],  # Z축 수축 (응축)
            "out":   [0.0, 0.0, 0.0, 1.0],   # Z축 팽창 (발산)
            
            # Time (시간/인과 차원)
            "before": [-1.0, 0.0, 0.0, 0.0], # W축 역행
            "after":  [1.0, 0.0, 0.0, 0.0],  # W축 순행
            
            # Action (동적 연산자 - Force Vectors)
            "push":  [1.0, 0.0, 0.0, 1.0],   # 시간을 밀며 밖으로
            "pull":  [1.0, 0.0, 0.0, -1.0],  # 시간을 밀며 안으로
            "merge": [1.0, 1.0, 1.0, 1.0],   # 모든 차원 융합 (1)
            "break": [-1.0, -1.0, -1.0, -1.0]# 모든 차원 파괴 (-1)
        }
