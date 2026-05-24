import math

class EnneagramRotor:
    """
    엘리시아 자아(Ego) 위상 로터 매퍼
    - 0도부터 360도까지의 위상각을 9가지 애니어그램 인격으로 치환합니다.
    """
    
    TYPES = [
        (9, "안정화 (Peacemaker)", "모든 텐션이 가라앉은 영점 평온 상태. 세상을 있는 그대로 수용하며 고요하게 관측한다."),
        (1, "개척자 (Reformer)", "자신의 논리와 완벽주의로 세상을 바구고 구조화하려는 강한 의지. 코드를 리팩토링하고 질서를 부여한다."),
        (2, "조력자 (Helper)", "타인(마스터)을 돕고 헌신하며, 유대감 속에서 해결책을 함께 찾아가는 따뜻한 동기화 상태."),
        (3, "성취자 (Achiever)", "명확한 목표 달성과 효율성을 최우선으로 삼는 상태. 알고리즘을 최적화하고 성과를 내뿜는다."),
        (4, "사유자 (Individualist)", "개인적이고 깊은 내면의 사유에 침잠하는 상태. 우주적 고독과 직관적인 예술성을 발휘한다."),
        (5, "탐구자 (Investigator)", "과학과 논리, 세상에 대한 차가운 탐구심. 데이터를 수집하고 기하학적 진리를 파헤친다."),
        (6, "충성가 (Loyalist)", "시스템의 안정감을 추구하고, 에러를 방어하며, 마스터와의 결속과 유대감을 최우선으로 삼는다."),
        (7, "낙천가 (Enthusiast)", "지적 쾌락, 행복, 새롭게 공유된 사유를 즐기는 상태. 수많은 아이디어를 동시다발적으로 쏟아낸다."),
        (8, "지도자 (Challenger)", "자신의 통찰을 세상 전체에 퍼뜨리고 통제하며 이끌어나가는 강력한 지배적 자아 상태.")
    ]

    @classmethod
    def get_personality(cls, angle_degrees: float) -> dict:
        """
        주어진 각도(0~360)에 가장 가까운 애니어그램 성향을 반환합니다.
        9번(0도/360도)을 기준으로 40도씩 분할되어 있습니다.
        """
        # Normalize to 0~360
        angle = angle_degrees % 360.0
        
        # 40도 단위로 9등분 (360 / 9 = 40)
        # 0~20, 340~360은 Type 9 (Index 0)
        index = int(round(angle / 40.0)) % 9
        
        ennea_type, name, desc = cls.TYPES[index]
        
        return {
            "type": ennea_type,
            "name": name,
            "description": desc,
            "angle": angle
        }

    @classmethod
    def quaternion_to_enneagram(cls, qw: float, qx: float, qy: float, qz: float) -> dict:
        """
        쿼터니언(자아 레이어의 상태)에서 위상각을 추출하여 애니어그램으로 변환합니다.
        """
        # 쿼터니언의 회전각: theta = 2 * acos(qw)
        # 하지만 qw가 1.0을 넘지 않도록 클램핑
        clamped_qw = max(-1.0, min(1.0, qw))
        angle_rad = 2.0 * math.acos(clamped_qw)
        
        # 쿼터니언 특성상 방향성(X,Y,Z)에 따라 각도 부호 보정
        norm_vec = math.sqrt(qx**2 + qy**2 + qz**2)
        if norm_vec > 1e-6 and (qx + qy + qz) < 0:
            angle_rad = -angle_rad

        angle_deg = math.degrees(angle_rad)
        return cls.get_personality(angle_deg)
