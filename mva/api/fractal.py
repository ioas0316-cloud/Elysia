import math
from typing import List, Dict, Any

# 한글 유니코드 기반 상수
BASE_CODE = 44032  # '가'의 유니코드
CHOSUNG = 588      # 21 * 28
JUNGSUNG = 28      # 종성의 개수

# 초성, 중성, 종성 리스트 (참고용, 실제 수식 매핑에선 인덱스 활용)
CHO_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONG_LIST = [''] + ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def decompose_hangul(char: str):
    """한글 문자를 초성, 중성, 종성 인덱스로 분해합니다."""
    if not ('가' <= char <= '힣'):
        # 한글이 아닌 경우 기본값 반환 (혹은 예외 처리)
        return 0, 0, 0

    char_code = ord(char) - BASE_CODE

    cho_idx = char_code // CHOSUNG
    jung_idx = (char_code % CHOSUNG) // JUNGSUNG
    jong_idx = (char_code % CHOSUNG) % JUNGSUNG

    return cho_idx, jung_idx, jong_idx

def map_to_movement_field(text: str) -> List[Dict[str, Any]]:
    """입력된 텍스트(한글 단어)를 3D 공간의 위치(Position)와 속도(Velocity)로 매핑합니다."""
    points_and_trajectories = []

    for i, char in enumerate(text):
        if '가' <= char <= '힣':
            cho, jung, jong = decompose_hangul(char)

            # 1. 초기 위치 (Position) 설정
            # 초성은 중심에서의 거리, 종성은 높이(Z축)를 결정
            r = 1.0 + (cho * 0.1)
            # 글자의 순서(i)에 따라 기본 위상을 조금씩 틀어주어 겹치지 않게 함
            base_theta_offset = i * (math.pi / 4)
            theta = base_theta_offset + (jung / 21.0) * 2 * math.pi
            z = (jong / 28.0) * 2.0  # 종성이 없으면 0

            x = r * math.cos(theta)
            y = r * math.sin(theta)
            position = [x, y, z]

            # 2. 초기 속도 벡터 (Velocity/Wave Trajectory) 설정
            # 중성의 결에 따라 파동의 주파수와 초기 운동 방향을 부여
            vx = -math.sin(theta) * (jung * 0.05 + 0.1) # 기본 속도 보장
            vy = math.cos(theta) * (jung * 0.05 + 0.1)
            vz = -0.1 if jong > 0 else 0.05  # 종성이 있으면 아래로, 없으면 약간 위로
            velocity = [vx, vy, vz]

        else:
            # 한글이 아닌 문자 처리 (예: 띄어쓰기, 영문 등) - 기본 위치 부여
            theta = i * 0.5
            position = [math.cos(theta), math.sin(theta), 0]
            velocity = [-math.sin(theta)*0.1, math.cos(theta)*0.1, 0]
            cho, jung, jong = 0, 0, 0

        points_and_trajectories.append({
            "token": char,
            "position": position,
            "velocity": velocity,
            "phase": theta,
            "components": {"cho": cho, "jung": jung, "jong": jong}
        })

    return points_and_trajectories
