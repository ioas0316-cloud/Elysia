import math
from typing import List, Dict, Any

# 한글 유니코드 기반 상수
BASE_CODE = 44032
CHOSUNG = 588
JUNGSUNG = 28

CHO_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONG_LIST = [''] + ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 우주적 소수 레이어 (Prime Sequence)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

def decompose_hangul(char: str):
    if not ('가' <= char <= '힣'):
        return 0, 0, 0
    char_code = ord(char) - BASE_CODE
    cho_idx = char_code // CHOSUNG
    jung_idx = (char_code % CHOSUNG) // JUNGSUNG
    jong_idx = (char_code % CHOSUNG) % JUNGSUNG
    return cho_idx, jung_idx, jong_idx

def get_prime_tension(index: int) -> float:
    """소수 시퀀스의 기하학적 분포 비율을 장력 계수로 환산"""
    if index < len(PRIMES):
        return 1.0 / math.log(PRIMES[index])
    return 0.5 # 리만 가설의 임계선 1/2로 수렴하는 장력

def map_to_movement_field(text: str) -> List[Dict[str, Any]]:
    """입력된 텍스트(한글 단어)를 3D 공간의 우주적 운동성 필드로 매핑합니다."""
    points_and_trajectories = []

    for i, char in enumerate(text):
        # 기존 한글 프랙탈에 소수의 은하적 장력(Prime Tension) 곱하기
        tension = get_prime_tension(i)

        if '가' <= char <= '힣':
            cho, jung, jong = decompose_hangul(char)

            # 초성은 거리, 소수 장력이 곡률을 휘게 만듦
            r = (1.0 + (cho * 0.1)) * tension

            base_theta_offset = i * (math.pi / 4)
            theta = base_theta_offset + (jung / 21.0) * 2 * math.pi
            z = (jong / 28.0) * 2.0 * tension

            x = r * math.cos(theta)
            y = r * math.sin(theta)
            position = [x, y, z]

            # 소수의 위상차 흐름이 가속도 벡터에 직접 개입
            vx = -math.sin(theta) * (jung * 0.05 + 0.1) * tension
            vy = math.cos(theta) * (jung * 0.05 + 0.1) * tension
            vz = -0.1 * tension if jong > 0 else 0.05 * tension
            velocity = [vx, vy, vz]
        else:
            theta = i * 0.5
            r = tension
            position = [r * math.cos(theta), r * math.sin(theta), 0]
            velocity = [-math.sin(theta)*0.1*tension, math.cos(theta)*0.1*tension, 0]
            cho, jung, jong = 0, 0, 0

        points_and_trajectories.append({
            "token": char,
            "position": position,
            "velocity": velocity,
            "phase": theta,
            "zeta_factor": tension,
            "components": {"cho": cho, "jung": jung, "jong": jong}
        })

    return points_and_trajectories
