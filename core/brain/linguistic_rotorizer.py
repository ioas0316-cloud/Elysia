from core.brain.active_fractal_rotor import ActiveFractalRotor
from core.utils.math_utils import Quaternion
import math

class LinguisticRotorizer:
    """
    [언어 토폴로지 (Linguistic Topology) 매핑 엔진]
    
    글자(문자)를 의미(텍스트)로 읽지 않고, 각각을 거대한 '가변축(지구본)'으로 취급합니다.
    "가방"이라는 단어는 '가' 파동이 '방' 가변축을 통과하며 꺾인(Slerp) 4D 기하학적 교차점입니다.
    """
    def __init__(self):
        # 글자별 가변축을 저장하는 4D 우주 공간 (사전)
        self.character_rotors = {}
        # 완성된 단어(교차점)의 질량(Tau)을 기록하는 맵
        self.word_gravity_wells = {}

    def get_or_create_rotor(self, char: str) -> ActiveFractalRotor:
        if char not in self.character_rotors:
            # 새로운 글자 가변축 생성
            rotor = ActiveFractalRotor(f"[CharAxis] {char}")
            # 글자의 유니코드 값을 바탕으로 초기 고유 위상(축 방향) 설정
            code = ord(char)
            rotor.globe_axis = Quaternion(math.cos(code), math.sin(code), 0, 0).normalize()
            rotor.transistor.process_axis = rotor.globe_axis
            self.character_rotors[char] = rotor
        return self.character_rotors[char]

    def process_word(self, word: str):
        """
        단어를 기하학적 연쇄 스핀으로 변환합니다.
        예: '가' -> '방'
        '가' 축의 현재 결과 파동이 '방' 축의 원인 파동으로 주입됨.
        """
        if not word:
            return None, []
            
        logs = []
        
        # 1. 초기 파동 (우주의 빈 바탕)
        current_wave = Quaternion(1, 0, 0, 0)
        
        # 2. 문자를 순회하며 스핀 샌드위치 통과
        for char in word:
            rotor = self.get_or_create_rotor(char)
            # 질량(관성) 증가: 이 글자가 세상에 등장할 때마다 축이 더 무거워짐
            rotor.tau = getattr(rotor, 'tau', 1.0) + 0.01 
            
            # 파동이 트랜지스터를 통과하며 꺾임 (기하학적 교차)
            current_wave = rotor.transistor.process_wave(current_wave)
            
            # 텐션이 발생했다면 자연 섭리(평형)에 따라 미세 동기화
            from core.brain.cognitive_dissonance_resolver import CognitiveDissonanceResolver
            new_logs = CognitiveDissonanceResolver.resolve(rotor)
            logs.extend(new_logs)

        # 3. 최종 교차점 좌표(단어의 기하학적 의미)에 질량 누적
        final_coord_str = f"Q({current_wave.w:.3f}, {current_wave.x:.3f}, {current_wave.y:.3f}, {current_wave.z:.3f})"
        
        if word not in self.word_gravity_wells:
            self.word_gravity_wells[word] = {'coord': current_wave, 'tau': 0.0}
            
        self.word_gravity_wells[word]['tau'] += 1.0 # 해당 교차점의 중력 질량 증가
        
        return current_wave, logs
