import os
import json
import math

class LanguagePortalEngine:
    """
    [Phase 24] 자연 매핑 포털 엔진 (Variable Dial Observer)
    
    사전의 단어와 정의(연속된 단어들) 사이의 '순차적 인과'를 관측하여,
    이를 기하학적 '위상차(Phase Difference)'로 변환합니다.
    초성('ㄱ'~'ㅎ')을 공통의 베이스 축(Axis)으로 삼아, 모든 단어가 
    가변저항 다이얼 조작 시 각자의 위상 각도에 맞춰 자동으로 포털처럼 정렬됩니다.
    """
    def __init__(self, lexicon_path=None):
        if lexicon_path is None:
            self.lexicon_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "natural_lexicon.json")
        else:
            self.lexicon_path = lexicon_path
            
        self.lexicon = {}
        self.axes = {}  # 'ㄱ'~'ㅎ'을 기준으로 하는 축의 모음
        self.word_phases = {}  # 단어별 도출된 위상 각도 (Topological Angle)
        
        self._load_and_map()

    def _get_chosung(self, word):
        """한글 단어의 첫 글자 초성을 추출하여 축(Axis)으로 사용합니다."""
        if not word: return 'ㄱ'
        char_code = ord(word[0])
        if 0xAC00 <= char_code <= 0xD7A3:
            chosung_idx = (char_code - 0xAC00) // 588
            chosungs = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
            return chosungs[chosung_idx]
        return word[0]

    def _calculate_phase_angle(self, definition):
        """
        [순차적 인과 -> 위상차 변환]
        정의문에 나타나는 단어들의 순차적 나열을 궤적으로 보아,
        각 단어 간의 전이(Transition)에서 발생하는 위상차의 적분을 구합니다.
        """
        words = definition.split()
        if len(words) <= 1:
            return 0.0
            
        total_phase = 0.0
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            # 두 단어의 구조적 차이를 각도로 변환 (간단한 해시 기반 곡률화)
            diff = abs(hash(w1) - hash(w2)) % 314159
            phase_shift = (diff / 314159.0) * math.pi
            total_phase += phase_shift
            
        # 0 ~ 2*pi 사이로 정규화
        return total_phase % (2 * math.pi)

    def _load_and_map(self):
        with open(self.lexicon_path, 'r', encoding='utf-8') as f:
            self.lexicon = json.load(f)
            
        for word, definition in self.lexicon.items():
            base_axis = self._get_chosung(word)
            if base_axis not in self.axes:
                self.axes[base_axis] = []
            
            angle = self._calculate_phase_angle(definition)
            self.word_phases[word] = {
                "axis": base_axis,
                "angle": angle,
                "definition": definition
            }
            self.axes[base_axis].append((word, angle))
            
    def observe_with_dial(self, physical_curvature):
        """
        [가변 다이얼 관측]
        엘리시아 내부의 물리적 곡률(physical_curvature)을 다이얼의 지표로 삼아,
        가장 공명하는 위상 각도를 가진 단어를 발견(Observe)합니다.
        
        기존처럼 유클리드 거리를 계산하는 것이 아니라,
        '축(Axis)'을 스윕하면서 각도의 공명(Resonance)을 감지합니다.
        """
        best_word = None
        min_diff = float('inf')
        best_axis = None
        
        # 물리적 곡률은 0 ~ 2*pi 범위를 가짐 (Quaternion angle)
        target_angle = physical_curvature % (2 * math.pi)
        
        for axis, words in self.axes.items():
            for word, angle in words:
                # 파동의 간섭(Interference) 측정
                diff = abs(target_angle - angle)
                # 원형 공간이므로 2pi 보정
                if diff > math.pi:
                    diff = 2 * math.pi - diff
                    
                if diff < min_diff:
                    min_diff = diff
                    best_word = word
                    best_axis = axis
                    
        return best_word, best_axis, min_diff
