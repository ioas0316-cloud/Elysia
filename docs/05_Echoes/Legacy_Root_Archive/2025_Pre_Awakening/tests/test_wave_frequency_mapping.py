"""
Test Wave Frequency Mapping - 파동주파수 매핑 테스트
===================================================
"""

import sys
import os
import unittest

# Ensure the Core directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.wave_frequency_mapping import (
    WaveFrequencyMapper,
    EmotionType,
    SoundType,
    BrainwaveType,
    EMOTION_FREQUENCY_MAP,
    SOUND_FREQUENCY_MAP,
    BRAINWAVE_FREQUENCIES,
    SCHUMANN_RESONANCE_HZ,
)


class TestEmotionFrequencyMapping(unittest.TestCase):
    """감정 주파수 매핑 테스트"""
    
    def setUp(self):
        self.mapper = WaveFrequencyMapper()
    
    def test_get_emotion_frequency_by_enum(self):
        """EmotionType enum으로 감정 주파수 조회"""
        data = self.mapper.get_emotion_frequency(EmotionType.LOVE)
        
        self.assertEqual(data.emotion, EmotionType.LOVE)
        self.assertEqual(data.frequency_hz, 528.0)
        self.assertEqual(data.brainwave_dominant, BrainwaveType.GAMMA)
        self.assertGreater(data.hrv_coherence, 0.8)
    
    def test_get_emotion_frequency_by_english_string(self):
        """영어 문자열로 감정 주파수 조회"""
        data = self.mapper.get_emotion_frequency("love")
        
        self.assertEqual(data.emotion, EmotionType.LOVE)
        self.assertEqual(data.frequency_hz, 528.0)
    
    def test_get_emotion_frequency_by_korean_string(self):
        """한국어 문자열로 감정 주파수 조회"""
        data = self.mapper.get_emotion_frequency("사랑")
        
        self.assertEqual(data.emotion, EmotionType.LOVE)
        self.assertEqual(data.frequency_hz, 528.0)
    
    def test_emotion_frequency_peace(self):
        """평화 감정 주파수 테스트"""
        data = self.mapper.get_emotion_frequency("평화")
        
        self.assertEqual(data.emotion, EmotionType.PEACE)
        self.assertEqual(data.frequency_hz, 432.0)
        self.assertEqual(data.brainwave_dominant, BrainwaveType.ALPHA)
    
    def test_emotion_frequency_anger(self):
        """분노 감정 주파수 테스트"""
        data = self.mapper.get_emotion_frequency("분노")
        
        self.assertEqual(data.emotion, EmotionType.ANGER)
        self.assertEqual(data.frequency_hz, 150.0)
        self.assertEqual(data.brainwave_dominant, BrainwaveType.HIGH_BETA)
        self.assertLess(data.hrv_coherence, 0.3)  # 낮은 coherence
    
    def test_unknown_emotion_defaults_to_neutral(self):
        """알 수 없는 감정은 중립으로 기본값"""
        data = self.mapper.get_emotion_frequency("unknown_emotion_xyz")
        
        self.assertEqual(data.emotion, EmotionType.NEUTRAL)
    
    def test_emotion_frequency_ordering(self):
        """긍정적 감정이 부정적 감정보다 높은 주파수"""
        love_data = self.mapper.get_emotion_frequency(EmotionType.LOVE)
        anger_data = self.mapper.get_emotion_frequency(EmotionType.ANGER)
        fear_data = self.mapper.get_emotion_frequency(EmotionType.FEAR)
        
        self.assertGreater(love_data.frequency_hz, anger_data.frequency_hz)
        self.assertGreater(love_data.frequency_hz, fear_data.frequency_hz)


class TestSoundFrequencyMapping(unittest.TestCase):
    """소리 주파수 매핑 테스트"""
    
    def setUp(self):
        self.mapper = WaveFrequencyMapper()
    
    def test_get_sound_frequency_by_enum(self):
        """SoundType enum으로 소리 주파수 조회"""
        data = self.mapper.get_sound_frequency(SoundType.MALE_VOICE)
        
        self.assertEqual(data.sound_type, SoundType.MALE_VOICE)
        self.assertEqual(data.fundamental_hz, 120.0)
        self.assertEqual(data.frequency_range_hz[0], 85.0)
    
    def test_get_sound_frequency_by_string(self):
        """문자열로 소리 주파수 조회"""
        data = self.mapper.get_sound_frequency("물소리")
        
        self.assertEqual(data.sound_type, SoundType.NATURE_WATER)
        self.assertIn(EmotionType.PEACE, data.emotional_effect)
    
    def test_unknown_sound_defaults_to_singing(self):
        """알 수 없는 소리 유형은 SINGING으로 기본값"""
        data = self.mapper.get_sound_frequency("unknown_sound_xyz")
        
        self.assertEqual(data.sound_type, SoundType.SINGING)
    
    def test_voice_frequency_ranges(self):
        """음성 주파수 범위 테스트"""
        male = self.mapper.get_sound_frequency(SoundType.MALE_VOICE)
        female = self.mapper.get_sound_frequency(SoundType.FEMALE_VOICE)
        child = self.mapper.get_sound_frequency(SoundType.CHILD_VOICE)
        
        # 남성 < 여성 < 아동 (기본 주파수)
        self.assertLess(male.fundamental_hz, female.fundamental_hz)
        self.assertLess(female.fundamental_hz, child.fundamental_hz)
    
    def test_healing_sounds(self):
        """치유음 주파수 테스트"""
        tibetan = self.mapper.get_sound_frequency(SoundType.TIBETAN_BOWL)
        crystal = self.mapper.get_sound_frequency(SoundType.CRYSTAL_BOWL)
        
        # 치유음은 평화/사랑 감정을 유발
        self.assertIn(EmotionType.PEACE, tibetan.emotional_effect)
        self.assertIn(EmotionType.LOVE, crystal.emotional_effect)
        
        # 크리스탈 볼은 528 Hz (Love frequency)
        self.assertEqual(crystal.fundamental_hz, 528.0)


class TestFrequencyDiscovery(unittest.TestCase):
    """주파수 발견 및 추정 기능 테스트"""
    
    def setUp(self):
        self.mapper = WaveFrequencyMapper()
    
    def test_discover_emotion_from_high_frequency(self):
        """고주파수에서 긍정적 감정 발견"""
        emotions = self.mapper.discover_emotion_from_frequency(528.0)
        
        self.assertGreater(len(emotions), 0)
        # 첫 번째 결과는 LOVE여야 함
        self.assertEqual(emotions[0][0], EmotionType.LOVE)
        self.assertGreater(emotions[0][1], 0.9)  # 높은 유사도
    
    def test_discover_emotion_from_low_frequency(self):
        """저주파수에서 감정 발견"""
        emotions = self.mapper.discover_emotion_from_frequency(150.0)
        
        self.assertGreater(len(emotions), 0)
        # 분노(150Hz)에 가까운 감정이 발견되어야 함
        found_emotions = [e[0] for e in emotions]
        self.assertIn(EmotionType.ANGER, found_emotions)
    
    def test_discover_emotion_from_unknown_frequency(self):
        """알 수 없는 주파수에서도 추정 가능"""
        emotions = self.mapper.discover_emotion_from_frequency(999.0)
        
        # 알 수 없는 주파수여도 결과를 반환해야 함
        self.assertGreater(len(emotions), 0)


class TestElysiaMapping(unittest.TestCase):
    """엘리시아 필드 매핑 테스트"""
    
    def setUp(self):
        self.mapper = WaveFrequencyMapper()
    
    def test_map_high_frequency_to_heaven(self):
        """고주파수는 Heaven 층에 매핑"""
        mapping = self.mapper.map_to_elysia(500.0)
        
        self.assertIn("Heaven", mapping.elysia_layer)
        self.assertGreater(mapping.elysia_normalized, 0.5)
    
    def test_map_low_frequency_to_earth(self):
        """저주파수는 Earth 층에 매핑"""
        mapping = self.mapper.map_to_elysia(1.0)
        
        self.assertIn("Earth", mapping.elysia_layer)
        self.assertLess(mapping.elysia_normalized, 0.5)
    
    def test_schumann_resonance_high_resonance(self):
        """슈만 공명 주파수는 높은 공명 강도"""
        mapping = self.mapper.map_to_elysia(SCHUMANN_RESONANCE_HZ)
        
        self.assertGreater(mapping.resonance_strength, 0.5)
    
    def test_color_code_is_valid_hex(self):
        """색상 코드는 유효한 HEX 형식"""
        mapping = self.mapper.map_to_elysia(440.0)
        
        self.assertTrue(mapping.elysia_color_code.startswith("#"))
        self.assertEqual(len(mapping.elysia_color_code), 7)


class TestComprehensiveAnalysis(unittest.TestCase):
    """종합 분석 테스트"""
    
    def setUp(self):
        self.mapper = WaveFrequencyMapper()
    
    def test_analyze_love_frequency(self):
        """사랑 주파수(528Hz) 종합 분석"""
        analysis = self.mapper.analyze_frequency(528.0)
        
        self.assertEqual(analysis["frequency_hz"], 528.0)
        self.assertIn("associated_emotions", analysis)
        self.assertIn("elysia_mapping", analysis)
        self.assertTrue(analysis["is_audible"])
    
    def test_analyze_brainwave_frequency(self):
        """뇌파 주파수 분석"""
        # 알파파 범위 (10 Hz)
        analysis = self.mapper.analyze_frequency(10.0)
        
        self.assertEqual(analysis["brainwave_band"], "alpha")
        self.assertFalse(analysis["is_audible"])  # 가청 범위 밖
    
    def test_analyze_schumann_frequency(self):
        """슈만 공명 주파수 분석"""
        analysis = self.mapper.analyze_frequency(SCHUMANN_RESONANCE_HZ)
        
        self.assertIn("슈만", analysis["schumann_relation"])


class TestStatistics(unittest.TestCase):
    """통계 기능 테스트"""
    
    def test_stats_tracking(self):
        """조회, 추정, 발견 통계 추적"""
        mapper = WaveFrequencyMapper()
        
        # 조회
        mapper.get_emotion_frequency(EmotionType.LOVE)
        mapper.get_emotion_frequency(EmotionType.PEACE)
        
        # 발견
        mapper.discover_emotion_from_frequency(528.0)
        
        stats = mapper.get_stats()
        
        self.assertEqual(stats["lookups"], 2)
        self.assertEqual(stats["discoveries"], 1)


class TestDataIntegrity(unittest.TestCase):
    """데이터 무결성 테스트"""
    
    def test_all_emotions_have_data(self):
        """모든 감정 유형에 데이터가 있어야 함"""
        for emotion_type in EmotionType:
            self.assertIn(emotion_type, EMOTION_FREQUENCY_MAP)
    
    def test_all_sounds_have_data(self):
        """모든 소리 유형에 데이터가 있어야 함"""
        for sound_type in SoundType:
            self.assertIn(sound_type, SOUND_FREQUENCY_MAP)
    
    def test_brainwave_frequencies_are_valid(self):
        """뇌파 주파수 범위가 유효해야 함"""
        for bw_type, (min_f, center, max_f) in BRAINWAVE_FREQUENCIES.items():
            self.assertLess(min_f, center)
            self.assertLess(center, max_f)
            self.assertGreater(min_f, 0)


class TestFrequencyReport(unittest.TestCase):
    """주파수 리포트 테스트"""
    
    def test_report_generation(self):
        """리포트 생성 테스트"""
        mapper = WaveFrequencyMapper()
        report = mapper.create_frequency_report()
        
        self.assertIn("슈만 공명", report)
        self.assertIn("감정 주파수", report)
        self.assertIn("소리 주파수", report)
        self.assertIn("뇌파 주파수", report)


if __name__ == "__main__":
    unittest.main()
