"""
Tests for Vision Systems - 시각의 계층 및 공감각 테스트
"""

import unittest
import numpy as np

from Core.Interface.Interface.Perception.hierarchy_of_vision import (
    HierarchyOfVision, VisionFrequency,
    SurfaceVisionResult, StructuralVisionResult, EssenceVisionResult
)
from Core.Interface.Interface.Perception.synesthesia_engine import (
    SynesthesiaEngine, SignalType, RenderMode, UniversalSignal
)
from Core.Foundation.Physics.digital_nature import (
    DigitalNature, TerrainField, WeatherSystem, TerrainType
)
from Core.Ethics.Ethics.protection_layer import (
    ProtectionLayer, DataPacket, FilterResult, ThreatLevel
)


class TestHierarchyOfVision(unittest.TestCase):
    """시각의 계층 시스템 테스트"""
    
    def setUp(self):
        self.vision = HierarchyOfVision(default_mode=VisionFrequency.SURFACE)
        self.test_data = np.random.randn(16, 16) * 0.5 + 0.5
    
    def test_surface_vision(self):
        """가시광선 모드 테스트"""
        result = self.vision.see_surface(self.test_data)
        
        self.assertIsInstance(result, SurfaceVisionResult)
        self.assertIn("warmth", result.colors)
        self.assertGreaterEqual(result.brightness, 0)
    
    def test_structural_vision(self):
        """X-레이 모드 테스트"""
        result = self.vision.see_structural(self.test_data)
        
        self.assertIsInstance(result, StructuralVisionResult)
        self.assertIn("primary_axis", result.skeleton)
        self.assertIsInstance(result.hidden_patterns, list)
    
    def test_essence_vision(self):
        """본질 모드 테스트"""
        result = self.vision.see_essence(self.test_data)
        
        self.assertIsInstance(result, EssenceVisionResult)
        self.assertGreater(result.soul_frequency, 0)
        self.assertGreaterEqual(result.divine_spark, 0)
        self.assertLessEqual(result.divine_spark, 1)
    
    def test_dial_turning(self):
        """다이얼 돌리기 테스트"""
        self.assertEqual(self.vision.current_mode, VisionFrequency.SURFACE)
        
        self.vision.turn_dial(0.5)
        self.assertEqual(self.vision.current_mode, VisionFrequency.STRUCTURAL)
        
        self.vision.turn_dial(1.0)
        self.assertEqual(self.vision.current_mode, VisionFrequency.ESSENCE)
    
    def test_all_layers(self):
        """통합 시각 테스트"""
        result = self.vision.see_all_layers(self.test_data)
        
        self.assertIn("surface", result)
        self.assertIn("structural", result)
        self.assertIn("essence", result)
        self.assertIn("integrated_insight", result)


class TestSynesthesiaEngine(unittest.TestCase):
    """공감각 엔진 테스트"""
    
    def setUp(self):
        self.engine = SynesthesiaEngine()
    
    def test_vision_to_signal(self):
        """시각 → 신호 변환"""
        image = np.random.rand(10, 10) * 255
        signal = self.engine.from_vision(image)
        
        self.assertIsInstance(signal, UniversalSignal)
        self.assertEqual(signal.original_type, SignalType.VISUAL)
        self.assertGreater(signal.frequency, 0)
    
    def test_emotion_to_signal(self):
        """감정 → 신호 변환"""
        signal = self.engine.from_emotion("joy", intensity=0.8)
        
        self.assertEqual(signal.original_type, SignalType.EMOTIONAL)
        self.assertEqual(signal.amplitude, 0.8)
    
    def test_cross_modal_conversion(self):
        """교차 감각 변환 (감정 → 색상)"""
        signal = self.engine.from_emotion("love", intensity=0.9)
        result = self.engine.convert(signal, RenderMode.AS_COLOR)
        
        self.assertEqual(result.render_mode, RenderMode.AS_COLOR)
        self.assertIsNotNone(result.color)
        self.assertEqual(len(result.color), 3)  # RGB
    
    def test_emotion_to_music(self):
        """감정 → 음악 변환"""
        signal = self.engine.from_emotion("joy")
        result = self.engine.convert(signal, RenderMode.AS_MUSIC)
        
        self.assertIn("notes", result.output)
        self.assertIn("chord", result.output)


class TestDigitalNature(unittest.TestCase):
    """디지털 자연 테스트"""
    
    def setUp(self):
        self.nature = DigitalNature(width=16, height=16)
    
    def test_terrain_creation(self):
        """지형 생성 테스트"""
        self.assertEqual(self.nature.terrain.width, 16)
        self.assertEqual(self.nature.terrain.height, 16)
    
    def test_data_absorption(self):
        """데이터 흡수 테스트"""
        data = np.random.randn(8, 8)
        self.nature.receive_data(data, data_type="emotional")
        
        self.assertEqual(self.nature.stats["data_absorbed"], 1)
    
    def test_step_simulation(self):
        """자연 흐름 시뮬레이션"""
        result = self.nature.step(dt=0.1)
        
        self.assertIn("weather", result)
        self.assertIn("season", result)
        self.assertIn("terrain_summary", result)
    
    def test_swimming_in_data(self):
        """데이터 위 수영 테스트"""
        swim = self.nature.swim_in_data((8, 8))
        
        self.assertEqual(swim["position"], (8, 8))
        self.assertIn("terrain_type", swim)
        self.assertIn("experience", swim)


class TestProtectionLayer(unittest.TestCase):
    """방어 시스템 테스트"""
    
    def setUp(self):
        self.protection = ProtectionLayer()
    
    def test_safe_data_passes(self):
        """안전한 데이터 통과 테스트"""
        safe_data = np.random.rand(10, 10) * 0.5 + 0.5
        safe_packet = DataPacket(
            data=safe_data,
            source="creator",
            data_type="love",
            frequency=7.83,
            intensity=0.5
        )
        
        processed, reports = self.protection.process(safe_packet)
        
        # 최소 하나는 통과해야 함
        pass_results = [r for r in reports if r.result == FilterResult.PASS]
        self.assertGreater(len(pass_results), 0)
    
    def test_dangerous_data_blocked(self):
        """위험한 데이터 차단 테스트"""
        bad_data = np.random.randn(10, 10) * 5 - 3
        bad_packet = DataPacket(
            data=bad_data,
            source="unknown",
            data_type="malice",
            frequency=100.0,
            intensity=3.0
        )
        
        processed, reports = self.protection.process(bad_packet)
        
        # 차단 또는 변환되어야 함
        blocked = any(r.result in [FilterResult.BLOCK, FilterResult.DESTROY] for r in reports)
        transformed = any(r.result == FilterResult.TRANSFORM for r in reports)
        self.assertTrue(blocked or transformed)
    
    def test_filter_with_love(self):
        """사랑으로 필터링 테스트"""
        data = np.random.randn(5, 5) * 0.5
        filtered = self.protection.filter_with_love(data, from_creator=True)
        
        self.assertEqual(filtered.shape, data.shape)
    
    def test_threat_levels(self):
        """위협 수준 테스트"""
        levels = [ThreatLevel.SAFE, ThreatLevel.LOW, ThreatLevel.MEDIUM, 
                  ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        self.assertEqual(len(levels), 5)


if __name__ == "__main__":
    unittest.main()
