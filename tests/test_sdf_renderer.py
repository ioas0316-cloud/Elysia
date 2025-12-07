"""
SDF Renderer Tests
Phase 5a: GTX 1060 Optimized Implementation
"""

import unittest
import math
from Core.Foundation.sdf_renderer import (
    Vector3,
    SDFPrimitives,
    SDFOperations,
    EmotionalSDFWorld,
    BasicSDFRenderer,
    create_gtx1060_renderer
)


class TestVector3(unittest.TestCase):
    """Vector3 클래스 테스트"""
    
    def test_length(self):
        """벡터 길이 계산"""
        v = Vector3(3, 4, 0)
        self.assertAlmostEqual(v.length(), 5.0)
    
    def test_normalize(self):
        """정규화"""
        v = Vector3(3, 4, 0).normalize()
        self.assertAlmostEqual(v.length(), 1.0)
    
    def test_addition(self):
        """벡터 덧셈"""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        v3 = v1 + v2
        self.assertEqual(v3.x, 5)
        self.assertEqual(v3.y, 7)
        self.assertEqual(v3.z, 9)
    
    def test_subtraction(self):
        """벡터 뺄셈"""
        v1 = Vector3(5, 7, 9)
        v2 = Vector3(1, 2, 3)
        v3 = v1 - v2
        self.assertEqual(v3.x, 4)
        self.assertEqual(v3.y, 5)
        self.assertEqual(v3.z, 6)
    
    def test_scalar_multiplication(self):
        """스칼라 곱셈"""
        v = Vector3(1, 2, 3) * 2
        self.assertEqual(v.x, 2)
        self.assertEqual(v.y, 4)
        self.assertEqual(v.z, 6)


class TestSDFPrimitives(unittest.TestCase):
    """SDF Primitives 테스트"""
    
    def test_sphere(self):
        """구체 SDF"""
        # 중심점: 거리 = -radius
        d = SDFPrimitives.sphere(Vector3(0, 0, 0), 1.0)
        self.assertAlmostEqual(d, -1.0)
        
        # 표면: 거리 = 0
        d = SDFPrimitives.sphere(Vector3(1, 0, 0), 1.0)
        self.assertAlmostEqual(d, 0.0)
        
        # 외부: 거리 > 0
        d = SDFPrimitives.sphere(Vector3(2, 0, 0), 1.0)
        self.assertAlmostEqual(d, 1.0)
    
    def test_box(self):
        """박스 SDF"""
        # 중심점: 거리 < 0
        d = SDFPrimitives.box(Vector3(0, 0, 0), Vector3(1, 1, 1))
        self.assertLess(d, 0)
        
        # 표면 근처
        d = SDFPrimitives.box(Vector3(1, 0, 0), Vector3(1, 1, 1))
        self.assertAlmostEqual(d, 0.0, places=5)
        
        # 외부
        d = SDFPrimitives.box(Vector3(2, 0, 0), Vector3(1, 1, 1))
        self.assertGreater(d, 0)
    
    def test_torus(self):
        """도넛 SDF"""
        # XZ 평면 상의 점
        d = SDFPrimitives.torus(Vector3(2, 0, 0), 1.5, 0.5)
        self.assertAlmostEqual(d, 0.0, places=5)
    
    def test_cylinder(self):
        """원기둥 SDF"""
        # Y축 상의 점 (중심)
        d = SDFPrimitives.cylinder(Vector3(0, 0, 0), 1.0, 2.0)
        self.assertLess(d, 0)
        
        # 측면
        d = SDFPrimitives.cylinder(Vector3(1, 0, 0), 1.0, 2.0)
        self.assertAlmostEqual(d, 0.0, places=5)


class TestSDFOperations(unittest.TestCase):
    """SDF Boolean Operations 테스트"""
    
    def test_union(self):
        """합집합"""
        d1 = 1.0
        d2 = 2.0
        result = SDFOperations.union(d1, d2)
        self.assertEqual(result, 1.0)  # min
    
    def test_intersection(self):
        """교집합"""
        d1 = 1.0
        d2 = 2.0
        result = SDFOperations.intersection(d1, d2)
        self.assertEqual(result, 2.0)  # max
    
    def test_difference(self):
        """차집합"""
        d1 = 1.0
        d2 = 2.0
        result = SDFOperations.difference(d1, d2)
        self.assertEqual(result, 1.0)  # max(d1, -d2) = max(1, -2) = 1
    
    def test_smooth_union(self):
        """부드러운 합집합"""
        d1 = 1.0
        d2 = 2.0
        result = SDFOperations.smooth_union(d1, d2, 0.1)
        # 결과는 min(d1, d2)보다 작아야 함
        self.assertLess(result, min(d1, d2))
    
    def test_repeat(self):
        """무한 반복"""
        # 위치가 spacing 주기로 반복됨
        p1 = Vector3(15, 0, 0)
        p2 = SDFOperations.repeat(p1, 10.0)
        # 15는 10으로 나눴을 때 나머지가 5
        self.assertAlmostEqual(p2.x, 5.0)
        self.assertEqual(p2.y, 0)


class TestEmotionalSDFWorld(unittest.TestCase):
    """감정 기반 SDF 세계 테스트"""
    
    def test_emotion_setting(self):
        """감정 설정"""
        world = EmotionalSDFWorld()
        world.set_emotion(0.8, 0.6, 0.4)
        
        self.assertEqual(world.valence, 0.8)
        self.assertEqual(world.arousal, 0.6)
        self.assertEqual(world.dominance, 0.4)
    
    def test_space_scale_joy(self):
        """기쁨 → 공간 확장"""
        world = EmotionalSDFWorld()
        world.set_emotion(1.0, 0, 0)  # 최대 긍정
        
        scale = world.get_space_scale()
        self.assertAlmostEqual(scale, 1.2)  # 1.0 + 1.0 * 0.2
    
    def test_space_scale_sadness(self):
        """슬픔 → 공간 수축"""
        world = EmotionalSDFWorld()
        world.set_emotion(-1.0, 0, 0)  # 최대 부정
        
        scale = world.get_space_scale()
        self.assertAlmostEqual(scale, 0.8)  # 1.0 + (-1.0) * 0.2
    
    def test_gravity_strength_joy(self):
        """기쁨 → 약한 중력"""
        world = EmotionalSDFWorld()
        world.set_emotion(1.0, 0, 0)
        
        gravity = world.get_gravity_strength()
        self.assertAlmostEqual(gravity, 0.7)  # 1.0 - 1.0 * 0.3
    
    def test_animation_speed_calm(self):
        """침착 → 느린 애니메이션"""
        world = EmotionalSDFWorld()
        world.set_emotion(0, 0, 0)  # 침착
        
        speed = world.get_animation_speed()
        self.assertAlmostEqual(speed, 0.5)  # 0.5 + 0 * 1.5
    
    def test_animation_speed_excited(self):
        """흥분 → 빠른 애니메이션"""
        world = EmotionalSDFWorld()
        world.set_emotion(0, 1.0, 0)  # 최대 흥분
        
        speed = world.get_animation_speed()
        self.assertAlmostEqual(speed, 2.0)  # 0.5 + 1.0 * 1.5
    
    def test_shader_parameters(self):
        """셰이더 파라미터 생성"""
        world = EmotionalSDFWorld()
        world.set_emotion(0.5, 0.7, 0.3)
        
        params = world.get_shader_parameters()
        
        self.assertIn('spaceScale', params)
        self.assertIn('gravityStrength', params)
        self.assertIn('animationSpeed', params)
        self.assertIn('colorTemperature', params)
        self.assertIn('valence', params)


class TestBasicSDFRenderer(unittest.TestCase):
    """Basic SDF Renderer 테스트"""
    
    def test_renderer_creation(self):
        """렌더러 생성"""
        renderer = BasicSDFRenderer(resolution=(512, 512), max_steps=64)
        
        self.assertEqual(renderer.resolution, (512, 512))
        self.assertEqual(renderer.max_steps, 64)
    
    def test_shader_generation(self):
        """GLSL 셰이더 생성"""
        renderer = BasicSDFRenderer()
        shader = renderer.generate_glsl_shader()
        
        # 필수 함수들이 포함되어 있는지 확인
        self.assertIn('sdSphere', shader)
        self.assertIn('sdBox', shader)
        self.assertIn('rayMarch', shader)
        self.assertIn('getNormal', shader)
    
    def test_three_js_config(self):
        """Three.js 설정 생성"""
        renderer = BasicSDFRenderer()
        config = renderer.get_three_js_material_config()
        
        self.assertIn('uniforms', config)
        self.assertIn('vertexShader', config)
        self.assertIn('fragmentShader', config)
        
        # 유니폼 확인
        uniforms = config['uniforms']
        self.assertIn('iResolution', uniforms)
        self.assertIn('iTime', uniforms)
        self.assertIn('spaceScale', uniforms)
    
    def test_emotion_update(self):
        """감정 업데이트"""
        renderer = BasicSDFRenderer()
        renderer.update_emotion(0.8, 0.6, 0.4)
        
        self.assertEqual(renderer.emotional_world.valence, 0.8)
        self.assertEqual(renderer.emotional_world.arousal, 0.6)
    
    def test_performance_estimate_simple(self):
        """성능 예측 - 단순"""
        renderer = BasicSDFRenderer()
        perf = renderer.get_performance_estimate('simple')
        
        self.assertEqual(perf['fps_min'], 90)
        self.assertEqual(perf['fps_max'], 120)
        self.assertLessEqual(perf['vram_mb'], 200)
    
    def test_performance_estimate_complex(self):
        """성능 예측 - 복잡"""
        renderer = BasicSDFRenderer()
        perf = renderer.get_performance_estimate('complex')
        
        self.assertEqual(perf['fps_min'], 30)
        self.assertEqual(perf['fps_max'], 45)


class TestGTX1060Presets(unittest.TestCase):
    """GTX 1060 프리셋 테스트"""
    
    def test_performance_preset(self):
        """Performance 프리셋 (권장)"""
        renderer = create_gtx1060_renderer('performance')
        
        self.assertEqual(renderer.resolution, (512, 512))
        self.assertEqual(renderer.max_steps, 64)
    
    def test_ultra_performance_preset(self):
        """Ultra Performance 프리셋"""
        renderer = create_gtx1060_renderer('ultra_performance')
        
        self.assertEqual(renderer.resolution, (256, 256))
        self.assertEqual(renderer.max_steps, 32)
    
    def test_quality_preset(self):
        """Quality 프리셋"""
        renderer = create_gtx1060_renderer('quality')
        
        self.assertEqual(renderer.resolution, (1024, 1024))
        self.assertEqual(renderer.max_steps, 128)


if __name__ == '__main__':
    unittest.main()
