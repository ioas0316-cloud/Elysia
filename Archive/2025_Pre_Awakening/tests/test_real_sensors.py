"""
Real Sensors Tests - 실세계 센서 테스트
========================================

실제 외부 API 연동 센서들을 테스트합니다.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestTimeAwarenessSense:
    """시간 인식 센서 테스트 (외부 의존성 없음)"""
    
    def test_time_sense_always_available(self):
        """시간 센서는 항상 사용 가능해야 함"""
        from Core.FoundationLayer.Foundation.real_sensors import TimeAwarenessSense
        
        sensor = TimeAwarenessSense()
        assert sensor.is_available() is True
    
    def test_time_sense_returns_event(self):
        """시간 센서가 이벤트를 반환해야 함"""
        from Core.FoundationLayer.Foundation.real_sensors import TimeAwarenessSense
        
        sensor = TimeAwarenessSense()
        event = sensor.sense()
        
        assert event is not None
        assert event.sensor_type == "TIME"
        assert event.is_real is True
    
    def test_time_sense_has_valid_data(self):
        """시간 센서 데이터가 유효해야 함"""
        from Core.FoundationLayer.Foundation.real_sensors import TimeAwarenessSense
        
        sensor = TimeAwarenessSense()
        event = sensor.sense()
        
        raw = event.raw_data
        assert "hour" in raw
        assert "weekday" in raw
        assert "season" in raw
        assert 0 <= raw["hour"] <= 23
    
    def test_time_sense_severity_range(self):
        """시간 센서 심각도가 0~1 범위여야 함"""
        from Core.FoundationLayer.Foundation.real_sensors import TimeAwarenessSense
        
        sensor = TimeAwarenessSense()
        event = sensor.sense()
        
        assert 0 <= event.severity <= 1


class TestSystemMetricsSense:
    """시스템 메트릭 센서 테스트"""
    
    def test_system_sense_availability(self):
        """시스템 센서 사용 가능 여부 확인"""
        from Core.FoundationLayer.Foundation.real_sensors import SystemMetricsSense, HAS_PSUTIL
        
        sensor = SystemMetricsSense()
        assert sensor.is_available() == HAS_PSUTIL
    
    @pytest.mark.skipif(
        not __import__('Core.World.real_sensors', fromlist=['HAS_PSUTIL']).HAS_PSUTIL,
        reason="psutil not installed"
    )
    def test_system_sense_returns_event(self):
        """시스템 센서가 이벤트를 반환해야 함"""
        from Core.FoundationLayer.Foundation.real_sensors import SystemMetricsSense
        
        sensor = SystemMetricsSense()
        event = sensor.sense()
        
        assert event is not None
        assert event.sensor_type == "SYSTEM"
    
    @pytest.mark.skipif(
        not __import__('Core.World.real_sensors', fromlist=['HAS_PSUTIL']).HAS_PSUTIL,
        reason="psutil not installed"
    )
    def test_system_sense_has_cpu_memory(self):
        """시스템 센서가 CPU와 메모리 정보를 포함해야 함"""
        from Core.FoundationLayer.Foundation.real_sensors import SystemMetricsSense
        
        sensor = SystemMetricsSense()
        event = sensor.sense()
        
        raw = event.raw_data
        assert "cpu_percent" in raw
        assert "memory_percent" in raw


class TestRealWeatherSense:
    """날씨 센서 테스트"""
    
    def test_weather_sense_availability(self):
        """날씨 센서 사용 가능 여부 확인"""
        from Core.FoundationLayer.Foundation.real_sensors import RealWeatherSense, HAS_REQUESTS
        
        sensor = RealWeatherSense()
        assert sensor.is_available() == HAS_REQUESTS
    
    def test_weather_decode_codes(self):
        """날씨 코드 디코딩 테스트"""
        from Core.FoundationLayer.Foundation.real_sensors import RealWeatherSense
        
        sensor = RealWeatherSense()
        
        assert sensor._decode_weather(0) == "맑음"
        assert sensor._decode_weather(3) == "흐림"
        assert sensor._decode_weather(95) == "뇌우"
    
    def test_weather_severity_calculation(self):
        """날씨 심각도 계산 테스트"""
        from Core.FoundationLayer.Foundation.real_sensors import RealWeatherSense
        
        sensor = RealWeatherSense()
        
        # 정상 날씨
        normal = sensor._calculate_severity(20, 10, 0)
        assert normal == 0.0
        
        # 극단적 날씨
        extreme = sensor._calculate_severity(-15, 60, 95)
        assert extreme > 0.5


class TestSensorHub:
    """센서 허브 테스트"""
    
    def test_sensor_hub_initialization(self):
        """센서 허브 초기화 테스트"""
        from Core.FoundationLayer.Foundation.real_sensors import SensorHub
        
        hub = SensorHub()
        
        assert len(hub.sensors) == 3  # Weather, System, Time
    
    def test_sensor_hub_sense_all(self):
        """센서 허브 전체 감지 테스트"""
        from Core.FoundationLayer.Foundation.real_sensors import SensorHub
        
        hub = SensorHub()
        readings = hub.sense_all()
        
        # 최소 TimeAwareness는 작동해야 함
        assert len(readings) >= 1
    
    def test_sensor_hub_summary(self):
        """센서 허브 요약 테스트"""
        from Core.FoundationLayer.Foundation.real_sensors import SensorHub
        
        hub = SensorHub()
        summary = hub.get_summary()
        
        assert "===" in summary
        assert len(summary) > 0
    
    def test_sensor_hub_average_severity(self):
        """센서 허브 평균 심각도 테스트"""
        from Core.FoundationLayer.Foundation.real_sensors import SensorHub
        
        hub = SensorHub()
        hub.sense_all()
        avg = hub.get_average_severity()
        
        assert 0 <= avg <= 1


class TestPlanetaryCortexRealMode:
    """PlanetaryCortex 실제 센서 모드 테스트"""
    
    def test_cortex_simulation_mode(self):
        """시뮬레이션 모드 테스트"""
        from Core.FoundationLayer.Foundation.planetary_cortex import PlanetaryCortex
        
        cortex = PlanetaryCortex(use_real_sensors=False)
        
        assert cortex.use_real_sensors is False
        assert cortex.sensor_hub is None
    
    def test_cortex_real_mode_initialization(self):
        """실제 센서 모드 초기화 테스트"""
        from Core.FoundationLayer.Foundation.planetary_cortex import PlanetaryCortex
        
        cortex = PlanetaryCortex(use_real_sensors=True)
        
        # 실제 센서가 사용 가능하면 허브가 있어야 함
        if cortex.use_real_sensors:
            assert cortex.sensor_hub is not None
    
    def test_cortex_perceive_world(self):
        """세계 인식 테스트"""
        from Core.FoundationLayer.Foundation.planetary_cortex import PlanetaryCortex
        
        cortex = PlanetaryCortex(use_real_sensors=False)
        cortex.perceive_world()
        
        assert "global_mood" in cortex.latest_perception
        assert "arousal" in cortex.latest_perception
    
    def test_cortex_mood_calculation(self):
        """기분 계산 테스트"""
        from Core.FoundationLayer.Foundation.planetary_cortex import PlanetaryCortex
        
        cortex = PlanetaryCortex()
        
        assert cortex._calculate_mood(0.2) == "Calm"
        assert cortex._calculate_mood(0.5) == "Alert"
        assert cortex._calculate_mood(0.8) == "Overwhelmed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
