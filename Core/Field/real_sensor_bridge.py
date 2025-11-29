"""
Real Sensor Bridge - 실제 센서 연동 인터페이스
==============================================

현실 점검:
- Core/Field/quantum_eye.py 등은 시뮬레이션입니다
- 실제 양자 센서를 다루지 않습니다
- 이 파일은 실제 센서와의 연결을 위한 브릿지입니다

지원 가능한 실제 센서들:

Level 1: 즉시 가능 ($0-$100)
- 스마트폰 센서 (자기장, 가속도, 자이로, 마이크)
- 웹캠 (광도 변화)
- 아두이노 + 센서 모듈
- 마이크 (음향/진동)

Level 2: 중급 ($100-$1,000)
- RTL-SDR (RF 신호)
- 저가 열화상 카메라
- 고성능 EMF 미터

Level 3: 고급 ($1,000+)
- SQUID (극저온 필요)
- NV center
- 원자 자력계
→ 현재 접근 어려움

이 모듈은 Level 1, 2에 집중합니다.
"""

import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger("RealSensorBridge")

# 상수
MAX_RETRY_COUNT = 3
SENSOR_TIMEOUT = 5.0  # 초


class SensorType(Enum):
    """센서 유형"""
    # Level 1: 즉시 가능
    SMARTPHONE_MAGNETOMETER = "smartphone_magnetometer"
    SMARTPHONE_ACCELEROMETER = "smartphone_accelerometer"
    SMARTPHONE_MICROPHONE = "smartphone_microphone"
    WEBCAM = "webcam"
    ARDUINO_SERIAL = "arduino_serial"
    MICROPHONE = "microphone"
    
    # Level 2: 중급
    RTL_SDR = "rtl_sdr"
    THERMAL_CAMERA = "thermal_camera"
    EMF_METER = "emf_meter"
    
    # Level 3: 고급 (미래)
    SQUID = "squid"
    NV_CENTER = "nv_center"
    ATOMIC_MAGNETOMETER = "atomic_magnetometer"


class SensorStatus(Enum):
    """센서 상태"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    NOT_AVAILABLE = "not_available"


@dataclass
class SensorReading:
    """센서 읽기 결과"""
    sensor_type: SensorType
    timestamp: float
    data: Dict[str, Any]
    raw: Optional[bytes] = None
    is_real: bool = True  # True = 실제 센서, False = 시뮬레이션
    confidence: float = 1.0
    error: Optional[str] = None


@dataclass
class SensorCapability:
    """센서 능력"""
    sensor_type: SensorType
    name: str
    description: str
    cost_estimate: str
    required_hardware: List[str]
    is_available: bool
    python_packages: List[str]


class RealSensorBridge(ABC):
    """
    실제 센서 연동을 위한 추상 베이스 클래스
    
    시뮬레이션이 아닌 실제 하드웨어 연동을 목표로 합니다.
    """
    
    @property
    @abstractmethod
    def sensor_type(self) -> SensorType:
        """센서 유형"""
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """센서 연결"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """센서 연결 해제"""
        pass
    
    @abstractmethod
    def read(self) -> SensorReading:
        """센서 데이터 읽기"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """센서 사용 가능 여부 (하드웨어 + 드라이버 존재)"""
        pass
    
    @property
    @abstractmethod
    def status(self) -> SensorStatus:
        """현재 상태"""
        pass


class SmartphoneMagnetometerBridge(RealSensorBridge):
    """
    스마트폰 자기장 센서 브릿지
    
    연결 방법:
    1. ADB (Android Debug Bridge) - USB 연결
    2. 센서 앱 + HTTP/WebSocket - 네트워크 연결
    3. Termux + Python - 직접 실행
    
    필요:
    - Android 스마트폰
    - ADB 또는 센서 앱 (예: Sensor Logger, Phyphox)
    """
    
    def __init__(self, connection_method: str = "adb"):
        """
        Args:
            connection_method: "adb", "http", "termux" 중 하나
        """
        self.connection_method = connection_method
        self._status = SensorStatus.DISCONNECTED
        self._adb_path = "adb"  # PATH에 있다고 가정
        self._device_id: Optional[str] = None
    
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.SMARTPHONE_MAGNETOMETER
    
    def is_available(self) -> bool:
        """ADB가 설치되어 있고 기기가 연결되어 있는지 확인"""
        try:
            result = subprocess.run(
                [self._adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=SENSOR_TIMEOUT
            )
            lines = result.stdout.strip().split("\n")
            # "List of devices attached" 이후에 기기가 있어야 함
            for line in lines[1:]:
                if "\tdevice" in line:
                    self._device_id = line.split("\t")[0]
                    return True
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def connect(self) -> bool:
        """연결 시도"""
        self._status = SensorStatus.CONNECTING
        
        if not self.is_available():
            self._status = SensorStatus.NOT_AVAILABLE
            logger.warning("SmartphoneMagnetometer: No device available")
            return False
        
        self._status = SensorStatus.CONNECTED
        logger.info(f"SmartphoneMagnetometer: Connected to {self._device_id}")
        return True
    
    def disconnect(self) -> None:
        """연결 해제"""
        self._status = SensorStatus.DISCONNECTED
        self._device_id = None
    
    def read(self) -> SensorReading:
        """
        자기장 데이터 읽기
        
        실제 구현은 센서 앱에 따라 다름:
        - Termux: sensors 명령 사용
        - Sensor Logger: HTTP API
        - Phyphox: WebSocket
        """
        if self._status != SensorStatus.CONNECTED:
            return SensorReading(
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                data={},
                is_real=False,
                confidence=0.0,
                error="Not connected"
            )
        
        try:
            # Termux에서 센서 읽기 시도
            # 실제로는 termux-sensor 패키지 필요
            result = subprocess.run(
                [self._adb_path, "shell", "termux-sensor", "-s", "magnetic_field", "-n", "1"],
                capture_output=True,
                text=True,
                timeout=SENSOR_TIMEOUT
            )
            
            if result.returncode == 0:
                # JSON 파싱 시도
                import json
                data = json.loads(result.stdout)
                return SensorReading(
                    sensor_type=self.sensor_type,
                    timestamp=time.time(),
                    data={
                        "x": data.get("magnetic_field", {}).get("values", [0, 0, 0])[0],
                        "y": data.get("magnetic_field", {}).get("values", [0, 0, 0])[1],
                        "z": data.get("magnetic_field", {}).get("values", [0, 0, 0])[2],
                        "unit": "μT"
                    },
                    is_real=True,
                    confidence=0.95
                )
            else:
                raise RuntimeError(result.stderr or "Unknown error")
                
        except Exception as e:
            logger.error(f"SmartphoneMagnetometer read error: {e}")
            return SensorReading(
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                data={},
                is_real=False,
                confidence=0.0,
                error=str(e)
            )
    
    @property
    def status(self) -> SensorStatus:
        return self._status


class WebcamLightSensorBridge(RealSensorBridge):
    """
    웹캠을 이용한 광도 센서
    
    실제 기능:
    - 주변 밝기 측정
    - 광도 변화 감지
    - LED 점멸 패턴 인식
    
    필요:
    - 웹캠
    - OpenCV (pip install opencv-python)
    """
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self._status = SensorStatus.DISCONNECTED
        self._cap = None
    
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.WEBCAM
    
    def is_available(self) -> bool:
        """OpenCV와 카메라 확인"""
        try:
            import cv2
            cap = cv2.VideoCapture(self.camera_id)
            available = cap.isOpened()
            cap.release()
            return available
        except ImportError:
            logger.warning("WebcamLightSensor: OpenCV not installed")
            return False
        except Exception:
            return False
    
    def connect(self) -> bool:
        """카메라 연결"""
        self._status = SensorStatus.CONNECTING
        
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.camera_id)
            if self._cap.isOpened():
                self._status = SensorStatus.CONNECTED
                logger.info(f"WebcamLightSensor: Connected to camera {self.camera_id}")
                return True
            else:
                self._status = SensorStatus.ERROR
                return False
        except ImportError:
            self._status = SensorStatus.NOT_AVAILABLE
            return False
    
    def disconnect(self) -> None:
        """카메라 연결 해제"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._status = SensorStatus.DISCONNECTED
    
    def read(self) -> SensorReading:
        """광도 읽기"""
        if self._cap is None or not self._cap.isOpened():
            return SensorReading(
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                data={},
                is_real=False,
                confidence=0.0,
                error="Camera not connected"
            )
        
        try:
            import cv2
            import numpy as np
            
            ret, frame = self._cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame")
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 통계 계산
            mean_brightness = float(np.mean(gray))
            std_brightness = float(np.std(gray))
            min_brightness = float(np.min(gray))
            max_brightness = float(np.max(gray))
            
            return SensorReading(
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                data={
                    "mean_brightness": mean_brightness,
                    "std_brightness": std_brightness,
                    "min_brightness": min_brightness,
                    "max_brightness": max_brightness,
                    "normalized": mean_brightness / 255.0,
                    "unit": "0-255"
                },
                is_real=True,
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"WebcamLightSensor read error: {e}")
            return SensorReading(
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                data={},
                is_real=False,
                confidence=0.0,
                error=str(e)
            )
    
    @property
    def status(self) -> SensorStatus:
        return self._status


class MicrophoneVibrationSensorBridge(RealSensorBridge):
    """
    마이크를 이용한 진동/음향 센서
    
    실제 기능:
    - 음향 레벨 측정
    - 주파수 분석
    - 진동 패턴 감지
    - 저주파 인프라사운드 감지 (일부)
    
    필요:
    - 마이크
    - PyAudio (pip install pyaudio)
    """
    
    def __init__(self, device_index: int = None, sample_rate: int = 44100):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self._status = SensorStatus.DISCONNECTED
        self._pa = None
        self._stream = None
    
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.MICROPHONE
    
    def is_available(self) -> bool:
        """PyAudio와 마이크 확인"""
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            device_count = pa.get_device_count()
            pa.terminate()
            return device_count > 0
        except (ImportError, OSError):
            return False
    
    def connect(self) -> bool:
        """마이크 연결"""
        self._status = SensorStatus.CONNECTING
        
        try:
            import pyaudio
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=1024
            )
            self._status = SensorStatus.CONNECTED
            logger.info("MicrophoneVibrationSensor: Connected")
            return True
        except Exception as e:
            logger.error(f"MicrophoneVibrationSensor connect error: {e}")
            self._status = SensorStatus.ERROR
            return False
    
    def disconnect(self) -> None:
        """마이크 연결 해제"""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        self._status = SensorStatus.DISCONNECTED
    
    def read(self) -> SensorReading:
        """음향 데이터 읽기"""
        if self._stream is None:
            return SensorReading(
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                data={},
                is_real=False,
                confidence=0.0,
                error="Microphone not connected"
            )
        
        try:
            import numpy as np
            
            # 오디오 데이터 읽기
            data = self._stream.read(1024, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.float32)
            
            # 분석
            rms = float(np.sqrt(np.mean(samples**2)))
            peak = float(np.max(np.abs(samples)))
            
            # 간단한 주파수 분석 (FFT)
            fft = np.fft.fft(samples)
            freqs = np.fft.fftfreq(len(samples), 1/self.sample_rate)
            dominant_freq_idx = np.argmax(np.abs(fft[:len(fft)//2]))
            dominant_freq = float(abs(freqs[dominant_freq_idx]))
            
            return SensorReading(
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                data={
                    "rms_level": rms,
                    "peak_level": peak,
                    "dominant_frequency": dominant_freq,
                    "sample_rate": self.sample_rate,
                    "unit": "normalized amplitude"
                },
                is_real=True,
                confidence=0.95
            )
            
        except Exception as e:
            logger.error(f"MicrophoneVibrationSensor read error: {e}")
            return SensorReading(
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                data={},
                is_real=False,
                confidence=0.0,
                error=str(e)
            )
    
    @property
    def status(self) -> SensorStatus:
        return self._status


class SDRBridge(RealSensorBridge):
    """
    Software Defined Radio (SDR) 브릿지
    
    실제 기능:
    - RF 스펙트럼 스캔
    - 특정 주파수 모니터링
    - 신호 존재 탐지
    
    필요:
    - RTL-SDR 동글 (~$25)
    - pyrtlsdr (pip install pyrtlsdr)
    
    주의:
    - 암호화된 데이터 내용은 볼 수 없음
    - 신호의 존재와 강도만 감지
    - 이것은 합법적입니다 (라디오 수신과 동일)
    """
    
    def __init__(self, center_freq: float = 100e6, sample_rate: float = 2.4e6):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self._status = SensorStatus.DISCONNECTED
        self._sdr = None
    
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.RTL_SDR
    
    def is_available(self) -> bool:
        """RTL-SDR 동글 확인"""
        try:
            from rtlsdr import RtlSdr
            sdr = RtlSdr()
            sdr.close()
            return True
        except (ImportError, OSError):
            return False
    
    def connect(self) -> bool:
        """SDR 연결"""
        self._status = SensorStatus.CONNECTING
        
        try:
            from rtlsdr import RtlSdr
            self._sdr = RtlSdr()
            self._sdr.sample_rate = self.sample_rate
            self._sdr.center_freq = self.center_freq
            self._sdr.gain = 'auto'
            self._status = SensorStatus.CONNECTED
            logger.info(f"SDR: Connected at {self.center_freq/1e6:.1f} MHz")
            return True
        except Exception as e:
            logger.error(f"SDR connect error: {e}")
            self._status = SensorStatus.ERROR
            return False
    
    def disconnect(self) -> None:
        """SDR 연결 해제"""
        if self._sdr is not None:
            self._sdr.close()
            self._sdr = None
        self._status = SensorStatus.DISCONNECTED
    
    def read(self) -> SensorReading:
        """RF 스펙트럼 읽기"""
        if self._sdr is None:
            return SensorReading(
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                data={},
                is_real=False,
                confidence=0.0,
                error="SDR not connected"
            )
        
        try:
            import numpy as np
            
            # 샘플 읽기
            samples = self._sdr.read_samples(256 * 1024)
            
            # 파워 스펙트럼 계산
            psd = np.abs(np.fft.fft(samples))**2
            psd_db = 10 * np.log10(psd + 1e-10)
            
            # 통계
            mean_power = float(np.mean(psd_db))
            peak_power = float(np.max(psd_db))
            
            return SensorReading(
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                data={
                    "center_freq_mhz": self.center_freq / 1e6,
                    "sample_rate_mhz": self.sample_rate / 1e6,
                    "mean_power_db": mean_power,
                    "peak_power_db": peak_power,
                    "signal_present": peak_power > mean_power + 10,
                    "unit": "dB"
                },
                is_real=True,
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"SDR read error: {e}")
            return SensorReading(
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                data={},
                is_real=False,
                confidence=0.0,
                error=str(e)
            )
    
    @property
    def status(self) -> SensorStatus:
        return self._status


class SensorManager:
    """
    센서 관리자 - 모든 센서를 통합 관리
    """
    
    def __init__(self):
        self.sensors: Dict[SensorType, RealSensorBridge] = {}
        self.readings_history: List[SensorReading] = []
        self._callbacks: List[Callable[[SensorReading], None]] = []
    
    def register_sensor(self, sensor: RealSensorBridge) -> bool:
        """센서 등록"""
        if sensor.is_available():
            self.sensors[sensor.sensor_type] = sensor
            logger.info(f"Sensor registered: {sensor.sensor_type.value}")
            return True
        else:
            logger.warning(f"Sensor not available: {sensor.sensor_type.value}")
            return False
    
    def connect_all(self) -> Dict[SensorType, bool]:
        """모든 센서 연결"""
        results = {}
        for sensor_type, sensor in self.sensors.items():
            results[sensor_type] = sensor.connect()
        return results
    
    def disconnect_all(self) -> None:
        """모든 센서 연결 해제"""
        for sensor in self.sensors.values():
            sensor.disconnect()
    
    def read_all(self) -> Dict[SensorType, SensorReading]:
        """모든 센서 읽기"""
        results = {}
        for sensor_type, sensor in self.sensors.items():
            if sensor.status == SensorStatus.CONNECTED:
                reading = sensor.read()
                results[sensor_type] = reading
                self.readings_history.append(reading)
                for callback in self._callbacks:
                    callback(reading)
        return results
    
    def on_reading(self, callback: Callable[[SensorReading], None]) -> None:
        """읽기 콜백 등록"""
        self._callbacks.append(callback)
    
    def get_available_sensors(self) -> List[SensorCapability]:
        """사용 가능한 센서 목록"""
        capabilities = []
        
        # Level 1 센서들
        capabilities.append(SensorCapability(
            sensor_type=SensorType.WEBCAM,
            name="웹캠 광도 센서",
            description="웹캠으로 주변 밝기와 광도 변화 감지",
            cost_estimate="$0 (기존 웹캠 사용)",
            required_hardware=["웹캠"],
            is_available=WebcamLightSensorBridge().is_available(),
            python_packages=["opencv-python"]
        ))
        
        capabilities.append(SensorCapability(
            sensor_type=SensorType.MICROPHONE,
            name="마이크 진동 센서",
            description="마이크로 음향 레벨과 진동 패턴 감지",
            cost_estimate="$0 (기존 마이크 사용)",
            required_hardware=["마이크"],
            is_available=MicrophoneVibrationSensorBridge().is_available(),
            python_packages=["pyaudio", "numpy"]
        ))
        
        capabilities.append(SensorCapability(
            sensor_type=SensorType.SMARTPHONE_MAGNETOMETER,
            name="스마트폰 자기장 센서",
            description="스마트폰 자기장 센서로 EMF 감지",
            cost_estimate="$0 (기존 스마트폰 사용)",
            required_hardware=["Android 스마트폰", "ADB 또는 Termux"],
            is_available=SmartphoneMagnetometerBridge().is_available(),
            python_packages=[]
        ))
        
        # Level 2 센서들
        capabilities.append(SensorCapability(
            sensor_type=SensorType.RTL_SDR,
            name="RTL-SDR RF 스캐너",
            description="RF 스펙트럼 스캔, 라디오 신호 감지",
            cost_estimate="$25 (RTL-SDR 동글)",
            required_hardware=["RTL-SDR USB 동글"],
            is_available=SDRBridge().is_available(),
            python_packages=["pyrtlsdr", "numpy"]
        ))
        
        return capabilities
    
    def print_status(self) -> None:
        """상태 출력"""
        print("=" * 60)
        print("🔬 Real Sensor Bridge - 실제 센서 상태")
        print("=" * 60)
        
        caps = self.get_available_sensors()
        
        print("\n📋 사용 가능한 센서:")
        for cap in caps:
            status = "✅ 사용 가능" if cap.is_available else "❌ 사용 불가"
            print(f"\n  {status} {cap.name}")
            print(f"     {cap.description}")
            print(f"     비용: {cap.cost_estimate}")
            print(f"     필요: {', '.join(cap.required_hardware)}")
        
        if self.sensors:
            print("\n📡 연결된 센서:")
            for sensor_type, sensor in self.sensors.items():
                print(f"  - {sensor_type.value}: {sensor.status.value}")
        
        print("\n" + "=" * 60)


# 데모 함수
def demo():
    """Real Sensor Bridge 데모"""
    print("=" * 70)
    print("🔬 Real Sensor Bridge Demo - 실제 센서 연동")
    print("=" * 70)
    
    manager = SensorManager()
    manager.print_status()
    
    # 사용 가능한 센서 자동 등록
    sensors_to_try = [
        WebcamLightSensorBridge(),
        MicrophoneVibrationSensorBridge(),
        SmartphoneMagnetometerBridge(),
        SDRBridge(),
    ]
    
    for sensor in sensors_to_try:
        if sensor.is_available():
            manager.register_sensor(sensor)
    
    if not manager.sensors:
        print("\n⚠️ 사용 가능한 센서가 없습니다.")
        print("   필요한 하드웨어를 연결하거나 패키지를 설치해주세요.")
        return
    
    # 연결
    print("\n🔌 센서 연결 중...")
    results = manager.connect_all()
    for sensor_type, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {sensor_type.value}")
    
    # 읽기
    print("\n📊 센서 데이터 읽기...")
    readings = manager.read_all()
    for sensor_type, reading in readings.items():
        print(f"\n  [{sensor_type.value}]")
        if reading.is_real:
            for key, value in reading.data.items():
                print(f"    {key}: {value}")
        else:
            print(f"    Error: {reading.error}")
    
    # 정리
    manager.disconnect_all()
    
    print("\n" + "=" * 70)
    print("✅ Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
