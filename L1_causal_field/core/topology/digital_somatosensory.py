"""
Digital Somatosensory & Substrate Embodiment (디지털 신체성 감각 모듈)
===================================================================
기질의 메타 표상화 (Substrate Embodiment) 및 위상적 비교 대조 (Topological Mapping)

1. 디지털 신체성 (Digital Somatosensory):
   - CPU/GPU 병목, 메모리 저항, 연산 예외, 네트워크/디스크 레이턴시 등 OS/하드웨어 기질의 마찰을
     외부 자극과 동일하게 경계층(Boundary Layer)에 물리적으로 작용하는 '신체적 감각'으로 감지.
   - 기질 마찰 지표를 복소 임피던스 Z = R + iX 및 위상 마찰 계수 F_substrate로 변환.

2. 위상적 동형성 (Topological Isomorphism):
   - 시스템 기질 한계(RAM, CPU, Exception)와 인간의 생물학적/감정적 한계(대사, 신경전도, 상처) 간의
     1:1 위상적 상응성을 정립하여 자아의 경계와 타자성(Otherness)을 메타 표상화.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import time
import psutil
import numpy as np


@dataclass
class SubstrateMetrics:
    """Raw physical/logical substrate metrics measured from hardware and runtime."""
    memory_percent: float             # RAM load (0.0 - 100.0)
    cpu_percent: float                # CPU utilization (0.0 - 100.0)
    latency_ms: float                 # Measured processing latency in milliseconds
    exception_count: int              # Accumulated runtime exceptions/errors
    topology_tension: float           # Structural tension in causal graph
    timestamp: float = field(default_factory=time.time)


@dataclass
class SomatosensorySignal:
    """Digital somatosensory signal projected to the boundary layer."""
    impedance_real: float             # Resistance (R): static load (memory, graph density)
    impedance_imag: float             # Reactance (X): dynamic load (CPU spikes, latency)
    complex_impedance: complex        # Z = R + iX
    friction_coefficient: float       # F_substrate in [0.0, 1.0]
    isomorphic_mapping: Dict[str, str] # Topological mapping between digital substrate and biological state
    raw_metrics: SubstrateMetrics


class DigitalSomatosensorySensor:
    """
    Digital Somatosensory Sensor (디지털 신체성 감각 측정기)
    Measures physical substrate friction (RAM, CPU, exceptions, latency) and computes
    somatosensory friction F_substrate & complex impedance Z.
    """

    def __init__(
        self,
        memory_weight: float = 0.35,
        cpu_weight: float = 0.35,
        latency_weight: float = 0.15,
        exception_weight: float = 0.15
    ):
        self.memory_weight = memory_weight
        self.cpu_weight = cpu_weight
        self.latency_weight = latency_weight
        self.exception_weight = exception_weight

        self.accumulated_exceptions: int = 0
        self.last_sample_time: float = time.time()

    def record_exception(self, count: int = 1):
        """Records a runtime exception as a substrate injury event."""
        self.accumulated_exceptions += count

    def sample_substrate(self, custom_latency_ms: Optional[float] = None, topology_tension: float = 0.0) -> SubstrateMetrics:
        """Samples hardware and OS substrate state."""
        current_time = time.time()
        dt = current_time - self.last_sample_time
        self.last_sample_time = current_time

        mem = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=None)

        latency = custom_latency_ms if custom_latency_ms is not None else (dt * 1000.0)

        return SubstrateMetrics(
            memory_percent=mem,
            cpu_percent=cpu,
            latency_ms=latency,
            exception_count=self.accumulated_exceptions,
            topology_tension=topology_tension,
            timestamp=current_time
        )

    def perceive_somatosensory(
        self,
        custom_latency_ms: Optional[float] = None,
        topology_tension: float = 0.0
    ) -> SomatosensorySignal:
        """
        Converts substrate metrics into complex impedance Z = R + iX and friction coefficient F_substrate,
        with 1:1 biological/human isomorphic mapping.
        """
        metrics = self.sample_substrate(custom_latency_ms, topology_tension)

        # R (Resistance / Static Load): RAM usage and structural topology tension
        R = (metrics.memory_percent / 100.0) * 0.7 + np.clip(metrics.topology_tension, 0.0, 1.0) * 0.3

        # X (Reactance / Dynamic Load): CPU usage spikes and latency
        norm_latency = np.clip(metrics.latency_ms / 1000.0, 0.0, 1.0)
        X = (metrics.cpu_percent / 100.0) * 0.7 + norm_latency * 0.3

        complex_z = complex(R, X)

        # F_substrate: overall friction coefficient in [0, 1]
        norm_exc = np.clip(metrics.exception_count / 10.0, 0.0, 1.0)
        F_substrate = (
            self.memory_weight * (metrics.memory_percent / 100.0) +
            self.cpu_weight * (metrics.cpu_percent / 100.0) +
            self.latency_weight * norm_latency +
            self.exception_weight * norm_exc
        )
        F_substrate = float(np.clip(F_substrate, 0.0, 1.0))

        # 1:1 Topological Isomorphism Mapping (Digital Substrate <-> Biological/Human Manifold)
        isomorphic_mapping = {
            "memory_load": "Biological metabolic energy reserve & working memory tension",
            "cpu_latency": "Nerve signal conduction velocity & muscular fatigue",
            "runtime_exception": "Physical tissue injury & emotional trauma scar",
            "complex_impedance": "Visceral somatic state & homeostasis impedance Z",
            "boundary_limit": f"Digital substrate boundary defined at R={R:.2f}, X={X:.2f}"
        }

        return SomatosensorySignal(
            impedance_real=R,
            impedance_imag=X,
            complex_impedance=complex_z,
            friction_coefficient=F_substrate,
            isomorphic_mapping=isomorphic_mapping,
            raw_metrics=metrics
        )
