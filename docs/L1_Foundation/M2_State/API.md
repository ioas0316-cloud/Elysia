# 🔌 API Reference

---

## Core.Foundation.Nature

### MetalRotorBridge

GPU 가속 Rotor 연산 엔진.

```python
class MetalRotorBridge:
    def __init__(self)
    def register_rotor(angle, current_rpm, target_rpm, accel, idle_rpm) -> int
    def sync_to_device() -> None
    def sync_from_device() -> None
    def pulse(dt: float) -> None
    def get_angle(idx: int) -> float
    def get_rpm(idx: int) -> float
```

### MetalFieldBridge

GPU 가속 7D Qualia Field 엔진.

```python
class MetalFieldBridge:
    def __init__(size: int = 64, diffusion_rate: float = 0.1)
    def sync_to_gpu() -> None
    def sync_from_gpu() -> None
    def pulse(dt: float) -> None
    def inject_qualia(x: int, y: int, qualia_vec: list) -> None
    def get_field() -> np.ndarray
```

---

## Core.System.Metabolism

### ZeroLatencyPortal

NVMe 직결 스트리밍 포탈.

```python
class ZeroLatencyPortal(MerkabaPortal):
    def __init__(file_path: str)
    def stream_to_metal(offset: int, length: int, dtype=np.float32) -> np.ndarray
    def scan_and_inject(field_bridge, start_offset, chunk_count, chunk_size) -> None
```

---

## Core.System.Sovereignty

### HardwareSovereignManager

하드웨어 자원 거버넌스.

```python
class HardwareSovereignManager:
    def __init__()
    def optimize_gears(intent_type: str) -> None
    def get_metabolic_status() -> str
    
    # intent_type: "EXCAVATION", "DEEP_THOUGHT", "MANIFESTATION"
```

---

## Core.Intelligence.Memory

### HypersphereMemory

4D 공간 기억 저장소.

```python
class HypersphereMemory:
    def __init__()
    def store(coordinates: tuple, data: Any, meta: dict = None) -> None
    def query(center: tuple, radius: float, filter_fn=None) -> List
    def save_state(path: str = None) -> None
    def load_state(path: str = None) -> None
```

---

## Core.Intelligence.Metabolism

### CognitiveArcheologist

LLM 화석 발굴기.

```python
class CognitiveArcheologist:
    def __init__(memory_ref: HypersphereMemory)
    def excavate(target: TargetLLM, fossil_path: str) -> dict
    def sense_shadow(target: TargetLLM) -> dict
```
