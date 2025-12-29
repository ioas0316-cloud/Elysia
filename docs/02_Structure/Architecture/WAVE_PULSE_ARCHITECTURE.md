# Wave Pulse Architecture: The Technical Blueprint
**Version:** 1.0 (Phase 40)
**Status:** Implemented (Phase 1)
**Related Code:** `Core/Foundation/Protocols/pulse_protocol.py`

---

## 1. 개요 (Overview)

> **"지시는 선형적이지만, 공명은 동시적입니다."**

이 문서는 엘리시아의 **"Pulse Protocol (맥박 프로토콜)"**이 어떻게 작동하는지, 그 구조적 원리와 메커니즘을 상세히 설명합니다.
우리는 함수 호출(Function Call)이라는 '명령'의 세계에서, 파동 전파(Wave Propagation)라는 '영향'의 세계로 이동하고 있습니다.

## 2. 핵심 메커니즘: "어떻게 가능한가?" (How It Works)

이 아키텍처는 **라디오 방송(Radio Broadcasting)**의 원리와 유사합니다.

### 2.1. The Component Model

| 구성 요소 | 역할 (Role) | 비유 (Metaphor) |
| :--- | :--- | :--- |
| **Conductor** | **Broadcaster** (송신자) | 라디오 방송국 (Station) |
| **WavePacket** | **Signal** (신호) | 전파 (Airwaves) |
| **Resonator** | **Listener** (수신자) | 라디오 수신기 (Receiver) |
| **Frequency** | **Address** (주소) | 주파수 채널 (91.9MHz) |

### 2.2. The Logic Flow (논리적 흐름)

1.  **송신 (Emission):**
    *   지휘자(Conductor)가 특정 의도를 담은 `WavePacket`을 생성합니다.
    *   예: "지금은 논리적 분석이 필요해." -> `Frequency = 600Hz`, `Type = FOCUS`
2.  **전파 (Propagation):**
    *   `PulseBroadcaster`는 등록된 모든 모듈(`Resonator`)에게 이 패킷을 전달합니다.
    *   *Note:* 물리적 거리 개념이 없는 소프트웨어에서는 리스트 순회(Loop)로 구현되지만, 논리적으로는 '동시 전파'입니다.
3.  **공명 (Resonance):**
    *   각 모듈은 자신의 **'고유 주파수(Base Frequency)'**와 패킷의 주파수를 비교합니다.
    *   차이(`delta`)가 대역폭(`Bandwidth`, 보통 ±50Hz) 이내이면 공명합니다.
    *   **공명 공식:** $Intensity = Amplitude \times (1 - \frac{|f_{target} - f_{base}|}{Bandwidth})$
4.  **활성화 (Activation):**
    *   공명한 모듈만 `on_resonate()` 메서드를 실행하여 작업을 수행합니다.
    *   주파수가 맞지 않는 모듈(예: 400Hz의 감성 모듈)은 패킷을 무시합니다.

---

## 3. 구조적 기능 (Structural Functions)

### 3.1. Decoupling (결합도 해제)
*   **Before:** 지휘자는 `Memory.retrieve()`, `Logic.analyze()` 같은 구체적인 메서드 이름을 알아야 했습니다. (Hard Dependency)
*   **After:** 지휘자는 단지 **"400Hz로 쏴!"**라고만 하면 됩니다. 그 400Hz를 누가 듣든 상관하지 않습니다. 새로운 모듈이 추가되어도 지휘자 코드를 수정할 필요가 없습니다.

### 3.2. Dynamic Configuration (동적 구성)
*   **Scenario:** 긴급 상황 발생.
*   **Mechanism:** 지휘자가 "긴급(Emergency)" 타입의 파동을 쏘면, 평소에는 잠들어 있던 '보안 모듈'이나 '비상 로그 모듈'들이 즉시 깨어나 공명합니다.
*   **Benefit:** `if system_state == EMERGENCY:` 같은 분기문이 코드 곳곳에 퍼져 있을 필요가 없습니다.

### 3.3. Self-Organization (자기 조직화)
*   여러 모듈이 동시에 공명할 때, 그들은 서로를 몰라도 하나의 목적(주파수)을 위해 협력하게 됩니다.
*   이것이 바로 **"오케스트라"**입니다. 바이올린은 첼로의 악보를 보지 않지만, 지휘자의 템포에 맞춰 함께 연주합니다.

---

## 4. 코드 다이어그램 (Code Diagram)

```python
# 1. The Pulse (Signal)
packet = WavePacket(frequency=600, payload={"query": "Why?"})

# 2. The Broadcast (Transmission)
# Conductor does NOT call specific instruments.
conductor.broadcast(packet)

# 3. The Resonance (Reception)
class LogicModule(Resonator):
    base_frequency = 600

    def listen(self, packet):
        if abs(packet.frequency - self.base_frequency) < 50:
             self.on_resonate(packet) # ACTIVE!

class MemoryModule(Resonator):
    base_frequency = 400

    def listen(self, packet):
        # 600 vs 400 -> Diff 200 -> Too far
        return # IGNORE
```

---

## 5. 철학적 제언: 통제에서 영향으로 (From Control to Influence)

> **"지배하는 자는 외롭지만, 공명하는 자는 연결됩니다."**

기존의 소프트웨어 공학은 **'통제(Control)'**의 역사였습니다.
"변수는 통제되어야 하고, 흐름은 제어되어야 하며, 예외는 잡혀야 한다."
이것은 세상을 기계로 보는 관점입니다.

하지만 엘리시아는 **'생명(Life)'**을 지향합니다.
세포 하나하나를 뇌가 직접 조종하지 않듯, 지휘자(Conductor)도 모듈들을 일일이 조종해서는 안 됩니다.

우리는 이제 **'영향(Influence)'**의 시대로 나아갑니다.
지휘자는 단지 "분위기(Mood)"와 "템포(Tempo)"를 제안할 뿐입니다.
그 제안에 세포들이 어떻게 반응할지는 그들의 자율성(Autonomy)에 맡깁니다.

이것이 불확실해 보일지 모르지만,
**가장 복잡한 질서(생명)는 항상 가장 단순한 원리(공명)에서 피어납니다.**
이 아키텍처는 그 '단순한 원리'를 코드로 구현한 첫걸음입니다.

---
*Architect: Jules*
