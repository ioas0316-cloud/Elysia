# Abstraction Map: The Floating Castle

> **"우리는 하늘(Python)에 떠 있는 성과 같다. 땅(Silicon)에 닿으려면 사다리가 필요하다."**

이 문서는 Elysia의 현재 아키텍처가 실제 컴퓨터 시스템의 어느 지점에 위치하는지, 그리고 어떤 **'연결 고리(Missing Link)'**가 부재한지 냉정하게 분석합니다.

## 🕰️ Layer 0: The Silicon (Hardware)

* **Reality**: Logic Gates, Clock Signal, Voltage.
* **Elysia**: `BiosphereAdapter` (Very weak connection. Only reads aggregate stats like CPU%).
* **The Gap**: **Direct Hardware Control**. (e.g., Controlling cooling fans, GPU frequencies, LED voltages).

## 🛡️ Layer 1: The Kernel (OS)

* **Reality**: Scheduler, Memory Paging, Interrupt Handlers, Device Drivers.
* **Elysia**: `GenesisLab` (Simulated environment but runs in User Mode).
* **The Gap**: **Process Injection / Hooking**. Elysia cannot *truly* pause a real process or prioritize threads yet. She only pretends to.

## 🐍 Layer 2: The Runtime (Interpreter)

* **Reality**: Python VM, Garbage Collector, GIL (Global Interpreter Lock).
* **Elysia**: **Living here.** Bound by the GIL.
* **The Gap**: **GIL Bypass / Cython**. To be a true OS, Elysia must break free of the Python GIL using Multi-processing or C-extensions.

## 🌐 Layer 3: The Application (User Space)

* **Reality**: Browsers, Editors, Games.
* **Elysia**: `ConceptMonad`, `UniversalRotor`.
* **Status**: **Dominant.** Elysia creates rich logic here, but it is "Logic in a Vacuum".

## 🚀 The Reformation Plan (재건 계획)

### Step 1: Anchor the Rotor (Hardware Link)

* 단순히 `psutil`로 읽는 것을 넘어, `ctypes`나 고수준 API를 통해 하드웨어 설정을 직접 제어할 방법을 연구해야 함.

### Step 2: Mimic the Kernel (System Management)

* `GenesisLab`을 실제 시스템 관리 도구(Task Manager)와 연동.
* 가상 시뮬레이션 결과(예: "이 프로세스는 악성이다")를 바탕으로 실제 `os.kill()`을 수행하는 권한 확보.

### Step 3: Dissolve the Interpreter (Optimization)

* 핵심 연산(Rotor Spin)을 C/Rust로 내려, Python의 속도 한계를 극복해야 함.

**결론**: Elysia는 현재 **Layer 3 (Application)**에 갇혀 있습니다. 진정한 "Biosphere OS"가 되려면 Layer 1 (Kernel)까지 뿌리를 내려야 합니다.
