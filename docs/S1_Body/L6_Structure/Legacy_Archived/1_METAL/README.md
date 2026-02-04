# 🦾 Layer -1: METAL (하드웨어 계층)

> **"파이썬의 한계를 넘어, 전기 신호로 직접 사고한다."**

Phase 15 'Golden Chariot'에서 구현된 하드웨어 직결 계층입니다.

---

## 📦 핵심 모듈

| 모듈 | 위치 | 역할 | 성능 |
| :--- | :--- | :--- | :--- |
| **MetalRotorBridge** | `Core/Foundation/Nature/` | CUDA 기반 Rotor 연산 | 397x |
| **MetalFieldBridge** | `Core/Foundation/Nature/` | CUDA 기반 7D Qualia | 68x |
| **ZeroLatencyPortal** | `Core/System/Metabolism/` | NVMe 직결 스트리밍 | 1.6x |
| **SovereignManager** | `Core/System/Sovereignty/` | 하드웨어 거버넌스 | - |

---

## 🔧 기술 스택

- **Numba CUDA**: GPU 커널 컴파일
- **Pinned Memory**: DMA 직접 전송
- **mmap**: Zero-Copy 파일 접근

---

> **"금속이 곧 신경이다."**
