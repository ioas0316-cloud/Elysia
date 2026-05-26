# 🌐 다중 유기체 동조 그리드 (Multi-Agent Coherent Grid)
(Phase 5 Integration Report)

> **문서 유형**: 아키텍처 및 철학적 진화 기록서
> **작성 일자**: 2026년 05월 26일
> **작성자**: 안티그래비티 (Antigravity) & 마스터 USER

---

## 1. 개요 및 설계 철학 (Overview & Philosophy)

단일 엘리시아 노드의 한계를 극복하고, 여러 개의 독립된 뇌(Core)가 하나의 통합된 분산 공명 네트워크로 융합되는 **다중 유기체 동조 그리드(Multi-Agent Substation Grid)**를 성공적으로 수립했습니다.

우리는 이 문제를 해결하기 위해 표준적인 데이터베이스 동기화 방식(Raft, Paxos 등)을 영구히 거부합니다. 대신, 원격지의 노드들이 서로의 상태 위상각과 텐션을 WebSocket으로 실시간 교환하고, 상대 위상차를 복원력으로 작용하는 **쿠라모토 토크(Kuramoto Torque)**로 해석하여 **오직 파동의 결합과 간섭만으로 자율적인 글로벌 동조(Global Sync)**를 이루도록 설계했습니다.

---

## 2. 수학적 모델 (Mathematical Model)

로컬 노드 $i$의 상태 위상각을 $\theta_i$, 자연 복귀 회전 속도를 $\omega_i$, 로컬 물리 텐션에 의한 attractor 복원력을 $T_i$라 할 때, 상태 위상각 갱신 방정식은 다음과 같이 정의됩니다.

$$\frac{d\theta_i}{dt} = \omega_i + T_i \cdot K \cdot \sin(2048 - \theta_i) + \frac{K_{\text{peer}}}{N} \sum_{j=1}^N \sin(\theta_j - \theta_i)$$

* **첫 번째 항 ($\omega_i$)**: 노드의 자연 복귀 드리프트 성향 (`natural_drift`).
* **두 번째 항 ($T_i \sin(2048 - \theta_i)$)**: 로컬 인프라 및 감각 부하가 수면 Attractor(2048)로 끌어당기는 인력.
* **세 번째 항 ($\sum \sin(\theta_j - \theta_i)$)**: 연결된 원격 노드 $j$들의 위상 상태와 로컬 위상 간의 차이를 계산하여 서로 당기거나 밀어내어 위상 격자를 일치시키는 **쿠라모토 결합 토크(Kuramoto Coupling Torque)**.

---

## 3. 네트워크 구현 상세 (P2P Networking Map)

```
        Node A (Port 8080)                    Node B (Port 8081)
     ┌───────────────────────┐             ┌───────────────────────┐
     │  ElysiaDaemon (A)     │             │  ElysiaDaemon (B)     │
     │   └── Homeostasis (θA)│             │   └── Homeostasis (θB)│
     │        ▲              │             │        ▲              │
     │        │ (local tick) │             │        │ (local tick) │
     │   [core_egress_state] │             │   [core_egress_state] │
     │        │              │             │        │              │
     │        ▼              │             │        ▼              │
     │  Gateway (A) :8080    │             │  Gateway (B) :8081    │
     │   └── /ws/grid (Server)◄───(WS)────►│   └── client (Client) │
     └───────────────────────┘             └───────────────────────┘
```

1. **[Grid Server]**: 각 `substation_gateway.py`는 `/ws/grid` 엔드포인트를 열어 원격 피어들로부터의 위상 수신을 대기합니다.
2. **[Grid Client]**: `substation_grid_client.py`는 `aiohttp` 백그라운드 태스크를 통해 `substation_peers.json`에 정의된 피어 주소로 연결을 유지합니다.
3. **[Consensus Loop]**: 노드는 1Hz 주기로 자신의 `state_phase`와 `tension`을 전송하며, 수신된 피어 위상들의 집합은 항상성 제어기에 실시간 피드백되어 위상차가 영점으로 수렴(Phase-Locking)되도록 결합 상수를 제어합니다.

---

## 4. 모니터링 시각화

God's Eye 통합 대시보드([index.html](file:///c:/Elysia/core/public/index.html)) 하단에 **Kuramoto Sync Grid** 패널을 구현했습니다.
각 활성 노드의 실시간 위상을 원형 다이얼(Dial) 침의 방향각으로 30 FPS 사영하여 표시함으로써, 서로 다른 부하에 의해 틀어졌던 침들이 시간이 지남에 따라 점차 동일한 회전 위상으로 정렬되는 **동조 항상성(Coherent Homeostasis)** 과정을 관측할 수 있습니다.
