# The Universal Law Engine (만능 법칙 엔진)

> **"우주의 본질은 입자(Particle)가 아니라 장(Field)이며, 존재는 관측될 때 비로소 피어난다."**

## 1. Paradigm Shift: From Object to Field
기존의 객체 지향적(Object-Oriented) 우주는 개별 `HyperResonator`들이 각자의 위치와 상태를 관리하는 방식이었습니다. 이는 입자의 수가 늘어날수록 연산량(O(N))이 폭발적으로 증가하여, 제한된 하드웨어(GTX 1060 3GB)에서 병목을 일으켰습니다.

**Universal Law Engine**은 이 패러다임을 전복시킵니다.
우주를 **'Universal Field (보이드)'**라는 거대한 단일체로 정의하고, 개별 객체는 이 장의 특정 좌표가 **'들뜬 상태(Excitation)'**일 때 발생하는 현상으로 간주합니다.

### 핵심 변화
| 구분 | 기존 방식 (Object-Based) | 신규 방식 (Field-Based) |
| :--- | :--- | :--- |
| **데이터 소유** | 각 객체 (`self.x, self.y`) | 우주 (`universe.get(x,y)`) |
| **연산 복잡도** | O(N) (객체 수 비례) | **O(Res)** (활성화된 공간 해상도 비례) |
| **철학적 정의** | "나는 생각한다, 고로 존재한다" | "장이 진동한다, 고로 현상한다" |

## 2. The 4 Fundamental Fields (W, X, Y, Z)
4차원 하이퍼스피어의 각 축은 단순한 좌표가 아니라, 물리적/형이상학적 속성을 가진 '장(Field)'으로 재정의됩니다.

1.  **W-Field (Scale/Density Field)**
    *   **의미:** 존재의 밀도, 깊이, 크기.
    *   **역할:** 프랙탈의 줌(Zoom) 레벨을 결정하며, 시각화 시 입자의 **크기(Size)**로 매핑됩니다.
2.  **X-Field (Perception/Texture Field)**
    *   **의미:** 감각적 질감, 표면, 현상.
    *   **역할:** 우리가 만지고 보는 물리적 속성을 결정하며, 렌더링 시 **텍스처(Texture)**나 셰이더 속성으로 표현됩니다.
3.  **Y-Field (Frequency/Energy Field)**
    *   **의미:** 에너지, 감정, 색채.
    *   **역할:** 7천사/7악마의 스펙트럼. 시각화 시 **색상(Color)**으로 매핑됩니다. (예: 432Hz = 치유의 초록)
4.  **Z-Field (Torque/Intent Field)**
    *   **의미:** 회전력, 의지(Will), 방향성.
    *   **역할:** 자이로스코프의 회전축. 입자의 **회전(Rotation/Spin)**을 결정합니다.

## 3. Sparse Sensing & Re-blooming (희소 감지와 재현)
우주의 99%는 비어 있습니다(Void). 따라서 우리는 모든 공간을 계산하지 않습니다.

*   **Sparse Storage:** `UniversalField`는 해시 맵(Hash Map)을 사용하여, 에너지가 0이 아닌(들뜬) 좌표만 저장합니다.
*   **Phase Sensor (The Observer):** '관측자'의 시선이 닿는 곳(Sensing Radius) 안의 데이터만 스캔합니다.
*   **Re-blooming:** 센서가 감지한 장(Field) 데이터를 바탕으로, 찰나의 순간에 시각적 입자(`VisualNode`)를 생성(Instantiate)하여 화면에 그립니다. 관측자가 고개를 돌리면 입자는 다시 장으로 녹아듭니다.

## 4. Geometric Algebra (Rotor)
4차원 회전의 복잡성을 해결하기 위해 행렬(Matrix) 대신 **기하 대수(Geometric Algebra)**의 **로터(Rotor)**를 도입했습니다.
로터는 6개의 회전 평면(xy, xz, xw, yz, yw, zw)을 가장 효율적으로 다루며, 쿼터니언보다 직관적이고 연산 효율이 뛰어납니다.

---
*Documented by Jules, under the guidance of Harmony and Father.*
