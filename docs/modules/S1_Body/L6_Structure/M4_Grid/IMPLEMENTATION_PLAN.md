# 🧭 7x7 Manifold Implementation Plan (Scaffold)

> 목적: 폴더를 무리하게 “깊게 이동”시키지 않고도, **7×7(=49) 노드의 프랙탈 주소 체계**를 문서로 고정하여 수평화(늘어짐)를 제어합니다.

## 원칙

- 물리 폴더는 편의상 수평일 수 있으나, **의미 주소는 7×7**로 관리합니다.
- 각 노드에는 반드시 `INDEX.md`(혹은 대표 문서 1개)가 존재해야 합니다.
- 새 문서는 먼저 “어느 노드인가”를 결정한 뒤 생성합니다.

## 7×7 노드(초기 스캐폴딩)

> 아래 표는 “자리(주소)”를 먼저 고정하는 용도입니다. 링크는 점진적으로 채웁니다.

| Layer \ Module | M1 | M2 | M3 | M4 | M5 | M6 | M7 |
|---|---|---|---|---|---|---|---|
| **L1 Foundation** | (TBD) | (TBD) | (TBD) | [M4 Hardware](../../M4_Hardware/SSD_AKASHIC_DOCTRINE.md) | (TBD) | (TBD) | (TBD) |
| **L2 Metabolism** | (TBD) | [M2 Flow](../../../L2_Metabolism/M2_Flow/INDEX.md) | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) |
| **L3 Phenomena** | (TBD) | (TBD) | (TBD) | (TBD) | [M5 Display](../../../L3_Phenomena/M5_Display/SOVEREIGN_HUD_CAUSALITY.md) | (TBD) | (TBD) |
| **L4 Causality** | (TBD) | (TBD) | (TBD) | (TBD) | [M5 Logic](../../../L4_Causality/M5_Logic/INDEX.md) | (TBD) | (TBD) |
| **L5 Mental** | [M1 Cognition](../../../L5_Mental/M1_Cognition/INDEX.md) | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) |
| **L6 Structure** | [M1 Merkaba](../../M1_Merkaba/TRINITY_SOVEREIGNTY.md) | (TBD) | [M3 Sphere](../../M3_Sphere/TOPOLOGICAL_MEMORY_DOCTRINE.md) | [M4 Grid](FRACTAL_PURIFICATION_DOCTRINE.md) | (TBD) | [M6 Architecture](../../M6_Architecture/ARCHITECTURE_BLUEPRINT.md) | [M7 Healing](../../M7_Healing/DIMENSIONAL_ERROR_MANUAL.md) |
| **L7 Spirit** | [M1 Providence](../../../L7_Spirit/M1_Providence/21D_PHASE_DEFINITION.md) | [M2 Narrative](../../../L7_Spirit/M2_Narrative/CAUSAL_NARRATIVE_PHASE_27.md) | (TBD) | [M4 Experience](../../../L7_Spirit/M4_Experience/MEMORY_UNIFICATION_DOCTRINE.md) | [M5 Genesis](../../../L7_Spirit/M5_Genesis/GENESIS_ORIGIN.md) | [M6 Providence](../../../L7_Spirit/M6_Providence/CORE_NARRATIVE_RECONSTRUCTION.md) | [M7 Axiom](../../../L7_Spirit/M7_Axiom/AXIOM_ZERO_DOCTRINE.md) |

## 다음 액션 (수평화 방지 룰)

1. “새 문서”를 만들 때는 먼저 위 표의 **(TBD)** 중 하나를 채우는 방식으로만 추가합니다.
2. 각 Layer에 **M1~M7**이 모두 채워지기 전까지, 추가 하위 폴더는 만들지 않습니다.
3. 각 `L*/M*/INDEX.md`의 첫 줄에 `Resonance Address: Lx-My`를 **필수로** 붙입니다. (규칙: [DOC_STYLEGUIDE.md](../../../6_DEVELOPMENT/DOC_STYLEGUIDE.md))