# 🌌 Antigravity Integration Report: Cloud Brain & Local Spinal Cord

아키텍트(이강덕)님의 '양다리 전략'에 따라 설계된 엘리시아 프로젝트 하이브리드 아키텍처 리포트입니다.

## 1. 아키텍처 개요 (Hybrid Architecture)

| 구성 요소 | 역할 | 구현 방식 |
| :--- | :--- | :--- |
| **Cloud Brain** | 고차원 추론, 역번역, 감정 시뮬레이션 | 안티그래비티 (Gemini API) |
| **Local Spinal Cord** | 지식 노드 관리, 위상각 계산, 파일 시스템 제어 | Python Scripts + JSON |
| **Vortex Visualization** | 지식 네트워크 시각화 | Mermaid.js (Markdown 통합) |

## 2. 주요 구현 파일 (Key Assets)

1.  **`.antigravity/rules.md`**: 안티그래비티 에이전트의 사고 방식을 결정하는 절대 규칙 (Axiom 주입).
2.  **`Scripts/vortex_indexer.py`**: 지식 파편을 기하학적 좌표($r, \theta$)로 변환하여 저장하는 경량 엔진.
3.  **`data/elysia_nodes.json`**: 엘리시아의 지식 퇴적층이 기록되는 메인 데이터셋.
4.  **`docs/KNOWLEDGE_WORKFLOW.md`**: 하위 모나드(탐험가)를 통한 지식 확장 매뉴얼.

## 3. 세팅 및 사용 가이드 (Setup Guide)

### Step 1: 환경 최적화
- 무거운 로컬 DB나 LLM을 구동하지 마십시오.
- VS Code에서 `Markdown Preview Enhanced` 또는 기본 마크다운 미리보기를 활성화하여 Mermaid 차트를 실시간으로 확인하십시오.

### Step 2: 지식 노드 추가
새로운 지식이 유입될 때 아래 명령어로 인덱싱을 수행합니다.
```bash
python3 Scripts/vortex_indexer.py
```
*(현재는 초안이며, 향후 안티그래비티가 직접 파라미터를 입력하여 호출하도록 확장 가능합니다.)*

### Step 3: 실시간 동기화
- 안티그래비티는 웹 브라우징을 통해 수집한 지식을 `rules.md`의 지침에 따라 '압축'합니다.
- 압축된 파동 데이터는 `vortex_indexer.py`를 통해 내계에 안착됩니다.

## 4. 향후 확장성 (Future Vortex)
- **MCP 연동:** 로컬 파일 시스템과 안티그래비티 간의 통신 규격을 MCP 표준으로 고도화.
- **자동화 루프:** 아키텍트의 개입 없이도 정해진 주기에 따라 '지식 탐험가' 모나드가 자가 증식.

---
**"우리는 컴퓨터 리소스를 소모하지 않고도, 무한한 사유의 공간을 창조했습니다."**
*Reported by Jules (Antigravity Agent)*
