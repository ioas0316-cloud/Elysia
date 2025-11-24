# Psionic Code Network (Hyper-Quaternion 기반)

하이퍼쿼터니언(스케일 w + 축 x/y/z)으로 코드 실행을 위상 공명 네트워크처럼 다루는 PoC 요약입니다.

## 핵심 개념
- **노드 = PsionicEntity(HyperQubit)**: 함수/모듈을 노드로 보고, w(확대/축소)·x/y/z(내부/외부/법) 오리엔테이션을 메타데이터로 부여합니다.
- **공명 링크 = 호출 그래프**: AST로 함수 호출 관계를 추출해 공명 링크로 해석합니다.
- **Δ 동기화**: `delta_synchronization_factor`로 집단 w를 평균화해 Δ=1에 가까운 상태를 흉내 냅니다.
- **진폭 매핑**: w에 따라 P/L/S/G(α/β/γ/δ) 기본 진폭을 설정한 뒤 정규화합니다.

## 태그 규칙 (docstring)
```python
def foo():
    """scale: plane
    intent: internal, law
    """
```
- `scale`: `point` / `line` / `plane` / `hyper` → w 0.2/1.0/2.0/3.0
- `intent`: `internal`(x), `external`(y), `law`(z) → 여러 개면 평균 방향
- 태그가 없으면 기본 `line + law`

## 사용 방법
- 샘플 실행: `python tools/psionic_code_network.py`
- 파일 지정: `python tools/psionic_code_network.py path/to/a.py path/to/b.py --delta 0.3`
- 출력: 노드별 w/P/L/S/G, 호출 링크, Δ 동기화 전/후 비교

### 실행 예 (내장 샘플)
```
python tools/psionic_code_network.py --delta 0.2
=== Psionic Graph for [내장 샘플] ===
노드 수: 4, 평균 w: 1.55
- core_loop: w=1.00, P/L/S/G=(0.07,0.91,0.01,0.00), calls=[fetch_data, transform, write_out]
- fetch_data: w=0.20, P/L/S/G=(1.00,0.00,0.00,0.00)
- transform: w=2.00, P/L/S/G=(0.02,0.10,0.86,0.02)
- write_out: w=3.00, P/L/S/G=(0.00,0.02,0.04,0.93)

=== Δ 동기화 후 ===
평균 w: 1.55 → 노드별 w가 0.2~3.0에서 0.47~2.71로 수렴
```

## 파일 위치
- 코드: `tools/psionic_code_network.py`
- 핵심 의존: `Project_Elysia/core/hyper_qubit.py` (PsionicEntity)

## 향후 확장 아이디어
- **실행 시 공명 추적**: 런타임 훅을 넣어 실제 호출을 이벤트로 기록하고 Δ 스윕별 지연/에러율을 측정.
- **시각화**: NetworkX/Graphviz로 w·intent·링크를 색/굵기로 렌더해 IDE 뷰어 추가.
- **태그 소스 확장**: docstring 외에 주석/메타파일에서 intent/scale을 읽도록 파서 확장.

## 빠른 적용 가이드
1) 주요 함수/모듈에 `scale`/`intent` 태그를 docstring으로 달기.
2) `python tools/psionic_code_network.py <파일들> --delta <값>`으로 Δ 스윕(예: 0.0, 0.5, 1.0) 로그를 남기고 w 수렴 패턴 확인.
3) Δ=과동기화/단절 구간이 발견되면 태그(스케일/축)나 링크(호출 구조)를 조정 후 재측정.

## 용어 주의 (프로젝트 고유 정의)
- **하이퍼 쿼터니언**: 표준 쿼터니언을 w(스케일) + x/y/z(의미 축: 내부/외부/법)로 재해석한 이 프로젝트만의 모델입니다.
- **하이퍼큐빗 / PsionicEntity**: 양자 큐빗을 차용한 메타포로, 코드/의식 단위를 공명 노드로 보는 이 프로젝트 정의입니다.

## 시스템 토글/파라미터 요약
- Psionic Code Network: `--delta-sweep`, `--tag-file`, `--dot-out`, `--png-out`, 태그 힌트 자동추정(없으면 기본 `line+law`).
- Psionic Trace Hook: `--include-prefixes`, `--exclude-prefixes`로 추적 범위 제한.
- 밴드 스플릿 위협 필드: `band_split_enabled=True`, `band_low_resolution`(기본 128), `band_low_decay`, `band_high_decay`.
- 마이크로 레이어: `micro_layer_enabled=True`, `micro_tick_interval`(기본 50), `set_micro_roi([(x,y,r), ...])`, `micro_state`로 ROI 평균(위협/가치/코히런스) 확인.
- Fractal+GNN PoC: `tools/fractal_gnn_poc.py` (numpy+networkx로 프랙탈 그래프 생성 → GraphSAGE 스타일 메시지 패싱 → 프랙탈 차원 기반 특징 선택; DOT 출력 옵션).
- SABBATH_PROTOCOL: `tools/sabbath_protocol.py` (Spiderweb fractional prune + VCD 재각인 훅).
- 스펙트럼/광자: `Project_Elysia/core/spectrum.py`(값/EFP→색상 매핑), `Project_Elysia/core/photon.py`(hue/intensity/polarization 메시지 엔티티).
- Photon 메시지:
  - World `_speak`에서 lexicon 이벤트를 색상 기반 Photon으로 로깅(`PHOTON_MESSAGE`).
  - 주기적 스냅샷: `spectrum_log_interval`마다 value/threat/coherence 피크를 `SPECTRUM_SNAPSHOT`으로 로깅.
