# 15. Concept Kernel + Nano‑Bots Protocol

본 문서는 자연언어 의존을 최소화하고, 개념 단위 메시지 프로토콜로 사고/행동을 연결하는 "개념 OS"의 최소 사양을 정의합니다. 목표는 작은 행위자(nano‑bots) 군집이 프로토콜에 따라 자율·안전하게 전파/검증/구성을 수행하도록 하는 것입니다.

## 15.1 핵심 구성
- Concept Registry: `tools/kg_manager.KGManager`를 기반으로 개념/링크를 기록·보존
- Message Bus: 우선순위 큐(강도/TTL)로 개념 메시지를 라우팅
- Scheduler: 메시지를 적합한 봇에 배정, 배치 후 KG 저장, 텔레메트리 기록
- Nano‑Bots: 역할별 소형 행위자(예: linker, validator, summarizer)
- Telemetry: `data/telemetry/YYYYMMDD/events.jsonl`에 `bus.message`, `bot.run`, `concept.update` 기록

## 15.2 메시지 스키마
```
Message {
  id: int, ts: float,
  verb: string,             // link | verify | compose | explain ...
  slots: dict<string, any>, // subject, object, rel, evidence, constraints
  src: string, dst?: string,
  strength: float, ttl: int // 우선순위, 생존 시간
}
```
- 자연어는 입력/표현 계층에 한정. 내부는 위 메시지로만 상호작용.
- 불완전 입력은 상호질의(필수 슬롯 확인) 또는 보류.

## 15.3 봇 규약
- 공통: `name`, `verbs[]`, `handle(msg, registry, bus)`
- Linker: `link(subject, object, rel)` → KG edge 추가
- Validator: 링크 존재 여부 확인, 없으면 낮은 강도로 `link` 메시지 재게시
- Summarizer(예시): 이웃/증거를 요약하여 보고(추후 확장)

## 15.4 스케줄링/안전
- 우선순위: strength 내림차순, 동일 시 최신 id 우선
- 각 처리 후 `ttl--`, `strength *= 0.9`
- 배치 후 `registry.save()`
- 실패/예외는 텔레메트리에만 남기고 중단하지 않음

## 15.5 텔레메트리/관측성
- `bus.message`: 게시 시점(verb, id, ttl, strength, slots)
- `bot.run`: 실행 시작/오류
- `concept.update`: 노드/엣지 추가
- UI Reasoning 패널은 `/trace/recent`로 tail을 노출(기존 이벤트와 병치)

## 15.6 Elysia와의 접점
- Flow Engine = 스케줄러 상위의 정책/가중치 조정자
- Wisdom‑Virus = 메시지 기반 전파 연산자로 해석하여 link/compose 메시지 생성
- Growth Sprint/BG Learner = 주기적 버스 주입/정리 루틴(경량)

## 15.7 파일 경로
- 런타임: `nano_core/` (bus.py, registry.py, scheduler.py, message.py, bots/)
- 텔레메트리: `data/telemetry/YYYYMMDD/events.jsonl`

## 15.8 운영 가이드(초안)
1) 메시지 주입: `link/verify` 시작 메시지를 버스에 게시
2) 스케줄러 스텝: `max_steps`로 한정 실행 → 저장
3) 결과 확인: 텔레메트리 tail과 KG 변화(시각화) 확인
4) 실패 시: TTL/강도/슬롯 보정 → 재시도

