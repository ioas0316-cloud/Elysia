World Model: Planet Overlay (View Layer)

요약
- 이 지구형 모델은 ‘표현/주의 뷰’다. KG/추론/도시 인프라(기능)는 불변이며, 행성 좌표는 모듈을 배치해서 흐름을 직관화하는 지도다.

레이어(alt)
- 핵 core (0.20): EventBus, Telemetry, Identity/Covenant
- 맨틀 mantle (0.50): KG, Wave(Activation), CognitionPipeline 추론 층
- 지각 crust (0.80): Reasoner/Planner/Sensory/Value/Journal/Action 등 코어 도시
- 대기 atmosphere (1.00): Lens/Attention/Starmap(표현/감응)

좌표 규칙
- 위도(lat): subject/district 해시(geometry/social/ethics/etc.)
- 경도(lon): module role 해시(reasoner/planner/sensory/value/action/journal/etc.)
- 고도(alt): 위 레이어 값으로 고정(기능 의미 불변)

파일
- 입력: 모듈 목록(meta) → `tools/geo_registry.py`
- 출력: `data/world/planet.json` (대륙/도시/항로 포함 가능)

주의
- Truth 불변: KG/기하/인과/데이터는 바꾸지 않는다.
- View 전용: 시각화·주의 가중·리포팅에만 사용한다.

