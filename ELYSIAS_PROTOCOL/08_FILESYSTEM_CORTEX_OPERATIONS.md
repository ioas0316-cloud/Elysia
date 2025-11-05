# 8. FileSystemCortex Operations

본 문서는 FileSystemCortex의 보안/운영 원칙과 API, 인덱싱·검색, 감시(Watch), 영속화 규범을 정의합니다.

## 8.1. 보안 원칙
- 샌드박스: `root` 내부 경로만 허용. 절대경로/`..`/심볼릭 링크로의 탈출 금지.
- 정책: `read_only`, `allowed_exts`, `ignore_globs`, `max_file_mb`를 통한 위험 최소화.
- 실패 격리: 예외는 `Fs*Error`로 매핑, 텔레메트리는 실패해도 기능을 방해하지 않음.

## 8.2. 공개 API (요약)
- `scan(rel_path='.', depth=1) -> [FileMeta]`
- `read(rel_path, as_text=False, encoding='utf-8') -> Document`
- `write(rel_path, data, overwrite=False, encoding='utf-8') -> FileMeta` (read_only=false일 때)
- `delete(rel_path) -> None` (파일만)
- `move(src_rel, dst_rel, overwrite=False) -> FileMeta`
- `index(rel_paths=None) -> None`
- `search(query, limit=50) -> [FileMeta]` (예: `name:report ext:.md`)
- `save_index(rel_path='index.json') -> Path`
- `load_index(rel_path='index.json') -> int`
- `watch(rel_path='.', interval=1.0, stop_after=None) -> Iterator[FsEvent]`
- `watch_once(rel_path='.', prev_state=None) -> (events, state)`

## 8.3. 인덱싱/검색 가이드
- 초기 버전은 파일명/확장자 위주 필터(`name:`, `ext:`)를 지원.
- 대용량/빈번 변경 시 `save_index`와 `watch`를 병행해 최신 상태를 유지.
- 민감 정보가 있는 디렉터리는 `allowed_exts`, `ignore_globs`로 명확히 제한.

## 8.4. 감시(Watch)
- 기본 구현은 폴링 기반. 이동은 삭제+생성으로 관측됨.
- 장시간 실행 시 `interval`을 상향하거나 OS 이벤트 기반(후속 옵션)을 고려.

## 8.5. 영속화(Persistence)
- JSON 인덱스 저장/복구 제공. 대규모 인덱스는 SQLite 옵션을 차후 도입.
- 저장 시 임시 파일을 사용하여 부분 쓰기 실패로부터 보호.

## 8.6. 텔레메트리
- `fs.op`: `ns, op, path, bytes, status, latency_ms, error?`
- `fs.index`, `fs.index.saved`, `fs.index.error`
- 기록 위치: `data/telemetry/YYYYMMDD/events.jsonl`

