# 10. Config Schema

본 문서는 `config.json`의 스키마와 기본값, 검증 경고(`config.warn`) 매핑을 정의합니다.

## 10.1. 최상위 구조 예시
{
  "filesystem": {
    "enabled": true,
    "root": "./Elysia_Input_Sanctum",
    "read_only": true,
    "allowed_exts": [".md", ".txt"],
    "ignore_globs": ["**/.git/**", "**/__pycache__/**"],
    "max_file_mb": 16,
    "hash_algo": null,
    "auto_index_on_start": false,
    "save_index": null
  }
}

## 10.2. 키 설명
- `enabled: boolean` — 기능 활성화
- `root: string` — 샌드박스 루트 경로(필수)
- `read_only: boolean` — 쓰기 금지 여부
- `allowed_exts: string[]` — 허용 확장자 화이트리스트(없으면 전체 허용)
- `ignore_globs: string[]` — 무시할 경로 패턴(glob)
- `max_file_mb: integer` — 허용 최대 파일 크기(MB)
- `hash_algo: string|null` — 예: "sha1" 또는 null
- `auto_index_on_start: boolean` — 초기 인덱스 자동 수행
- `save_index: string|null` — 인덱스 저장 경로

## 10.3. 검증 및 경고 매핑
검증은 `Project_Sophia/config_loader.py: validate_config()`에서 수행되며, 문제 발견 시 다음과 같이 이벤트가 발행됩니다.
- `config.warn { section: "filesystem", issue: "not_a_dict" }`
- `config.warn { section: "filesystem", issue: "unknown_keys", keys: [..] }`
- `config.warn { section: "filesystem", issue: "root_missing_or_invalid" }`
- `config.warn { section: "filesystem", issue: "allowed_exts_not_list" }` 등

## 10.4. 기본값 권장안
- 운영 안전 기본값: `read_only=true`, `max_file_mb=16`, `.git`/`__pycache__` 무시
- 콘텐츠 민감도에 따라 `allowed_exts`를 축소 적용

