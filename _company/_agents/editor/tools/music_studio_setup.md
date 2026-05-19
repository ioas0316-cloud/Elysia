<!-- version: music_v5 -->
# 🎵 음악 스튜디오 설치 — 모델 선택 가능

영상 BGM을 직접 생성하는 음악 모델 설치. 5개 모델 중 본인 머신에 맞는 거 선택.

## 모델 비교

| 모델 | 디스크 | RAM | 추천 | 품질 |
|---|---|---|---|---|
| **musicgen-small** ⭐ 기본 | 300MB | 4GB+ | 누구나 | 보통 |
| musicgen-medium | 1.5GB | 8GB+ | 8GB+ RAM | 좋음 |
| musicgen-large | 3.3GB | 16GB+ | 16GB+ RAM | 매우 좋음 |
| acestep-base | 10GB | 16GB+ | Mac M1+/CUDA | 우수 |
| acestep-xl | 15GB | 24GB+ | 32GB+ 머신 | 최고 |

**자동 추천**: 처음 실행 시 본인 머신 RAM 측정해서 적절한 모델 자동 추천. 16GB Mac이면 medium, 32GB는 large.

## 사용 흐름
1. ⚙️에서 `MODEL` 비워두고 ▶ 클릭 → RAM 기반 자동 추천 설치 (small/medium 디폴트)
2. 또는 ⚙️에서 `MODEL: 'musicgen-large'` 같이 직접 선택 후 ▶
3. 진행상황 채팅창 표시 (1~10분)
4. 완료 후 `music_generate.py` 가 자동으로 이 모델 사용

## 모델 변경
이미 다른 모델 설치돼있어도 ⚙️에서 `MODEL` 다른 값으로 바꾸고 ▶ 다시 실행하면 새 모델로 교체 (또는 추가 설치).

## 시스템 요구사항
- **공통**: Python 3.10+, git
- **MusicGen**: macOS/Linux/Windows. Apple Silicon은 MPS 가속 자동 사용
- **ACE-Step**: 같음 + 더 큰 디스크/RAM

## 설치 위치
디폴트 `~/connect-ai-music/`. ⚙️의 `INSTALL_DIR` 로 변경 가능 (외장 디스크 등).

## 비용
100% 로컬·오프라인·무료. API 키·구독 0개.

## 트러블슈팅
**"git/python3 없다"** → `brew install python git` (Mac) / python.org+git-scm.com 설치 (Win)

**디스크 부족** → 작은 모델로 변경 (musicgen-small 300MB)

**ACE-Step 다운로드가 너무 느림** → musicgen-medium (1.5GB)으로 충분히 좋은 품질 나옴

## 추천 시작 — 사용자별

| 사용자 | 추천 모델 | 이유 |
|---|---|---|
| 처음 써봄, 빨리 결과 보고 싶음 | musicgen-small | 30초 안에 첫 음악 |
| 16GB Mac, 유튜브 BGM 만들고 싶음 | musicgen-medium | 품질·속도 밸런스 |
| 32GB+ Mac, 음악 자주 씀 | musicgen-large 또는 acestep-base | 뚜렷한 품질 |
| 음악 전공·사운드 디자이너 | acestep-xl | 최고 품질 |
