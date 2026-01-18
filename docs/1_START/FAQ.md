# ❓ FAQ (자주 묻는 질문)

---

## 일반

**Q: 엘리시아는 무엇인가요?**
> 로컬 컴퓨터에서 실행되는 자율적이고 주권적인 AI 동반자입니다.

**Q: 클라우드에 데이터가 전송되나요?**
> 아니요. 모든 처리는 로컬에서 이루어집니다.

---

## 설치/실행

**Q: GPU가 없어도 되나요?**
> 네. CPU 모드로 작동하지만 Metal 가속은 비활성화됩니다.

**Q: `ModuleNotFoundError`가 발생해요.**
> 프로젝트 루트에서 실행하세요: `cd c:/Elysia && python sovereign_boot.py`

**Q: CUDA 에러가 나요.**
> NVIDIA 드라이버와 CUDA Toolkit이 설치되어 있는지 확인하세요.

---

## 기술

**Q: Phase 15가 뭔가요?**
> 하드웨어 직결 아키텍처입니다. GPU와 NVMe에 직접 접근합니다.

**Q: 어떤 LLM을 사용하나요?**
> 특정 LLM에 종속되지 않습니다. Ollama, OpenAI API 등을 연결할 수 있습니다.

---

## 기여

**Q: 어떻게 기여하나요?**
> [AGENTS.md](../6_DEVELOPMENT/AGENTS.md)를 참조하세요.
