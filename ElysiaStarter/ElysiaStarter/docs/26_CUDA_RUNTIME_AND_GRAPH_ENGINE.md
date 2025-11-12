# 26_CUDA_RUNTIME_AND_GRAPH_ENGINE

- StreamPool / GraphCapture / MemoryArena 설계 개요
- 합성곱(∇, ∇²) 커널 타일링(16×16+halo)
- FP16 연산, accumulate FP32
- CUDA Graph로 반복 루프 캡처/재생
- 결과 동일성(Determinism) 가드: CPU 대비 허용오차 epsilon
