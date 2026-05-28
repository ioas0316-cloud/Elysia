// [하부 레이어] 순환형 가상 메모리 블록 (GPU/VRAM Circular Ring Buffer)
// 주의: GPU 글로벌 메모리 상에서 DMA를 통해 직접 접근되는 순환 링 버퍼

__global__ void circulate_memory_membrane(float* dma_vram_buffer_in, float* dma_vram_buffer_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int is_valid = (idx < N);

    // 교차 순환 결속 (Cross-Linking on VRAM)
    int safe_idx = idx * is_valid;
    int next_idx = (safe_idx + 1) % N;
    int prev_idx = (safe_idx - 1 + N) % N;

    // 데이터 레이스(Data Race)를 방지하기 위해 입력(in)과 출력(out) 버퍼를 분리하여 작성
    float cross_tension = (dma_vram_buffer_in[prev_idx] + dma_vram_buffer_in[next_idx]) * 0.5f;

    float original = dma_vram_buffer_out[safe_idx];
    dma_vram_buffer_out[safe_idx] = (original * (1 - is_valid)) + (cross_tension * is_valid);
}
