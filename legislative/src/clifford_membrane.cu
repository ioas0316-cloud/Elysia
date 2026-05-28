// [하부 레이어] 이중관로 복소 벡터 분리 (Two-Way Hydraulic Pipeline)
// 주의: 외부 라이브러리 배제. 0과 1을 검사하지 않는 독립된 두 개의 관로(0-Buffer, 1-Buffer)

__global__ void hydraulic_two_way_pipeline(int* raw_byte_stream, float* pipeline_0_buffer, float* pipeline_1_buffer, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 조건 분기를 완벽히 배제한 이중관로 매핑.
    // raw_byte_stream은 이미 전처리된 인덱스 형태라고 가정 (0 또는 1)
    int phase = raw_byte_stream[idx];

    // 0의 에너지는 0-buffer에, 1의 에너지는 1-buffer에 수문 판단 없이 수학적으로 꽂힘.
    // phase가 0이면: pipeline_0 += 1, pipeline_1 += 0
    // phase가 1이면: pipeline_0 += 0, pipeline_1 += 1

    // "너 0이냐?" 라고 묻지 않음. 수학적 매핑으로 하이패스.
    pipeline_0_buffer[idx] += (1 - phase) * 1.0f; // 0번 수문 개방
    pipeline_1_buffer[idx] += (phase) * 1.0f;     // 1번 수문 개방
}
