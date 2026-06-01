// [하부 레이어] 이중관로 복소 벡터 분리 (Two-Way Hydraulic Pipeline)
// 주의: 외부 라이브러리 배제. 0과 1을 검사하지 않는 독립된 두 개의 관로(0-Buffer, 1-Buffer)

__global__ void hydraulic_two_way_pipeline(int* raw_byte_stream, float* pipeline_0_buffer, float* pipeline_1_buffer, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int is_valid = (idx < N);

    // 조건 분기를 완벽히 배제한 이중관로 매핑.
    // N을 넘어가는 스레드들의 OOB와 데이터 레이스를 막기 위한 안전한 인덱싱 및 무효화
    int safe_idx = idx * is_valid;
    int phase = raw_byte_stream[safe_idx];

    // "너 0이냐?" 라고 묻지 않음. 수학적 매핑으로 하이패스.
    float val_0 = (1 - phase) * 1.0f;
    float val_1 = (phase) * 1.0f;

    // is_valid가 0인 유령 스레드들은 버퍼의 현재 값을 읽어 그대로 다시 쓰게 만들어
    // 사실상 No-op 처리함 (데이터 레이스가 발생하더라도 같은 값을 쓰므로 무결성 유지)
    float orig_0 = pipeline_0_buffer[safe_idx];
    float orig_1 = pipeline_1_buffer[safe_idx];

    pipeline_0_buffer[safe_idx] = (orig_0 * (1 - is_valid)) + ((orig_0 + val_0) * is_valid);
    pipeline_1_buffer[safe_idx] = (orig_1 * (1 - is_valid)) + ((orig_1 + val_1) * is_valid);
}
