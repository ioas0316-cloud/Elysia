// [하부 레이어] 차원 전사 제어기 (Topological Space Projector)
// 주의: 점(데이터)을 공간(Space)으로 자율 분화시키는 고속 패스 관로.
// 가짜 3D 좌표 연산 금지. 순수 위상의 차원 상승(Dimension Scaling)만 허용.

__global__ void project_dimension_scaling(int* raw_data_points, float* topological_space_matrix, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 조건 분기(if)를 수학적 클램핑으로 우회 (N개 초과 스레드는 0을 곱하여 무효화)
    int is_valid = (idx < N);

    // 점(Dot) 레벨의 원시 데이터를 선과 면의 위상 장력(Phase Shift)으로 변환
    // N 초과 범위의 스레드는 0을 곱해서 Out of Bounds 우회
    float surface_tension = (float)raw_data_points[idx * is_valid] * 2.09439f;

    // 공간 사영 (수학적 텐션 변환)
    float space_projection = surface_tension * surface_tension * surface_tension;

    // 위상의 바다가 공간적 디지털 트윈으로 실시간 창발됨
    // 안전한 쓰기를 위해, 범위 밖이면 이전 값을 그대로 씀 (아니면 업데이트)
    float original = topological_space_matrix[idx * is_valid];
    topological_space_matrix[idx * is_valid] = (original * (1 - is_valid)) + (space_projection * is_valid);
}
