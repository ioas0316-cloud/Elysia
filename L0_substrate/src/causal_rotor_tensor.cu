#include "causal_rotor_tensor.hpp"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace elysia {

// Device 전용 Helper: 사차수 켤레 (Quaternion Conjugate) -> q^-1
__device__ __forceinline__ float4 QuaternionConjugate(float4 q) {
    return make_float4(-q.x, -q.y, -q.z, q.w);
}

// Device 전용 Helper: 사차수 곱셈 (Quaternion Multiplication)
// q1 = (x1, y1, z1, w1), q2 = (x2, y2, z2, w2)
__device__ __forceinline__ float4 QuaternionMultiply(float4 q1, float4 q2) {
    return make_float4(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

// Device 전용 Helper: Logarithmic Map (S^3 -> so(3) Torque Vector)
__device__ __forceinline__ float3 QuaternionToLogTorque(float4 qErr) {
    float vecLen = sqrtf(qErr.x * qErr.x + qErr.y * qErr.y + qErr.z * qErr.z);
    if (vecLen < 1e-6f) {
        return make_float3(0.0f, 0.0f, 0.0f); // 위상 오차 없음
    }
    float angle = 2.0f * atan2f(vecLen, qErr.w);
    float factor = angle / vecLen;
    return make_float3(qErr.x * factor, qErr.y * factor, qErr.z * factor);
}

// [Main CUDA Kernel] 병렬 사차수 위상 고정 및 동역학 갱신
__global__ void ComputeRotorPhaseLockKernel(
    RotorNodeData* __restrict__ nodePool,
    uint32_t totalNodes,
    float deltaTime,
    float lockTolerance)
{
    uint32_t nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (nodeIdx >= totalNodes) return;

    RotorNodeData node = nodePool[nodeIdx];
    float4 q_i = node.quaternion;
    float4 q_i_inv = QuaternionConjugate(q_i);

    float3 accumulatedTorque = make_float3(0.0f, 0.0f, 0.0f);
    float totalTorqueMag = 0.0f;
    uint32_t activeEdges = 0;

    // 1. 최대 8개의 기어 연결 엣지 순회 (Coalesced Access)
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        uint32_t targetIdx = node.connectedNodeIndices[k];
        if (targetIdx == 0xFFFFFFFF || targetIdx >= totalNodes) continue; // 연결 없음

        float4 q_j = nodePool[targetIdx].quaternion;
        float gearRatio = node.gearRatios[k];

        // 상대 위상 오차 사차수 계산: q_err = q_i^-1 * q_j
        float4 q_err = QuaternionMultiply(q_i_inv, q_j);

        // Log Map으로 so(3) 토크 추출 및 기어비 반영
        float3 torque = QuaternionToLogTorque(q_err);
        accumulatedTorque.x += torque.x * gearRatio;
        accumulatedTorque.y += torque.y * gearRatio;
        accumulatedTorque.z += torque.z * gearRatio;

        totalTorqueMag += sqrtf(torque.x * torque.x + torque.y * torque.y + torque.z * torque.z);
        activeEdges++;
    }

    // 2. 각속도(omega) 업데이트: d(omega)/dt = -beta * omega + sum(Torque)
    float3 omega = node.angularVelocity;
    omega.x = (omega.x + accumulatedTorque.x * deltaTime) * (1.0f - node.dampingBeta * deltaTime);
    omega.y = (omega.y + accumulatedTorque.y * deltaTime) * (1.0f - node.dampingBeta * deltaTime);
    omega.z = (omega.z + accumulatedTorque.z * deltaTime) * (1.0f - node.dampingBeta * deltaTime);

    // 3. 사차수 위상 갱신 적분: q(t+dt) = q(t) + 0.5 * dt * (q(t) (x) (0, omega))
    float4 omegaQuat = make_float4(omega.x, omega.y, omega.z, 0.0f);
    float4 dq = QuaternionMultiply(q_i, omegaQuat);

    q_i.x += 0.5f * dq.x * deltaTime;
    q_i.y += 0.5f * dq.y * deltaTime;
    q_i.z += 0.5f * dq.z * deltaTime;
    q_i.w += 0.5f * dq.w * deltaTime;

    // 사차수 정규화 (Unit Quaternion 유지)
    float qLen = sqrtf(q_i.x * q_i.x + q_i.y * q_i.y + q_i.z * q_i.z + q_i.w * q_i.w);
    if (qLen > 1e-6f) {
        q_i.x /= qLen; q_i.y /= qLen; q_i.z /= qLen; q_i.w /= qLen;
    }

    // 4. Phase-Lock 달성 판단 (토크 변동성이 수렴했는가)
    uint32_t isLock = 0;
    if (activeEdges > 0 && (totalTorqueMag / activeEdges) < lockTolerance) {
        isLock = 1; // Attractor Basin 진입 -> VRAM Pinned 고정 대상
    }

    // VRAM 메모리 쓰기
    nodePool[nodeIdx].quaternion = q_i;
    nodePool[nodeIdx].angularVelocity = omega;
    nodePool[nodeIdx].isAttractor = isLock;
}

extern "C" void LaunchComputeRotorPhaseLockKernel(
    RotorNodeData* d_nodePool,
    uint32_t totalNodes,
    float deltaTime,
    float lockTolerance)
{
    if (totalNodes == 0) return;
    int blockSize = 256;
    int gridSize = (totalNodes + blockSize - 1) / blockSize;
    ComputeRotorPhaseLockKernel<<<gridSize, blockSize>>>(d_nodePool, totalNodes, deltaTime, lockTolerance);
    cudaDeviceSynchronize();
}

} // namespace elysia

#endif // WITH_CUDA
