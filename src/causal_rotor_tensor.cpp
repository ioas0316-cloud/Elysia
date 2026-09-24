#include "causal_rotor_tensor.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
extern "C" void LaunchComputeRotorPhaseLockKernel(
    elysia::RotorNodeData* d_nodePool,
    uint32_t totalNodes,
    float deltaTime,
    float lockTolerance);
#endif

namespace elysia {

CausalRotorTensorSystem::CausalRotorTensorSystem(uint32_t maxNodes)
    : m_maxNodes(maxNodes), m_nodeCount(0), d_nodesBuffer(nullptr), m_cudaInitialized(false)
{
    m_nodes.reserve(m_maxNodes);
    initializeCUDA();
}

CausalRotorTensorSystem::~CausalRotorTensorSystem() {
    freeCUDA();
}

void CausalRotorTensorSystem::initializeCUDA() {
#ifdef WITH_CUDA
    cudaError_t err = cudaMalloc((void**)&d_nodesBuffer, m_maxNodes * sizeof(RotorNodeData));
    if (err == cudaSuccess) {
        m_cudaInitialized = true;
    } else {
        std::cerr << "[CausalRotorTensorSystem] CUDA allocation failed, falling back to CPU mode.\n";
        m_cudaInitialized = false;
        d_nodesBuffer = nullptr;
    }
#else
    m_cudaInitialized = false;
    d_nodesBuffer = nullptr;
#endif
}

void CausalRotorTensorSystem::freeCUDA() {
#ifdef WITH_CUDA
    if (d_nodesBuffer) {
        cudaFree(d_nodesBuffer);
        d_nodesBuffer = nullptr;
    }
#endif
    m_cudaInitialized = false;
}

uint32_t CausalRotorTensorSystem::addRotorNode(uint32_t gridX, uint32_t gridY, uint32_t gridZ, float dampingBeta) {
    if (m_nodeCount >= m_maxNodes) {
        std::cerr << "[CausalRotorTensorSystem] Max nodes capacity reached (" << m_maxNodes << ")\n";
        return 0xFFFFFFFF;
    }

    uint32_t newIdx = m_nodeCount++;
    RotorNodeData node;
    node.spatialHashKey = Morton3D::encode(gridX, gridY, gridZ);
    node.dampingBeta = dampingBeta;
    node.usageFrequency = 1;
    node.isAttractor = 0;
    // VRAM Slab virtual address lookup mock
    node.vramSlabAddress = 0x7FFF00000000ULL + newIdx * sizeof(RotorNodeData);

    if (newIdx < m_nodes.size()) {
        m_nodes[newIdx] = node;
    } else {
        m_nodes.push_back(node);
    }

    return newIdx;
}

bool CausalRotorTensorSystem::connectNodes(uint32_t sourceIdx, uint32_t targetIdx, float gearRatio) {
    if (sourceIdx >= m_nodeCount || targetIdx >= m_nodeCount || sourceIdx == targetIdx) {
        return false;
    }

    RotorNodeData& sourceNode = m_nodes[sourceIdx];
    for (int k = 0; k < 8; ++k) {
        if (sourceNode.connectedNodeIndices[k] == 0xFFFFFFFF || sourceNode.connectedNodeIndices[k] == targetIdx) {
            sourceNode.connectedNodeIndices[k] = targetIdx;
            sourceNode.gearRatios[k] = gearRatio;
            return true;
        }
    }
    return false; // slots full
}

void CausalRotorTensorSystem::injectImpulse(uint32_t nodeIdx, float wx, float wy, float wz) {
    if (nodeIdx >= m_nodeCount) return;

    m_nodes[nodeIdx].angularVelocity.x += wx;
    m_nodes[nodeIdx].angularVelocity.y += wy;
    m_nodes[nodeIdx].angularVelocity.z += wz;
    m_nodes[nodeIdx].usageFrequency++;
}

void CausalRotorTensorSystem::stepSimulation(float deltaTime, float lockTolerance) {
    if (m_nodeCount == 0) return;

#ifdef WITH_CUDA
    if (m_cudaInitialized && d_nodesBuffer) {
        // Copy Host -> Device
        cudaMemcpy(d_nodesBuffer, m_nodes.data(), m_nodeCount * sizeof(RotorNodeData), cudaMemcpyHostToDevice);

        // Launch Kernel
        LaunchComputeRotorPhaseLockKernel(d_nodesBuffer, m_nodeCount, deltaTime, lockTolerance);

        // Copy Device -> Host
        cudaMemcpy(m_nodes.data(), d_nodesBuffer, m_nodeCount * sizeof(RotorNodeData), cudaMemcpyDeviceToHost);
        return;
    }
#endif

    // CPU Fallback Simulation
    for (uint32_t i = 0; i < m_nodeCount; ++i) {
        RotorNodeData& node = m_nodes[i];

        Float4 q_i = node.quaternion;
        // q_i inverse = (-x, -y, -z, w)
        Float4 q_i_inv(-q_i.x, -q_i.y, -q_i.z, q_i.w);

        Float3 accumulatedTorque(0.0f, 0.0f, 0.0f);
        float totalTorqueMag = 0.0f;
        uint32_t activeEdges = 0;

        for (int k = 0; k < 8; ++k) {
            uint32_t targetIdx = node.connectedNodeIndices[k];
            if (targetIdx == 0xFFFFFFFF || targetIdx >= m_nodeCount) continue;

            Float4 q_j = m_nodes[targetIdx].quaternion;
            float gearRatio = node.gearRatios[k];

            // Quaternion multiply q_err = q_i_inv * q_j
            Float4 q_err(
                q_i_inv.w * q_j.x + q_i_inv.x * q_j.w + q_i_inv.y * q_j.z - q_i_inv.z * q_j.y,
                q_i_inv.w * q_j.y - q_i_inv.x * q_j.z + q_i_inv.y * q_j.w + q_i_inv.z * q_j.x,
                q_i_inv.w * q_j.z + q_i_inv.x * q_j.y - q_i_inv.y * q_j.x + q_i_inv.z * q_j.w,
                q_i_inv.w * q_j.w - q_i_inv.x * q_j.x - q_i_inv.y * q_j.y - q_i_inv.z * q_j.z
            );

            // Log torque
            float vecLen = std::sqrt(q_err.x * q_err.x + q_err.y * q_err.y + q_err.z * q_err.z);
            Float3 torque(0.0f, 0.0f, 0.0f);
            if (vecLen >= 1e-6f) {
                float angle = 2.0f * std::atan2(vecLen, q_err.w);
                float factor = angle / vecLen;
                torque = Float3(q_err.x * factor, q_err.y * factor, q_err.z * factor);
            }

            accumulatedTorque.x += torque.x * gearRatio;
            accumulatedTorque.y += torque.y * gearRatio;
            accumulatedTorque.z += torque.z * gearRatio;

            totalTorqueMag += std::sqrt(torque.x * torque.x + torque.y * torque.y + torque.z * torque.z);
            activeEdges++;
        }

        // Velocity update
        Float3 omega = node.angularVelocity;
        omega.x = (omega.x + accumulatedTorque.x * deltaTime) * (1.0f - node.dampingBeta * deltaTime);
        omega.y = (omega.y + accumulatedTorque.y * deltaTime) * (1.0f - node.dampingBeta * deltaTime);
        omega.z = (omega.z + accumulatedTorque.z * deltaTime) * (1.0f - node.dampingBeta * deltaTime);

        // Quaternion phase update
        Float4 omegaQuat(omega.x, omega.y, omega.z, 0.0f);
        Float4 dq(
            q_i.w * omegaQuat.x + q_i.x * omegaQuat.w + q_i.y * omegaQuat.z - q_i.z * omegaQuat.y,
            q_i.w * omegaQuat.y - q_i.x * omegaQuat.z + q_i.y * omegaQuat.w + q_i.z * omegaQuat.x,
            q_i.w * omegaQuat.z + q_i.x * omegaQuat.y - q_i.y * omegaQuat.x + q_i.z * omegaQuat.w,
            q_i.w * omegaQuat.w - q_i.x * omegaQuat.x - q_i.y * omegaQuat.y - q_i.z * omegaQuat.z
        );

        q_i.x += 0.5f * dq.x * deltaTime;
        q_i.y += 0.5f * dq.y * deltaTime;
        q_i.z += 0.5f * dq.z * deltaTime;
        q_i.w += 0.5f * dq.w * deltaTime;

        float qLen = std::sqrt(q_i.x * q_i.x + q_i.y * q_i.y + q_i.z * q_i.z + q_i.w * q_i.w);
        if (qLen > 1e-6f) {
            q_i.x /= qLen; q_i.y /= qLen; q_i.z /= qLen; q_i.w /= qLen;
        }

        uint32_t isLock = 0;
        if (activeEdges > 0 && (totalTorqueMag / activeEdges) < lockTolerance) {
            isLock = 1;
        }

        node.quaternion = q_i;
        node.angularVelocity = omega;
        node.isAttractor = isLock;
    }
}

uint32_t CausalRotorTensorSystem::getAttractorCount() const {
    uint32_t count = 0;
    for (uint32_t i = 0; i < m_nodeCount; ++i) {
        if (m_nodes[i].isAttractor == 1) {
            count++;
        }
    }
    return count;
}

std::vector<RotorNodeData> CausalRotorTensorSystem::getNodeBuffer() const {
    return std::vector<RotorNodeData>(m_nodes.begin(), m_nodes.begin() + m_nodeCount);
}

const RotorNodeData* CausalRotorTensorSystem::getRawDataPointer() const {
    return m_nodes.data();
}

} // namespace elysia
