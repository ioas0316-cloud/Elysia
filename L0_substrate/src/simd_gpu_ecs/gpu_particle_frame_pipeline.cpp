#include "causal_engine/simd_gpu_ecs/gpu_particle_frame_pipeline.hpp"
#include <algorithm>

void GPUParticleSystemManager::ExecuteParticleFrame(
    ID3D12GraphicsCommandList* cmdList,
    const FrameConstants& frameConstants,
    ID3D12PipelineState* psoTorquePhysicsCS,
    ID3D12PipelineState* psoParticleSimCS,
    ID3D12PipelineState* psoHzbBuildCS,
    ID3D12PipelineState* psoClearIndirectArgsCS,
    ID3D12PipelineState* psoDynamicHzbCullCS,
    ID3D12PipelineState* psoParticleRenderVSPS,
    ID3D12RootSignature* rootSigCompute,
    ID3D12RootSignature* rootSigGraphics,
    ID3D12CommandSignature* commandSignature,
    ID3D12Resource* resDerivedOutputUAV,
    ID3D12Resource* resAngularVelocityUAV,
    ID3D12Resource* resParticleBufferUAV,
    ID3D12Resource* resHzbTexture,
    ID3D12Resource* resMotionVectorSRV,
    ID3D12Resource* resIndirectArgsBufferUAV,
    ID3D12Resource* resCulledIndicesUAV,
    D3D12_GPU_DESCRIPTOR_HANDLE hzbMip0UAV,
    D3D12_GPU_DESCRIPTOR_HANDLE hzbFullSRV
)
{
    // =================================================================
    // PASS 1: Torque Transmission & Particle Physics Integration CS
    // =================================================================
    cmdList->SetComputeRootSignature(rootSigCompute);

    // [Stage 1A] Torque Physics Dispatch
    cmdList->SetPipelineState(psoTorquePhysicsCS);
    cmdList->Dispatch((frameConstants.TotalParticleCount + 255) / 256, 1, 1);

    // Barrier: Torque Buffers UAV -> Shader Resource
    D3D12_RESOURCE_BARRIER torqueBarriers[2] = {
        CD3DX12_RESOURCE_BARRIER::UAV(resDerivedOutputUAV),
        CD3DX12_RESOURCE_BARRIER::UAV(resAngularVelocityUAV)
    };
    cmdList->ResourceBarrier(2, torqueBarriers);

    // [Stage 1B] Particle Emission & Simulation Dispatch
    cmdList->SetPipelineState(psoParticleSimCS);
    cmdList->Dispatch((frameConstants.TotalParticleCount + 255) / 256, 1, 1);

    // Barrier: Particle Buffer Physics Write 완료
    D3D12_RESOURCE_BARRIER particleUavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(resParticleBufferUAV);
    cmdList->ResourceBarrier(1, &particleUavBarrier);


    // =================================================================
    // PASS 2: HZB Depth Pyramid Max-Reduction Downsample CS
    // =================================================================
    cmdList->SetPipelineState(psoHzbBuildCS);

    // Depth Buffer -> SRV 전환
    D3D12_RESOURCE_BARRIER hzbInitBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        resHzbTexture, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmdList->ResourceBarrier(1, &hzbInitBarrier);

    // HZB Mipmap 피라미드 순차 연산 (Mip N -> Mip N+1)
    uint32_t currentWidth  = static_cast<uint32_t>(frameConstants.HzbScreenDimensions.x);
    uint32_t currentHeight = static_cast<uint32_t>(frameConstants.HzbScreenDimensions.y);

    for (uint32_t mip = 0; mip < frameConstants.HzbMaxMipLevel; ++mip)
    {
        currentWidth  = std::max(1u, currentWidth >> 1);
        currentHeight = std::max(1u, currentHeight >> 1);

        uint32_t dispatchX = (currentWidth  + 15) / 16;
        uint32_t dispatchY = (currentHeight + 15) / 16;

        cmdList->Dispatch(dispatchX, dispatchY, 1);

        D3D12_RESOURCE_BARRIER hzbMipBarrier = CD3DX12_RESOURCE_BARRIER::UAV(resHzbTexture);
        cmdList->ResourceBarrier(1, &hzbMipBarrier);
    }


    // =================================================================
    // PASS 3: Indirect Argument Reset Kernel (Clear Instance Count)
    // =================================================================
    cmdList->SetPipelineState(psoClearIndirectArgsCS);
    cmdList->Dispatch(1, 1, 1);

    D3D12_RESOURCE_BARRIER argClearBarrier = CD3DX12_RESOURCE_BARRIER::UAV(resIndirectArgsBufferUAV);
    cmdList->ResourceBarrier(1, &argClearBarrier);


    // =================================================================
    // PASS 4: Frustum + Motion Vector Dynamic HZB Occlusion Culling CS
    // =================================================================
    cmdList->SetPipelineState(psoDynamicHzbCullCS);

    cmdList->SetComputeRoot32BitConstants(0, sizeof(FrameConstants) / 4, &frameConstants, 0);

    uint32_t cullDispatchX = (frameConstants.TotalParticleCount + 255) / 256;
    cmdList->Dispatch(cullDispatchX, 1, 1);

    D3D12_RESOURCE_BARRIER renderTransitions[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(
            resIndirectArgsBufferUAV,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        CD3DX12_RESOURCE_BARRIER::Transition(
            resCulledIndicesUAV,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
    };
    cmdList->ResourceBarrier(2, renderTransitions);


    // =================================================================
    // PASS 5: ExecuteIndirect Zero-VB Instanced Particle Render Pass
    // =================================================================
    cmdList->SetGraphicsRootSignature(rootSigGraphics);
    cmdList->SetPipelineState(psoParticleRenderVSPS);

    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmdList->IASetVertexBuffers(0, 0, nullptr);

    cmdList->ExecuteIndirect(
        commandSignature,
        1,
        resIndirectArgsBufferUAV,
        0,
        nullptr,
        0
    );


    // =================================================================
    // PASS 6: Post-Render State Restoration
    // =================================================================
    D3D12_RESOURCE_BARRIER restoreTransitions[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(
            resIndirectArgsBufferUAV,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        CD3DX12_RESOURCE_BARRIER::Transition(
            resCulledIndicesUAV,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
    };
    cmdList->ResourceBarrier(2, restoreTransitions);
}
