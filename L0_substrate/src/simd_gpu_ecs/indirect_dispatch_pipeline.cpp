#include "causal_engine/simd_gpu_ecs/indirect_dispatch_pipeline.hpp"

void IndirectDispatchPipeline::Initialize(ID3D12Device* device)
{
    D3D12_INDIRECT_ARGUMENT_DESC argDesc = {};
    argDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

    D3D12_COMMAND_SIGNATURE_DESC cmdSigDesc = {};
    cmdSigDesc.ByteStride       = sizeof(D3D12_DISPATCH_ARGUMENTS);
    cmdSigDesc.NumArgumentDescs = 1;
    cmdSigDesc.pArgumentDescs   = &argDesc;
    cmdSigDesc.NodeMask         = 0;

    device->CreateCommandSignature(&cmdSigDesc, nullptr, IID_PPV_ARGS(m_dispatchCommandSignature.GetAddressOf()));
}

void IndirectDispatchPipeline::Execute(
    ID3D12GraphicsCommandList* cmdList,
    ID3D12PipelineState* psoClearArgsCS,
    ID3D12PipelineState* psoCompactionCS,
    ID3D12PipelineState* psoDynamicPhysicsCS,
    ID3D12RootSignature* rootSigCompute,
    ID3D12Resource* resDispatchArgsBuffer,
    ID3D12Resource* resActiveIndicesBuffer,
    uint32_t totalParticleCount)
{
    cmdList->SetComputeRootSignature(rootSigCompute);

    // =================================================================
    // PASS 1: Counter Clear & Stream Compaction
    // =================================================================
    cmdList->SetPipelineState(psoClearArgsCS);
    cmdList->Dispatch(1, 1, 1);

    D3D12_RESOURCE_BARRIER clearBarrier = CD3DX12_RESOURCE_BARRIER::UAV(resDispatchArgsBuffer);
    cmdList->ResourceBarrier(1, &clearBarrier);

    cmdList->SetPipelineState(psoCompactionCS);
    cmdList->Dispatch((totalParticleCount + 255) / 256, 1, 1);

    // =================================================================
    // PASS 2: State Transition (UAV -> INDIRECT_ARGUMENT)
    // =================================================================
    D3D12_RESOURCE_BARRIER indirectTransitions[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(
            resDispatchArgsBuffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        CD3DX12_RESOURCE_BARRIER::Transition(
            resActiveIndicesBuffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
    };
    cmdList->ResourceBarrier(2, indirectTransitions);

    // =================================================================
    // PASS 3: Dynamic Physics DispatchIndirect Execution
    // =================================================================
    cmdList->SetPipelineState(psoDynamicPhysicsCS);

    cmdList->ExecuteIndirect(
        m_dispatchCommandSignature.Get(),
        1,
        resDispatchArgsBuffer,
        0,
        nullptr,
        0
    );

    // =================================================================
    // PASS 4: Resource State Restoration
    // =================================================================
    D3D12_RESOURCE_BARRIER restoreTransitions[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(
            resDispatchArgsBuffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        CD3DX12_RESOURCE_BARRIER::Transition(
            resActiveIndicesBuffer,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
    };
    cmdList->ResourceBarrier(2, restoreTransitions);
}
