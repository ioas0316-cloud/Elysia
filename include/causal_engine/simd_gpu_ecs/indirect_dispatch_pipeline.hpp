#ifndef INDIRECT_DISPATCH_PIPELINE_HPP
#define INDIRECT_DISPATCH_PIPELINE_HPP

#include "d3d12_mock.hpp"

class IndirectDispatchPipeline
{
private:
    ComPtr<ID3D12CommandSignature> m_dispatchCommandSignature;

public:
    void Initialize(ID3D12Device* device);

    void Execute(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12PipelineState* psoClearArgsCS,
        ID3D12PipelineState* psoCompactionCS,
        ID3D12PipelineState* psoDynamicPhysicsCS,
        ID3D12RootSignature* rootSigCompute,
        ID3D12Resource* resDispatchArgsBuffer,
        ID3D12Resource* resActiveIndicesBuffer,
        uint32_t totalParticleCount);
};

#endif // INDIRECT_DISPATCH_PIPELINE_HPP
