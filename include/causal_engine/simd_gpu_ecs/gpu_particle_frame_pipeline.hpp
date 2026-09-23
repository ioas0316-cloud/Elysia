#ifndef GPU_PARTICLE_FRAME_PIPELINE_HPP
#define GPU_PARTICLE_FRAME_PIPELINE_HPP

#include "d3d12_mock.hpp"

struct FrameConstants
{
    XMMATRIX ViewProjMatrix;
    XMMATRIX PrevViewProjMatrix;
    XMMATRIX ProjMatrix;
    XMMATRIX PrevProjMatrix;
    XMFLOAT3 CameraWorldVelocity;
    float    DeltaTime;
    XMFLOAT2 HzbScreenDimensions;
    uint32_t HzbMaxMipLevel;
    float    BaseDepthBias;
    uint32_t TotalParticleCount;
    uint32_t IndexCountPerQuad;
};

class GPUParticleSystemManager
{
public:
    void ExecuteParticleFrame(
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
    );
};

#endif // GPU_PARTICLE_FRAME_PIPELINE_HPP
