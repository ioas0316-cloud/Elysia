#ifndef D3D12_MOCK_HPP
#define D3D12_MOCK_HPP

// Platform / Mock DirectX 12 definitions for non-Windows / Linux compilation & architecture verification
#include <cstdint>
#include <algorithm>

#if defined(_WIN32) || defined(__CYGWIN__)
#include <d3d12.h>
#include <DirectXMath.h>
using namespace DirectX;
#else

struct XMFLOAT2 { float x, y; };
struct XMFLOAT3 { float x, y, z; };
struct XMFLOAT4 { float x, y, z, w; };

struct XMMATRIX {
    float m[4][4];
};

struct D3D12_RESOURCE_BARRIER {};
struct D3D12_GPU_DESCRIPTOR_HANDLE { uint64_t ptr; };

class ID3D12Resource {};
class ID3D12PipelineState {};
class ID3D12RootSignature {};
class ID3D12CommandSignature {};

#define D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST 4

class ID3D12GraphicsCommandList {
public:
    void SetComputeRootSignature(ID3D12RootSignature*) {}
    void SetGraphicsRootSignature(ID3D12RootSignature*) {}
    void SetPipelineState(ID3D12PipelineState*) {}
    void Dispatch(uint32_t x, uint32_t y, uint32_t z) {}
    void ResourceBarrier(uint32_t count, const D3D12_RESOURCE_BARRIER* barriers) {}
    void SetComputeRoot32BitConstants(uint32_t rootParam, uint32_t num32BitValuesToSet, const void* pSrcData, uint32_t destOffset) {}
    void IASetPrimitiveTopology(int topology) {}
    void IASetVertexBuffers(uint32_t startSlot, uint32_t numViews, const void* pViews) {}
    void ExecuteIndirect(ID3D12CommandSignature* cmdSig, uint32_t maxCmdCount, ID3D12Resource* argBuf, uint64_t argBufOffset, ID3D12Resource* countBuf, uint64_t countBufOffset) {}
};

class ID3D12Device {
public:
    void CreateCommandSignature(const void* desc, void* rootSig, const void* riid, void** ppv) {}
};

namespace CD3DX12_RESOURCE_BARRIER {
    inline D3D12_RESOURCE_BARRIER UAV(ID3D12Resource*) { return {}; }
    inline D3D12_RESOURCE_BARRIER Transition(ID3D12Resource*, int, int) { return {}; }
}

enum D3D12_RESOURCE_STATES {
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS = 0,
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE = 1,
    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT = 2,
    D3D12_RESOURCE_STATE_DEPTH_WRITE = 3
};

enum D3D12_INDIRECT_ARGUMENT_TYPE {
    D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH = 0,
    D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED = 1
};

struct D3D12_INDIRECT_ARGUMENT_DESC {
    D3D12_INDIRECT_ARGUMENT_TYPE Type;
};

struct D3D12_COMMAND_SIGNATURE_DESC {
    uint32_t ByteStride;
    uint32_t NumArgumentDescs;
    const D3D12_INDIRECT_ARGUMENT_DESC* pArgumentDescs;
    uint32_t NodeMask;
};

struct D3D12_DISPATCH_ARGUMENTS {
    uint32_t ThreadGroupCountX;
    uint32_t ThreadGroupCountY;
    uint32_t ThreadGroupCountZ;
};

template <typename T>
class ComPtr {
    T* ptr = nullptr;
public:
    ComPtr() = default;
    ComPtr(T* p) : ptr(p) {}
    T* Get() const { return ptr; }
    T** GetAddressOf() { return &ptr; }
    T* operator->() const { return ptr; }
};

#define IID_PPV_ARGS(ppType) nullptr, (void**)(ppType)

#endif // _WIN32

#endif // D3D12_MOCK_HPP
