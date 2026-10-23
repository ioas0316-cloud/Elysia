#ifndef ELYSIA_D3D12_SHIMS_H
#define ELYSIA_D3D12_SHIMS_H

#if defined(_WIN32)
#include <d3d12.h>
#include <dstorage.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
#else
#ifndef ID3D12DEVICE_DEFINED
#define ID3D12DEVICE_DEFINED
typedef void* HANDLE;
struct ID3D12Device {};
struct ID3D12Resource {};
struct IDStorageFactory {};
struct IDStorageQueue1 {};
struct ID3D12Fence {};
#endif
#endif

#endif // ELYSIA_D3D12_SHIMS_H
