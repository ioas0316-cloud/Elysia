// elysia_d3d12_raymarcher.cpp - Windows D3D12 Volume Raymarcher Application
#if defined(_WIN32)
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <iostream>
#include "include/topir_runtime_engine.hpp"

using Microsoft::WRL::ComPtr;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    std::cout << "Elysia D3D12 Volume Raymarcher initialized.\n";
    return 0;
}
#endif
