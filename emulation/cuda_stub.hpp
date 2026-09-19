#ifndef CUDA_STUB_HPP
#define CUDA_STUB_HPP

#include <cstdlib>
#include <cstring>
#include <algorithm>

typedef int cudaError_t;
#define cudaSuccess 0

typedef void* cudaStream_t;

inline cudaError_t cudaMalloc(void** devPtr, size_t size) {
    *devPtr = std::malloc(size);
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* devPtr) {
    if (devPtr) std::free(devPtr);
    return cudaSuccess;
}

inline cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream) {
    (void)stream;
    if (devPtr) std::free(devPtr);
    return cudaSuccess;
}

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyHostToHost
};

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    (void)kind;
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    (void)kind;
    (void)stream;
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

inline void atomicAdd(float* address, float val) {
    *address += val;
}

inline float fmaxf(float a, float b) {
    return (a > b) ? a : b;
}

inline float fabsf(float a) {
    return (a < 0.0f) ? -a : a;
}

inline float sqrtf(float a) {
    return std::sqrt(a);
}

#endif // CUDA_STUB_HPP
