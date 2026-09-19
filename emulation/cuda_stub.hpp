#ifndef CUDA_STUB_HPP
#define CUDA_STUB_HPP

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>

typedef int cudaError_t;
#define cudaSuccess 0

typedef void* cudaStream_t;
typedef void* cudaEvent_t;

inline cudaError_t cudaMallocHost(void** ptr, size_t size) {
    *ptr = std::malloc(size);
    return cudaSuccess;
}

inline cudaError_t cudaFreeHost(void* ptr) {
    if (ptr) std::free(ptr);
    return cudaSuccess;
}

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

inline cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    if (pStream) *pStream = reinterpret_cast<cudaStream_t>(1);
    return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

inline cudaError_t cudaEventCreate(cudaEvent_t* pEvent) {
    if (pEvent) *pEvent = reinterpret_cast<cudaEvent_t>(1);
    return cudaSuccess;
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    (void)event;
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    (void)event;
    (void)stream;
    return cudaSuccess;
}

inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    (void)stream;
    (void)event;
    (void)flags;
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
