#pragma once

// Host stubs for CUDA types and API calls when compiling with non-CUDA C++ host compiler
#if !defined(__CUDACC__) && !defined(__CUDACC_RTC__)

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>

typedef int cudaError_t;
#define cudaSuccess 0

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

#define cudaStreamNonBlocking 0x01

typedef void* cudaStream_t;
typedef void* cudaEvent_t;

struct float3 {
    float x, y, z;
};

struct int2 {
    int x, y;
};

struct int3 {
    int x, y, z;
};

struct uchar4 {
    unsigned char x, y, z, w;
};

inline int2 make_int2(int x, int y) {
    int2 v; v.x = x; v.y = y; return v;
}

inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) {
    uchar4 v; v.x = x; v.y = y; v.z = z; v.w = w; return v;
}


struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};

struct uint3_stub {
    unsigned int x, y, z;
    uint3_stub(unsigned int vx = 0, unsigned int vy = 0, unsigned int vz = 0) : x(vx), y(vy), z(vz) {}
};

static thread_local uint3_stub blockIdx{0, 0, 0};
static thread_local uint3_stub blockDim{1, 1, 1};
static thread_local uint3_stub threadIdx{0, 0, 0};

struct cudaDeviceProp {
    char name[256];
    int major;
    int minor;
    size_t totalGlobalMem;
};

inline cudaError_t cudaGetDeviceCount(int* count) {
    if (count) *count = 0;
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    if (prop) {
        std::memset(prop, 0, sizeof(cudaDeviceProp));
        std::strncpy(prop->name, "Host CPU Fallback Device", sizeof(prop->name) - 1);
        prop->major = 6;
        prop->minor = 1;
        prop->totalGlobalMem = 4096ULL * 1024 * 1024;
    }
    return cudaSuccess;
}

inline cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (devPtr) *devPtr = std::malloc(size);
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* devPtr) {
    if (devPtr) std::free(devPtr);
    return cudaSuccess;
}

inline cudaError_t cudaMallocHost(void** ptr, size_t size) {
    if (ptr) *ptr = std::malloc(size);
    return cudaSuccess;
}

inline cudaError_t cudaFreeHost(void* ptr) {
    if (ptr) std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    if (devPtr) std::memset(devPtr, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if (dst && src) std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = nullptr) {
    if (dst && src) std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) {
    if (pStream) *pStream = (cudaStream_t)1;
    return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return cudaSuccess;
}

inline cudaError_t cudaEventCreate(cudaEvent_t* event) {
    if (event) *event = (cudaEvent_t)1;
    return cudaSuccess;
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr) {
    return cudaSuccess;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return cudaSuccess;
}

inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (ms) *ms = 0.1f;
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

inline float3 make_float3(float x, float y, float z) {
    float3 v; v.x = x; v.y = y; v.z = z; return v;
}

inline int3 make_int3(int x, int y, int z) {
    int3 v; v.x = x; v.y = y; v.z = z; return v;
}

#define __global__
#define __device__
#define __host__
#define __restrict__
#define __shared__ static

#endif
