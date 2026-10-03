#ifndef ELYSIA_HOLOGRAPHIC_ENGINE_HPP
#define ELYSIA_HOLOGRAPHIC_ENGINE_HPP

#include <cstdint>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <fcntl.h>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#else
#include <elysia/cuda_host_stub.h>
#endif

namespace elysia {

// ----------------------------------------------------------------------------
// 1. 3D Morton Z-Order (Space-Filling Curve) Indexing
// ----------------------------------------------------------------------------
__device__ __host__ inline uint64_t splitBy3(uint32_t a) {
    uint64_t res = 0;
    a &= 0x3ff;
    for (int i = 0; i < 10; ++i) {
        if ((a >> i) & 1) {
            res |= (1ULL << (i * 3));
        }
    }
    return res;
}

__device__ __host__ inline uint64_t encodeMorton3D(uint32_t x, uint32_t y, uint32_t z) {
    return (splitBy3(z) << 2) | (splitBy3(y) << 1) | splitBy3(x);
}

__device__ __host__ inline uint32_t compactBy3(uint64_t x) {
    uint32_t res = 0;
    for (int i = 0; i < 10; ++i) {
        if ((x >> (i * 3)) & 1ULL) {
            res |= (1U << i);
        }
    }
    return res;
}

__device__ __host__ inline void decodeMorton3D(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z) {
    x = compactBy3(code);
    y = compactBy3(code >> 1);
    z = compactBy3(code >> 2);
}

// ----------------------------------------------------------------------------
// 2. 128-Byte Aligned Zero-Copy Topological Header (POD)
// ----------------------------------------------------------------------------
struct alignas(128) TopologicalHeader {
    uint32_t magic_number;     // 0x454C5953 ('ELYS')
    uint32_t dimension;        // Tensor dimensions (e.g. 3D = 3, 4D = 4)
    uint32_t grid_dim[4];      // [X, Y, Z, W] resolution
    float    spatial_step_dx;  // Topological grid step size
    uint64_t payload_bytes;    // Continuous payload byte length
    uint8_t  padding[88];      // 128-byte cache line padding
};

static_assert(sizeof(TopologicalHeader) == 128, "TopologicalHeader must be exactly 128 bytes");

// ----------------------------------------------------------------------------
// 3. Hangul Jamo 2x2 Spin Matrix & 3-Way Kronecker Tensor Encoder
// ----------------------------------------------------------------------------
struct Matrix2x2 {
    uint8_t m[2][2];
};

__host__ __device__ inline uint64_t encodeHangulKronecker(uint32_t utf32_char) {
    if (utf32_char < 0xAC00 || utf32_char > 0xD7A3) return 0ULL;

    uint32_t base = utf32_char - 0xAC00;
    uint32_t cho_idx  = base / 588;          // Cho (0~18)
    uint32_t jung_idx = (base % 588) / 28;   // Jung (0~20)
    uint32_t jong_idx = base % 28;           // Jong (0~27)

    uint8_t J1 = (cho_idx & 0xF) | 0x1;
    uint8_t J2 = (jung_idx & 0xF) | 0x1;
    uint8_t J3 = (jong_idx & 0xF) | 0x1;

    uint64_t syllable_tensor = 0ULL;

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            uint8_t b1 = (J1 >> ((r & 1) * 2 + (c & 1))) & 1;
            uint8_t b2 = (J2 >> (((r >> 1) & 1) * 2 + ((c >> 1) & 1))) & 1;
            uint8_t b3 = (J3 >> (((r >> 2) & 1) * 2 + ((c >> 2) & 1))) & 1;

            if (b1 & b2 & b3) {
                syllable_tensor |= (1ULL << (r * 8 + c));
            }
        }
    }
    return syllable_tensor;
}

__host__ __device__ inline int computeTopologicalBitDistance(uint64_t tensor1, uint64_t tensor2) {
    uint64_t diff = tensor1 ^ tensor2;
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    return __popcll(diff);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(diff);
#else
    int count = 0;
    while (diff) {
        count += diff & 1;
        diff >>= 1;
    }
    return count;
#endif
}

// ----------------------------------------------------------------------------
// 4. Zero-Copy Topological Buffer Class
// ----------------------------------------------------------------------------
class ZeroCopyTopologicalBuffer {
public:
    TopologicalHeader* header;
    float*             tensor_data;
    size_t             total_size;
    void*              mapped_ptr;
    bool               is_cuda_registered;

    ZeroCopyTopologicalBuffer()
        : header(nullptr), tensor_data(nullptr), total_size(0),
          mapped_ptr(nullptr), is_cuda_registered(false) {}

    static ZeroCopyTopologicalBuffer* MapFromFile(const char* filepath) {
#if defined(__unix__) || defined(__APPLE__)
        int fd = open(filepath, O_RDWR);
        if (fd == -1) return nullptr;

        size_t file_len = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);

        if (file_len < sizeof(TopologicalHeader)) {
            close(fd);
            return nullptr;
        }

        void* ptr = mmap(NULL, file_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);

        if (ptr == MAP_FAILED) return nullptr;

        auto* buffer = new ZeroCopyTopologicalBuffer();
        buffer->total_size = file_len;
        buffer->mapped_ptr = ptr;
        buffer->header = reinterpret_cast<TopologicalHeader*>(ptr);

        if (buffer->header->magic_number != 0x454C5953) {
            munmap(ptr, file_len);
            delete buffer;
            return nullptr;
        }

        uint8_t* byte_ptr = static_cast<uint8_t*>(ptr);
        buffer->tensor_data = reinterpret_cast<float*>(byte_ptr + sizeof(TopologicalHeader));
        return buffer;
#else
        // Fallback for non-mmap environments
        return nullptr;
#endif
    }

    bool RegisterCudaZeroCopy() {
        if (!mapped_ptr || is_cuda_registered) return false;
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
        cudaError_t err = cudaHostRegister(mapped_ptr, total_size, cudaHostRegisterMapped);
        if (err == cudaSuccess) {
            is_cuda_registered = true;
            return true;
        }
#endif
        return false;
    }

    ~ZeroCopyTopologicalBuffer() {
        if (mapped_ptr) {
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
            if (is_cuda_registered) {
                cudaHostUnregister(mapped_ptr);
            }
#endif
#if defined(__unix__) || defined(__APPLE__)
            munmap(mapped_ptr, total_size);
#endif
        }
    }
};

} // namespace elysia

#endif // ELYSIA_HOLOGRAPHIC_ENGINE_HPP
