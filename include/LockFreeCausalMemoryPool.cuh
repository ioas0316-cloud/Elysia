#pragma once

#include <iostream>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#else
#include "elysia/cuda_host_stub.h"
#endif

struct CausalMemoryPoolDeviceHandle {
    float2*  rotor_pool;        // 위상 로터 버퍼 (전체 수용량)
    float*   curvature_pool;    // 곡률 버퍼
    int*     free_index_stack;  // 해제된 노드 슬롯 오프셋 스택
    int*     stack_top;         // 원자적 스택 Top 인덱스 (Lock-Free)
    int*     allocated_count;   // 현재 할당된 총 노드 수
    int      max_capacity;
};

class LockFreeCausalMemoryPool {
private:
    CausalMemoryPoolDeviceHandle h_handle_;
    CausalMemoryPoolDeviceHandle* d_handle_ptr_{nullptr};

public:
    LockFreeCausalMemoryPool(int max_capacity) {
        h_handle_.max_capacity = max_capacity;

        cudaMalloc(reinterpret_cast<void**>(&h_handle_.rotor_pool), max_capacity * sizeof(float2));
        cudaMalloc(reinterpret_cast<void**>(&h_handle_.curvature_pool), max_capacity * sizeof(float));
        cudaMalloc(reinterpret_cast<void**>(&h_handle_.free_index_stack), max_capacity * sizeof(int));
        cudaMalloc(reinterpret_cast<void**>(&h_handle_.stack_top), sizeof(int));
        cudaMalloc(reinterpret_cast<void**>(&h_handle_.allocated_count), sizeof(int));

        int* h_stack = new int[max_capacity];
        for (int i = 0; i < max_capacity; ++i) {
            h_stack[i] = max_capacity - 1 - i;
        }
        cudaMemcpy(h_handle_.free_index_stack, h_stack, max_capacity * sizeof(int), cudaMemcpyHostToDevice);
        delete[] h_stack;

        int init_top = max_capacity - 1;
        int init_alloc = 0;
        cudaMemcpy(h_handle_.stack_top, &init_top, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(h_handle_.allocated_count, &init_alloc, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(reinterpret_cast<void**>(&d_handle_ptr_), sizeof(CausalMemoryPoolDeviceHandle));
        cudaMemcpy(d_handle_ptr_, &h_handle_, sizeof(CausalMemoryPoolDeviceHandle), cudaMemcpyHostToDevice);
    }

    ~LockFreeCausalMemoryPool() {
        if (h_handle_.rotor_pool) cudaFree(h_handle_.rotor_pool);
        if (h_handle_.curvature_pool) cudaFree(h_handle_.curvature_pool);
        if (h_handle_.free_index_stack) cudaFree(h_handle_.free_index_stack);
        if (h_handle_.stack_top) cudaFree(h_handle_.stack_top);
        if (h_handle_.allocated_count) cudaFree(h_handle_.allocated_count);
        if (d_handle_ptr_) cudaFree(d_handle_ptr_);
    }

    CausalMemoryPoolDeviceHandle* get_device_handle() { return d_handle_ptr_; }
};

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ inline int gpu_causal_node_alloc(CausalMemoryPoolDeviceHandle* pool) {
    int stack_idx = atomicSub(pool->stack_top, 1);
    if (stack_idx < 0) {
        atomicAdd(pool->stack_top, 1);
        return -1;
    }
    atomicAdd(pool->allocated_count, 1);
    return pool->free_index_stack[stack_idx];
}

__device__ inline void gpu_causal_node_free(CausalMemoryPoolDeviceHandle* pool, int node_idx) {
    if (node_idx < 0 || node_idx >= pool->max_capacity) return;

    int stack_idx = atomicAdd(pool->stack_top, 1) + 1;
    pool->free_index_stack[stack_idx] = node_idx;
    atomicSub(pool->allocated_count, 1);
}
#endif
