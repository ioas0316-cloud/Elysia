#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <atomic>
#include <stdexcept>

namespace elysia::memory {

template <typename T>
class VRAMRingBuffer {
public:
    VRAMRingBuffer(size_t capacity_per_slot, size_t num_slots = 3)
        : slot_capacity_(capacity_per_slot), num_slots_(num_slots), write_head_(0), read_head_(0) {

        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        d_slots_.resize(num_slots_, nullptr);

        for (size_t i = 0; i < num_slots_; ++i) {
            cudaMalloc(&d_slots_[i], slot_capacity_ * sizeof(T));
        }
    }

    ~VRAMRingBuffer() {
        for (auto ptr : d_slots_) {
            if (ptr) cudaFree(ptr);
        }
        if (stream_) cudaStreamDestroy(stream_);
    }

    // Host -> Device 비동기 Zero-Copy Push
    void PushAsync(const T* host_data, size_t count) {
        if (count > slot_capacity_) {
            throw std::runtime_error("Push count exceeds VRAM slot capacity.");
        }

        size_t current_slot = write_head_.fetch_add(1, std::memory_order_relaxed) % num_slots_;
        cudaMemcpyAsync(
            d_slots_[current_slot],
            host_data,
            count * sizeof(T),
            cudaMemcpyHostToDevice,
            stream_
        );
    }

    // 커널에 전달할 현재 읽기 슬롯 디바이스 포인터 획득
    const T* GetCurrentReadPtr() const {
        size_t current_slot = read_head_.load(std::memory_order_relaxed) % num_slots_;
        return d_slots_[current_slot];
    }

    // 프레임 동기화 및 슬롯 전진
    void AdvanceReadSlot() {
        read_head_.fetch_add(1, std::memory_order_relaxed);
    }

    cudaStream_t GetStream() const { return stream_; }

private:
    size_t slot_capacity_;
    size_t num_slots_;
    std::atomic<size_t> write_head_;
    std::atomic<size_t> read_head_;

    std::vector<T*> d_slots_;
    cudaStream_t stream_{nullptr};
};

} // namespace elysia::memory
