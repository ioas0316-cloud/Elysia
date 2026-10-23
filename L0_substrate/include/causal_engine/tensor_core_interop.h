#ifndef CAUSAL_ENGINE_TENSOR_CORE_INTEROP_H
#define CAUSAL_ENGINE_TENSOR_CORE_INTEROP_H

#include <cstdint>
#include <cstddef>

namespace causal_engine {

// Conceptual WMMA / INT4 Tensor Core and Graphics API Zero-Copy Interop Spec
struct TensorCoreInteropConfig {
    static constexpr size_t WMMA_TILE_M = 16;
    static constexpr size_t WMMA_TILE_N = 16;
    static constexpr size_t WMMA_TILE_K_FP16 = 16;
    static constexpr size_t WMMA_TILE_K_INT4 = 64;

    bool is_zero_copy_vram_mapped;
    void* external_graphics_handle;
    void* mapped_vram_ptr;

    TensorCoreInteropConfig()
        : is_zero_copy_vram_mapped(false),
          external_graphics_handle(nullptr),
          mapped_vram_ptr(nullptr) {}
};

} // namespace causal_engine

#endif // CAUSAL_ENGINE_TENSOR_CORE_INTEROP_H
