#ifndef SIMD_COMPONENT_BUFFERS_HPP
#define SIMD_COMPONENT_BUFFERS_HPP

#include <immintrin.h>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>

// Custom 32-Byte Aligned Allocator to ensure pointer memory alignment for AVX2 (_mm256_load_ps / _mm256_store_ps)
template <typename T, size_t Alignment = 32>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
#if defined(_MSC_VER) || defined(__MINGW32__)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, size_t) noexcept {
        if (!p) return;
#if defined(_MSC_VER) || defined(__MINGW32__)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

template <typename T>
struct alignas(32) SIMDAlignedVector {
    std::vector<T, AlignedAllocator<T, 32>> Data;

    void Resize(size_t Size) { Data.resize(Size); }
    size_t Size() const { return Data.size(); }
    T* data() { return Data.data(); }
    const T* data() const { return Data.data(); }
    T& operator[](size_t Index) { return Data[Index]; }
    const T& operator[](size_t Index) const { return Data[Index]; }
};

// 1. Fixed Gear Component Buffer (SoA)
struct FFixedGearComponentBuffer {
    SIMDAlignedVector<float> FixedRatio;   // [R_fixed_0, R_fixed_1, ...]
    SIMDAlignedVector<float> SystemInertia; // [Inertia_0, Inertia_1, ...]

    void Resize(size_t count) {
        FixedRatio.Resize(count);
        SystemInertia.Resize(count);
    }
};

// 2. Variable Gear Component Buffer (SoA)
struct FVariableGearComponentBuffer {
    SIMDAlignedVector<float> CurrentRatio; // [R_var_0, R_var_1, ...]
    SIMDAlignedVector<float> TargetRatio;  // [R_target_0, R_target_1, ...]
    SIMDAlignedVector<float> ShiftSpeed;   // [Speed_0, Speed_1, ...]
    SIMDAlignedVector<float> Friction;     // [Friction_0, Friction_1, ...]

    void Resize(size_t count) {
        CurrentRatio.Resize(count);
        TargetRatio.Resize(count);
        ShiftSpeed.Resize(count);
        Friction.Resize(count);
    }
};

// 3. Flywheel Component Buffer (SoA - Dynamic State & Torque)
struct FFlywheelComponentBuffer {
    SIMDAlignedVector<float> AngularVelocity; // [w_0, w_1, ...] (Vital Energy)
    SIMDAlignedVector<float> DampingFactor;   // [D_0, D_1, ...]
    SIMDAlignedVector<float> InputTorque;      // [T_in_0, T_in_1, ...]
    SIMDAlignedVector<float> BrakingTorque;    // [T_brake_0, T_brake_1, ...]
    SIMDAlignedVector<float> DerivedOutput;    // [Damage_0, Damage_1, ...]

    void Resize(size_t count) {
        AngularVelocity.Resize(count);
        DampingFactor.Resize(count);
        InputTorque.Resize(count);
        BrakingTorque.Resize(count);
        DerivedOutput.Resize(count);
    }
};

#endif // SIMD_COMPONENT_BUFFERS_HPP
