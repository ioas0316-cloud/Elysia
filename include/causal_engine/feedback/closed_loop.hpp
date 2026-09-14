#ifndef CAUSAL_ENGINE_FEEDBACK_CLOSED_LOOP_HPP
#define CAUSAL_ENGINE_FEEDBACK_CLOSED_LOOP_HPP

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstdint>
#include <cstring>

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#endif

namespace causal_engine {
namespace feedback {

// =========================================================================
// 1. Telemetry Snapshot Data Structure (Trivially Copyable POD)
// =========================================================================
struct alignas(64) TelemetrySnapshot {
    double cpu_cache_miss_rate = 0.05;
    uint32_t gpu_temp_celsius = 45;
    bool gpu_is_throttled = false;
    uint32_t gpu_clock_mhz = 1500;
    float frame_time_ms = 14.0f;
    uint64_t sample_timestamp_ns = 0;
};

// =========================================================================
// 2. Performance & Hardware Telemetry Samplers (With S/W Fallback)
// =========================================================================
class CpuPerfSampler {
private:
    int fd_ref_ = -1;
    int fd_miss_ = -1;
    uint64_t prev_ref_ = 0;
    uint64_t prev_miss_ = 0;
    bool available_ = false;

public:
    CpuPerfSampler() {
#if defined(__linux__) && defined(__NR_perf_event_open)
        struct perf_event_attr pe;
        std::memset(&pe, 0, sizeof(pe));
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(pe);
        pe.disabled = 1;
        pe.exclude_kernel = 1;

        pe.config = PERF_COUNT_HW_CACHE_REFERENCES;
        fd_ref_ = static_cast<int>(syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0));

        pe.config = PERF_COUNT_HW_CACHE_MISSES;
        fd_miss_ = static_cast<int>(syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0));

        if (fd_ref_ != -1 && fd_miss_ != -1) {
            ioctl(fd_ref_, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd_miss_, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd_ref_, PERF_EVENT_IOC_ENABLE, 0);
            ioctl(fd_miss_, PERF_EVENT_IOC_ENABLE, 0);
            available_ = true;
        }
#endif
    }

    ~CpuPerfSampler() {
#if defined(__linux__)
        if (fd_ref_ != -1) close(fd_ref_);
        if (fd_miss_ != -1) close(fd_miss_);
#endif
    }

    bool is_available() const { return available_; }

    double sample_miss_rate() {
#if defined(__linux__)
        if (!available_) return 0.05;

        uint64_t curr_ref = 0, curr_miss = 0;
        if (read(fd_ref_, &curr_ref, sizeof(uint64_t)) <= 0 ||
            read(fd_miss_, &curr_miss, sizeof(uint64_t)) <= 0) {
            return 0.05;
        }

        uint64_t delta_ref = curr_ref - prev_ref_;
        uint64_t delta_miss = curr_miss - prev_miss_;
        prev_ref_ = curr_ref;
        prev_miss_ = curr_miss;

        if (delta_ref == 0) return 0.0;
        return static_cast<double>(delta_miss) / static_cast<double>(delta_ref);
#else
        return 0.05;
#endif
    }
};

struct GpuMetrics {
    uint32_t temp_celsius = 45;
    uint32_t clock_mhz = 1500;
    bool is_throttled = false;
};

class GpuNvmlSampler {
private:
    bool initialized_ = false;

public:
    GpuNvmlSampler() {
        // NVML initialization can be dynamic or mocked in S/W fallback mode
        initialized_ = false;
    }

    bool is_available() const { return initialized_; }

    GpuMetrics sample() {
        GpuMetrics m;
        m.temp_celsius = 45;
        m.clock_mhz = 1500;
        m.is_throttled = false;
        return m;
    }
};

class EnvironmentObserver {
private:
    CpuPerfSampler cpu_sampler_;
    GpuNvmlSampler gpu_sampler_;

public:
    bool init() {
        return cpu_sampler_.is_available() || gpu_sampler_.is_available();
    }

    TelemetrySnapshot poll(float frame_delta_ms, bool force_mock_throttle = false) {
        TelemetrySnapshot snapshot;
        snapshot.cpu_cache_miss_rate = cpu_sampler_.sample_miss_rate();
        GpuMetrics g = gpu_sampler_.sample();

        if (force_mock_throttle) {
            snapshot.gpu_temp_celsius = 85;
            snapshot.gpu_is_throttled = true;
            snapshot.gpu_clock_mhz = 900;
            snapshot.cpu_cache_miss_rate = 0.20;
        } else {
            snapshot.gpu_temp_celsius = g.temp_celsius;
            snapshot.gpu_is_throttled = g.is_throttled;
            snapshot.gpu_clock_mhz = g.clock_mhz;
        }

        snapshot.frame_time_ms = frame_delta_ms;
        snapshot.sample_timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        return snapshot;
    }
};

// =========================================================================
// 3. Lock-Free Async Telemetry Collector (Double-Buffered Atomic Swapper)
// =========================================================================
class AsyncTelemetryCollector {
private:
    TelemetrySnapshot buffers_[2];
    std::atomic<TelemetrySnapshot*> active_snapshot_{nullptr};
    std::atomic<bool> running_{false};
    std::thread worker_thread_;
    EnvironmentObserver observer_;
    std::atomic<bool> mock_throttle_{false};

public:
    AsyncTelemetryCollector() {
        buffers_[0] = TelemetrySnapshot();
        buffers_[1] = TelemetrySnapshot();
        active_snapshot_.store(&buffers_[0], std::memory_order_relaxed);
        observer_.init();
    }

    ~AsyncTelemetryCollector() {
        stop();
    }

    void start(uint32_t sampling_interval_ms = 20) {
        if (running_.exchange(true, std::memory_order_relaxed)) return;
        worker_thread_ = std::thread(&AsyncTelemetryCollector::telemetry_loop, this, sampling_interval_ms);
    }

    void stop() {
        if (running_.exchange(false, std::memory_order_relaxed)) {
            if (worker_thread_.joinable()) {
                worker_thread_.join();
            }
        }
    }

    void inject_mock_throttle(bool enable) {
        mock_throttle_.store(enable, std::memory_order_relaxed);
    }

    TelemetrySnapshot get_latest_snapshot() const noexcept {
        return *active_snapshot_.load(std::memory_order_acquire);
    }

private:
    void telemetry_loop(uint32_t interval_ms) {
        int write_index = 1;
        while (running_.load(std::memory_order_relaxed)) {
            bool throttle = mock_throttle_.load(std::memory_order_relaxed);
            TelemetrySnapshot sample = observer_.poll(14.0f, throttle);

            buffers_[write_index] = sample;
            active_snapshot_.store(&buffers_[write_index], std::memory_order_release);
            write_index = 1 - write_index;

            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        }
    }
};

// =========================================================================
// 4. Closed-Loop Governor Engine
// =========================================================================
enum class SimdMode {
    AVX2_256 = 8,
    SSE_128  = 4,
    Scalar   = 1
};

struct FrameExecutionProfile {
    SimdMode simd_width = SimdMode::AVX2_256;
    int causal_lod_step = 1;       // 1: 100%, 2: 50% interpolation, 4: 25%
    size_t chunk_size = 1024;
};

class RealtimeGovernor {
private:
    float target_frame_ms_;

public:
    explicit RealtimeGovernor(float target_frame_ms = 16.67f)
        : target_frame_ms_(target_frame_ms) {}

    FrameExecutionProfile evaluate(const TelemetrySnapshot& telemetry) const {
        FrameExecutionProfile profile;

        // Rule 1: GPU Thermal or Throttle -> reduce SIMD lane width to reduce power/heat
        if (telemetry.gpu_is_throttled || telemetry.gpu_temp_celsius > 80) {
            profile.simd_width = SimdMode::SSE_128;
        } else {
            profile.simd_width = SimdMode::AVX2_256;
        }

        // Rule 2: High CPU Cache Miss Rate -> scale down batch chunk size
        if (telemetry.cpu_cache_miss_rate > 0.15) {
            profile.chunk_size = 256;
        } else {
            profile.chunk_size = 1024;
        }

        // Rule 3: Frame Time Overshoot -> drop Causal LOD step resolution
        if (telemetry.frame_time_ms > target_frame_ms_ * 1.3f) {
            profile.causal_lod_step = 4;
        } else if (telemetry.frame_time_ms > target_frame_ms_) {
            profile.causal_lod_step = 2;
        } else {
            profile.causal_lod_step = 1;
        }

        return profile;
    }
};

} // namespace feedback
} // namespace causal_engine

#endif // CAUSAL_ENGINE_FEEDBACK_CLOSED_LOOP_HPP
