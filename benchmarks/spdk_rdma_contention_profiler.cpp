/*
 * SPDK NVMe-oF User-Space + 1-Sided RDMA CAS Lock Contention Profiler
 * Includes high-resolution profiling metrics and hardware-independent simulation fallback.
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cstring>
#include <cmath>

constexpr size_t PAGE_SIZE = 64 * 1024; // 64KB Phase Page

union GlobalDirectoryWord {
    uint64_t raw;
    struct {
        uint16_t version;   // ABA Prevention Counter
        uint16_t state;     // 0: FREE, 1: LOCKED
        uint32_t owner_node;// Node/Thread ID holding the lock
    } bits;
};

struct PerOpMetric {
    double cas_acquire_us;  // Latency to acquire lock via RDMA CAS
    double spdk_io_us;      // Latency to perform SPDK NVMe-oF Read
    double total_e2e_us;    // E2E Latency
    uint32_t retry_count;   // Number of CAS retries due to lock contention
};

// ----------------------------------------------------------------------------
// Simulated 1-Sided RDMA CAS Lock Manager
// ----------------------------------------------------------------------------
class RDMACASManagerSimulation {
private:
    std::atomic<uint64_t>* remote_dir_word;

public:
    RDMACASManagerSimulation(std::atomic<uint64_t>* target_addr) : remote_dir_word(target_addr) {}

    bool try_cas_lock(uint32_t thread_id, uint16_t version) {
        GlobalDirectoryWord expected_val{0};
        expected_val.bits.version = version;
        expected_val.bits.state = 0; // FREE

        GlobalDirectoryWord desired_val{0};
        desired_val.bits.version = version + 1;
        desired_val.bits.state = 1; // LOCKED
        desired_val.bits.owner_node = thread_id;

        uint64_t expected_raw = expected_val.raw;
        // Hardware 1-Sided Atomic CAS Simulation (~800ns delay)
        std::this_thread::sleep_for(std::chrono::nanoseconds(100));

        return remote_dir_word->compare_exchange_strong(expected_raw, desired_val.raw);
    }

    void release_lock(uint16_t version) {
        GlobalDirectoryWord free_val{0};
        free_val.bits.version = version + 1;
        free_val.bits.state = 0; // FREE

        remote_dir_word->store(free_val.raw, std::memory_order_release);
    }
};

// ----------------------------------------------------------------------------
// SPDK NVMe-oF User-Space Engine Simulation
// ----------------------------------------------------------------------------
class SPDKEngineSimulation {
public:
    bool execute_user_space_read(void* buffer, uint64_t lba) {
        // User-Space PMD Polling Read Simulation (~4.5us)
        std::this_thread::sleep_for(std::chrono::nanoseconds(4500));
        return true;
    }
};

// ----------------------------------------------------------------------------
// Integrated Contention Profiler Engine
// ----------------------------------------------------------------------------
class IntegratedContentionProfiler {
private:
    std::atomic<uint64_t> mock_remote_directory_word{0};
    std::atomic<bool> start_flag{false};

public:
    IntegratedContentionProfiler() {
        mock_remote_directory_word.store(0);
    }

    void worker_routine(int thread_id, int num_ops, SPDKEngineSimulation* spdk,
                        std::vector<PerOpMetric>& thread_metrics)
    {
        RDMACASManagerSimulation rdma(&mock_remote_directory_word);
        std::vector<uint8_t> vram_buffer(PAGE_SIZE);

        while (!start_flag.load(std::memory_order_relaxed)) {
            std::this_thread::yield();
        }

        for (int i = 0; i < num_ops; ++i) {
            PerOpMetric metric{};
            uint32_t retries = 0;
            uint16_t version = 0;

            auto t_start = std::chrono::high_resolution_clock::now();
            auto t_cas_start = t_start;

            // Step 1: 1-Sided RDMA CAS Lock Acquisition with Exponential Backoff
            while (!rdma.try_cas_lock(thread_id, version)) {
                retries++;
                version++;

                // Truncated Exponential Backoff
                uint32_t backoff_ns = std::min(10000u, (1u << std::min(retries, 10u)) * 50);
                auto b_start = std::chrono::high_resolution_clock::now();
                while (std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::high_resolution_clock::now() - b_start).count() < backoff_ns) {
                    std::this_thread::yield();
                }
            }

            auto t_cas_end = std::chrono::high_resolution_clock::now();

            // Step 2: Execute SPDK NVMe-oF User-Space Read
            auto t_spdk_start = t_cas_end;
            spdk->execute_user_space_read(vram_buffer.data(), 0);
            auto t_spdk_end = std::chrono::high_resolution_clock::now();

            // Step 3: Release Lock via 1-Sided RDMA Write
            rdma.release_lock(version);

            // Record Metrics
            metric.cas_acquire_us = std::chrono::duration<double, std::micro>(t_cas_end - t_cas_start).count();
            metric.spdk_io_us = std::chrono::duration<double, std::micro>(t_spdk_end - t_spdk_start).count();
            metric.total_e2e_us = std::chrono::duration<double, std::micro>(t_spdk_end - t_start).count();
            metric.retry_count = retries;

            thread_metrics.push_back(metric);
        }
    }

    void run_contention_benchmark(int num_threads, int ops_per_thread) {
        std::cout << "\n=========================================================================" << std::endl;
        std::cout << "  SPDK NVMe-oF + 1-Sided RDMA CAS Lock Contention Profiler" << std::endl;
        std::cout << "  Concurrency Level: " << num_threads << " Threads | Total Ops: " << (num_threads * ops_per_thread) << std::endl;
        std::cout << "=========================================================================\n" << std::endl;

        SPDKEngineSimulation spdk_engine;

        std::vector<std::thread> workers;
        std::vector<std::vector<PerOpMetric>> all_metrics(num_threads);

        start_flag.store(false);

        for (int i = 0; i < num_threads; ++i) {
            all_metrics[i].reserve(ops_per_thread);
            workers.emplace_back(&IntegratedContentionProfiler::worker_routine, this,
                                 i + 1, ops_per_thread, &spdk_engine, std::ref(all_metrics[i]));
        }

        // Start Benchmark Synchronously
        auto t_bench_start = std::chrono::high_resolution_clock::now();
        start_flag.store(true);

        for (auto& t : workers) {
            t.join();
        }
        auto t_bench_end = std::chrono::high_resolution_clock::now();

        double total_time_sec = std::chrono::duration<double>(t_bench_end - t_bench_start).count();
        analyze_and_print_results(all_metrics, total_time_sec);
    }

private:
    void analyze_and_print_results(const std::vector<std::vector<PerOpMetric>>& all_metrics, double total_time_sec) {
        std::vector<double> cas_latencies;
        std::vector<double> spdk_latencies;
        std::vector<double> e2e_latencies;
        uint64_t total_retries = 0;
        size_t total_ops = 0;

        for (const auto& tm : all_metrics) {
            for (const auto& m : tm) {
                cas_latencies.push_back(m.cas_acquire_us);
                spdk_latencies.push_back(m.spdk_io_us);
                e2e_latencies.push_back(m.total_e2e_us);
                total_retries += m.retry_count;
                total_ops++;
            }
        }

        std::sort(cas_latencies.begin(), cas_latencies.end());
        std::sort(spdk_latencies.begin(), spdk_latencies.end());
        std::sort(e2e_latencies.begin(), e2e_latencies.end());

        auto get_percentile = [](const std::vector<double>& sorted, double p) {
            if (sorted.empty()) return 0.0;
            size_t idx = static_cast<size_t>(p * sorted.size());
            return sorted[std::min(idx, sorted.size() - 1)];
        };

        double avg_retries = static_cast<double>(total_retries) / total_ops;
        double ops_per_sec = total_ops / total_time_sec;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[ THROUGHPUT & CONCURRENCY METRICS ]" << std::endl;
        std::cout << "  - Total Execution Time : " << total_time_sec * 1000.0 << " ms" << std::endl;
        std::cout << "  - Aggregated Throughput: " << ops_per_sec << " IOPS" << std::endl;
        std::cout << "  - Avg Retries per Lock : " << avg_retries << " retries/op" << std::endl;
        std::cout << "  - Total Lock Conflicts : " << total_retries << " collisions" << std::endl;
        std::cout << "\n[ LATENCY PROFILE BREAKDOWN (us) ]" << std::endl;
        std::cout << "  -------------------------------------------------------------------" << std::endl;
        std::cout << "  Metric Domain     |   P50 (Median)   |     P90      |     P99      |     MAX" << std::endl;
        std::cout << "  -------------------------------------------------------------------" << std::endl;
        std::cout << "  1. RDMA CAS Lock  | " << std::setw(12) << get_percentile(cas_latencies, 0.50)
                  << " | " << std::setw(12) << get_percentile(cas_latencies, 0.90)
                  << " | " << std::setw(12) << get_percentile(cas_latencies, 0.99)
                  << " | " << std::setw(8) << cas_latencies.back() << std::endl;

        std::cout << "  2. SPDK NVMe Read | " << std::setw(12) << get_percentile(spdk_latencies, 0.50)
                  << " | " << std::setw(12) << get_percentile(spdk_latencies, 0.90)
                  << " | " << std::setw(12) << get_percentile(spdk_latencies, 0.99)
                  << " | " << std::setw(8) << spdk_latencies.back() << std::endl;

        std::cout << "  3. End-to-End     | " << std::setw(12) << get_percentile(e2e_latencies, 0.50)
                  << " | " << std::setw(12) << get_percentile(e2e_latencies, 0.90)
                  << " | " << std::setw(12) << get_percentile(e2e_latencies, 0.99)
                  << " | " << std::setw(8) << e2e_latencies.back() << std::endl;
        std::cout << "  -------------------------------------------------------------------\n" << std::endl;
    }
};

int main() {
    IntegratedContentionProfiler profiler;

    // Run Profile across varying levels of Thread Contention
    std::vector<int> contention_levels = {1, 4, 16};
    int ops_per_thread = 100;

    for (int threads : contention_levels) {
        profiler.run_contention_benchmark(threads, ops_per_thread);
    }

    return 0;
}
