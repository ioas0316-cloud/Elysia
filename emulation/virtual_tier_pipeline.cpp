#include "spatiotemporal_memory.hpp"
#include <iostream>

namespace elysia::emulation {

VirtualTierPipeline::VirtualTierPipeline() = default;
VirtualTierPipeline::~VirtualTierPipeline() = default;

void VirtualTierPipeline::trigger_dma_ssd_inscription(uint32_t chart_id, const MetricTensor3x3* metric_data) {
    (void)chart_id;
    (void)metric_data;
    stats_.ssd_dma_transfers++;
}

void VirtualTierPipeline::simulate_eviction_fetch_cycle(
    StaticRotorUnit& sru, uint64_t cache_line_id, const Rotor& r_curr, const Rotor& r_base) {

    stats_.cache_misses++;
    // Pin evicted phase in SRU
    sru.pin_evicted_phase(cache_line_id, r_curr, r_base);

    stats_.ram_accesses++;
    // Restore active phase
    sru.restore_active_phase(cache_line_id, r_curr);
}

} // namespace elysia::emulation
