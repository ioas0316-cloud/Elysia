#include "spatiotemporal_memory.hpp"

namespace elysia::emulation {

StaticRotorUnit::StaticRotorUnit(size_t max_tag_capacity) : capacity_(max_tag_capacity) {}

StaticRotorUnit::~StaticRotorUnit() = default;

Rotor StaticRotorUnit::pin_evicted_phase(uint64_t cache_line_id, const Rotor& r_curr, const Rotor& r_base) {
    // Delta Rotor calculation: \Delta R = R_{curr} * \tilde{R}_{base}
    Rotor r_base_rev = r_base.reverse();
    Rotor delta_r = r_curr.multiply(r_base_rev);

    RotorTagEntry entry;
    entry.delta_r = delta_r;
    entry.bg_drift_accum = 0.0f;
    entry.is_pinned = true;

    tag_array_[cache_line_id] = entry;
    return delta_r;
}

Rotor StaticRotorUnit::restore_active_phase(uint64_t cache_line_id, const Rotor& r_global_now) {
    auto it = tag_array_.find(cache_line_id);
    if (it == tag_array_.end()) {
        // Fallback: return global rotor as-is
        return r_global_now;
    }

    RotorTagEntry& entry = it->second;
    // R_{active} = R_{global} * \Delta R
    Rotor restored = r_global_now.multiply(entry.delta_r);

    // Apply accumulated background drift phase adjustment if present
    if (entry.bg_drift_accum != 0.0f) {
        float half_angle = entry.bg_drift_accum * 0.5f;
        Rotor drift_rotor{ std::cos(half_angle), std::sin(half_angle), 0.0f, 0.0f };
        restored = restored.multiply(drift_rotor);
    }

    return restored;
}

void StaticRotorUnit::update_background_drift(float delta_t, float omega_bg) {
    for (auto& pair : tag_array_) {
        if (pair.second.is_pinned) {
            pair.second.bg_drift_accum += omega_bg * delta_t;
        }
    }
}

size_t StaticRotorUnit::get_pinned_count() const {
    size_t count = 0;
    for (const auto& pair : tag_array_) {
        if (pair.second.is_pinned) count++;
    }
    return count;
}

} // namespace elysia::emulation
