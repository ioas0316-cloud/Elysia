#include "spatiotemporal_memory.hpp"
#include <cmath>
#include <iostream>

namespace elysia::emulation {

AtlasManager::AtlasManager(size_t initial_chart_count) {
    charts_.reserve(initial_chart_count);

    // Grid layout for initial charts across unit cube [0, 1]^3
    size_t side = static_cast<size_t>(std::ceil(std::cbrt(initial_chart_count)));
    if (side < 1) side = 1;
    float step = 1.0f / side;
    float radius = step * 0.4f;

    size_t count = 0;
    for (size_t x = 0; x < side && count < initial_chart_count; ++x) {
        for (size_t y = 0; y < side && count < initial_chart_count; ++y) {
            for (size_t z = 0; z < side && count < initial_chart_count; ++z) {
                uint32_t cid = generate_next_chart_id();
                LocalChart chart;
                chart.chart_id = cid;
                chart.center[0] = (x + 0.5f) * step;
                chart.center[1] = (y + 0.5f) * step;
                chart.center[2] = (z + 0.5f) * step;
                chart.radius = radius;
                chart.is_active = true;

                charts_.push_back(chart);

                ScaledLocalChartNode node;
                node.chart_data = chart;
                node.parent_id = 0xFFFFFFFF;
                node.is_leaf = true;
                node.depth = 0;
                chart_tree_[cid] = node;

                count++;
            }
        }
    }
}

AtlasManager::~AtlasManager() = default;

uint32_t AtlasManager::generate_next_chart_id() {
    return next_chart_id_++;
}

std::vector<uint32_t> AtlasManager::query_active_charts(const float pos[3]) {
    std::vector<uint32_t> active_ids;

    for (const auto& pair : chart_tree_) {
        const auto& node = pair.second;
        if (!node.chart_data.is_active || !node.is_leaf) continue;

        float dx = pos[0] - node.chart_data.center[0];
        float dy = pos[1] - node.chart_data.center[1];
        float dz = pos[2] - node.chart_data.center[2];
        float dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq <= node.chart_data.radius * node.chart_data.radius) {
            active_ids.push_back(node.chart_data.chart_id);
        }
    }
    return active_ids;
}

Rotor AtlasManager::compute_chart_transition_rotor(uint32_t src_chart_id, uint32_t dst_chart_id) {
    float angle = static_cast<float>(dst_chart_id - src_chart_id) * 0.1f;
    float half_angle = angle * 0.5f;
    return { std::cos(half_angle), std::sin(half_angle), 0.0f, 0.0f };
}

void AtlasManager::update_chart_active_states(const std::vector<uint32_t>& active_ids) {
    for (auto& chart : charts_) {
        bool found = (std::find(active_ids.begin(), active_ids.end(), chart.chart_id) != active_ids.end());
        chart.is_active = found;
        if (chart_tree_.count(chart.chart_id)) {
            chart_tree_[chart.chart_id].chart_data.is_active = found;
        }
    }
}

const std::vector<LocalChart>& AtlasManager::get_all_charts() const {
    return charts_;
}

float AtlasManager::compute_max_bottleneck_in_chart(
    const LocalChart& chart, const float* bottleneck_map,
    size_t dim_x, size_t dim_y, size_t dim_z) {

    float max_p = 0.0f;

    int min_x = std::max(0, static_cast<int>((chart.center[0] - chart.radius) * dim_x));
    int max_x = std::min(static_cast<int>(dim_x) - 1, static_cast<int>((chart.center[0] + chart.radius) * dim_x));
    int min_y = std::max(0, static_cast<int>((chart.center[1] - chart.radius) * dim_y));
    int max_y = std::min(static_cast<int>(dim_y) - 1, static_cast<int>((chart.center[1] + chart.radius) * dim_y));
    int min_z = std::max(0, static_cast<int>((chart.center[2] - chart.radius) * dim_z));
    int max_z = std::min(static_cast<int>(dim_z) - 1, static_cast<int>((chart.center[2] + chart.radius) * dim_z));

    for (int z = min_z; z <= max_z; ++z) {
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                size_t idx = x + y * dim_x + z * dim_x * dim_y;
                max_p = std::max(max_p, bottleneck_map[idx]);
            }
        }
    }
    return max_p;
}

std::vector<AtlasManager::ChartScaleTrigger> AtlasManager::evaluate_chart_bottlenecks(
    const float* host_bottleneck_map, float split_threshold, float merge_threshold,
    size_t dim_x, size_t dim_y, size_t dim_z) {

    std::vector<ChartScaleTrigger> triggers;
    std::unordered_map<uint32_t, std::vector<uint32_t>> parent_to_children_map;

    for (const auto& pair : chart_tree_) {
        const auto& node = pair.second;
        if (!node.chart_data.is_active || !node.is_leaf) continue;

        float max_p = compute_max_bottleneck_in_chart(
            node.chart_data, host_bottleneck_map, dim_x, dim_y, dim_z);

        ChartScaleTrigger trigger;
        trigger.chart_id = node.chart_data.chart_id;
        trigger.max_bottleneck_pressure = max_p;
        trigger.requires_subdivision = (max_p > split_threshold) && (node.depth < max_depth_);
        trigger.requires_merge = false;

        triggers.push_back(trigger);

        if (node.parent_id != 0xFFFFFFFF && max_p < merge_threshold) {
            parent_to_children_map[node.parent_id].push_back(node.chart_data.chart_id);
        }
    }

    for (auto& trigger : triggers) {
        auto it = chart_tree_.find(trigger.chart_id);
        if (it != chart_tree_.end()) {
            uint32_t p_id = it->second.parent_id;
            if (p_id != 0xFFFFFFFF && parent_to_children_map[p_id].size() == 8) {
                trigger.requires_merge = true;
            }
        }
    }

    return triggers;
}

void AtlasManager::subdivide_chart(uint32_t parent_chart_id) {
    auto it = chart_tree_.find(parent_chart_id);
    if (it == chart_tree_.end() || !it->second.is_leaf) return;

    ScaledLocalChartNode& parent = it->second;
    parent.is_leaf = false;
    parent.chart_data.is_active = false;

    float parent_r = parent.chart_data.radius;
    float child_r = parent_r * 0.5f;
    float offset = child_r;

    const float offsets[8][3] = {
        {-offset, -offset, -offset}, { offset, -offset, -offset},
        {-offset,  offset, -offset}, { offset,  offset, -offset},
        {-offset, -offset,  offset}, { offset, -offset,  offset},
        {-offset,  offset,  offset}, { offset,  offset,  offset}
    };

    for (int i = 0; i < 8; ++i) {
        uint32_t child_id = generate_next_chart_id();
        parent.children_ids[i] = child_id;

        ScaledLocalChartNode child_node;
        child_node.chart_data.chart_id = child_id;
        child_node.chart_data.center[0] = parent.chart_data.center[0] + offsets[i][0];
        child_node.chart_data.center[1] = parent.chart_data.center[1] + offsets[i][1];
        child_node.chart_data.center[2] = parent.chart_data.center[2] + offsets[i][2];
        child_node.chart_data.radius = child_r;
        child_node.chart_data.is_active = true;

        child_node.parent_id = parent_chart_id;
        child_node.is_leaf = true;
        child_node.depth = parent.depth + 1;

        charts_.push_back(child_node.chart_data);
        chart_tree_[child_id] = child_node;
    }
}

void AtlasManager::merge_charts(uint32_t parent_chart_id) {
    auto it = chart_tree_.find(parent_chart_id);
    if (it == chart_tree_.end() || it->second.is_leaf) return;

    ScaledLocalChartNode& parent = it->second;

    for (int i = 0; i < 8; ++i) {
        uint32_t child_id = parent.children_ids[i];
        chart_tree_.erase(child_id);
        charts_.erase(std::remove_if(charts_.begin(), charts_.end(),
            [child_id](const LocalChart& c) { return c.chart_id == child_id; }), charts_.end());
        parent.children_ids[i] = 0xFFFFFFFF;
    }

    parent.is_leaf = true;
    parent.chart_data.is_active = true;
}

} // namespace elysia::emulation
