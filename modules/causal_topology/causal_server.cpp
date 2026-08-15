#include "causal_server.h"

CausalServer *CausalServer::singleton = nullptr;

void CausalServer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("register_causal_cell", "mesh", "parent_idx", "initial_transform"), &CausalServer::register_causal_cell, DEFVAL(-1), DEFVAL(Transform3D()));
    ClassDB::bind_method(D_METHOD("shift_variable", "cell_idx", "delta_shift"), &CausalServer::shift_variable);
    ClassDB::bind_method(D_METHOD("propagate_and_project"), &CausalServer::propagate_and_project);
}

CausalServer::CausalServer() {
    singleton = this;
    if (RenderingServer::get_singleton()) {
        world_scenario = RenderingServer::get_singleton()->scenario_create();
    }
}

CausalServer::~CausalServer() {
    singleton = nullptr;
}

int32_t CausalServer::register_causal_cell(RID p_mesh, int32_t p_parent_idx, const Transform3D &p_initial_transform) {
    CausalCell cell;
    cell.local_transform = p_initial_transform;
    cell.parent_index = p_parent_idx;

    if (RenderingServer::get_singleton() && p_mesh.is_valid()) {
        cell.rendering_instance = RenderingServer::get_singleton()->instance_create();
        RenderingServer::get_singleton()->instance_set_base(cell.rendering_instance, p_mesh);
        RenderingServer::get_singleton()->instance_set_scenario(cell.rendering_instance, world_scenario);
    }

    int32_t new_idx = causal_matrix.size();
    causal_matrix.push_back(cell);

    if (p_parent_idx >= 0 && p_parent_idx < new_idx) {
        causal_matrix[p_parent_idx].child_indices.push_back(new_idx);
    }

    return new_idx;
}

void CausalServer::shift_variable(int32_t p_cell_idx, const Vector3 &p_delta_shift) {
    if (p_cell_idx < 0 || p_cell_idx >= (int32_t)causal_matrix.size()) return;

    causal_matrix[p_cell_idx].state_variable += p_delta_shift;
    causal_matrix[p_cell_idx].local_transform.origin += p_delta_shift;
    causal_matrix[p_cell_idx].is_dirty = true;
}

void CausalServer::propagate_and_project() {
    uint32_t size = causal_matrix.size();
    for (uint32_t i = 0; i < size; i++) {
        CausalCell &cell = causal_matrix[i];

        if (cell.is_dirty || (cell.parent_index >= 0 && causal_matrix[cell.parent_index].is_dirty)) {
            if (cell.parent_index >= 0) {
                cell.global_transform = causal_matrix[cell.parent_index].global_transform * cell.local_transform;
            } else {
                cell.global_transform = cell.local_transform;
            }

            if (RenderingServer::get_singleton() && cell.rendering_instance.is_valid()) {
                RenderingServer::get_singleton()->instance_set_transform(cell.rendering_instance, cell.global_transform);
            }

            cell.is_dirty = false;
        }
    }
}
