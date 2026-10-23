#include "causal_server_simd.h"
#include "core/os/os.h"

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

CausalServerSIMD *CausalServerSIMD::singleton = nullptr;

void CausalServerSIMD::_bind_methods() {
    ClassDB::bind_method(D_METHOD("register_causal_cell_simd", "mesh", "parent_idx", "initial_transform"), &CausalServerSIMD::register_causal_cell_simd, DEFVAL(-1), DEFVAL(Transform3D()));
    ClassDB::bind_method(D_METHOD("shift_variable", "cell_idx", "delta_shift"), &CausalServerSIMD::shift_variable);
    ClassDB::bind_method(D_METHOD("propagate_and_project_parallel"), &CausalServerSIMD::propagate_and_project_parallel);
    ClassDB::bind_method(D_METHOD("reparent_cell", "cell_idx", "new_parent_idx"), &CausalServerSIMD::reparent_cell);
    ClassDB::bind_method(D_METHOD("rebuild_depth_layers"), &CausalServerSIMD::rebuild_depth_layers);
}

FORCE_INLINE void simd_transform_multiply(const SIMDTransform3D &p_parent, const SIMDTransform3D &p_local, SIMDTransform3D &r_out) {
#if defined(__AVX2__)
    __m256 p_row0 = _mm256_loadu_ps(&p_parent.basis[0]);
    __m256 p_row1 = _mm256_loadu_ps(&p_parent.basis[4]);

    for (int i = 0; i < 3; i++) {
        __m256 l_x = _mm256_set1_ps(p_local.basis[i * 3 + 0]);
        __m256 l_y = _mm256_set1_ps(p_local.basis[i * 3 + 1]);
        __m256 l_z = _mm256_set1_ps(p_local.basis[i * 3 + 2]);

        __m256 res = _mm256_fmadd_ps(l_x, p_row0, _mm256_mul_ps(l_y, p_row1));
        _mm256_storeu_ps(&r_out.basis[i * 3], res);
    }
    r_out.origin[0] = p_parent.basis[0] * p_local.origin[0] + p_parent.basis[1] * p_local.origin[1] + p_parent.basis[2] * p_local.origin[2] + p_parent.origin[0];
    r_out.origin[1] = p_parent.basis[3] * p_local.origin[0] + p_parent.basis[4] * p_local.origin[1] + p_parent.basis[5] * p_local.origin[2] + p_parent.origin[1];
    r_out.origin[2] = p_parent.basis[6] * p_local.origin[0] + p_parent.basis[7] * p_local.origin[1] + p_parent.basis[8] * p_local.origin[2] + p_parent.origin[2];
#else
    const float *p_b = p_parent.basis;
    const float *l_b = p_local.basis;

    r_out.basis[0] = p_b[0] * l_b[0] + p_b[1] * l_b[3] + p_b[2] * l_b[6];
    r_out.basis[1] = p_b[0] * l_b[1] + p_b[1] * l_b[4] + p_b[2] * l_b[7];
    r_out.basis[2] = p_b[0] * l_b[2] + p_b[1] * l_b[5] + p_b[2] * l_b[8];

    r_out.basis[3] = p_b[3] * l_b[0] + p_b[4] * l_b[3] + p_b[5] * l_b[6];
    r_out.basis[4] = p_b[3] * l_b[1] + p_b[4] * l_b[4] + p_b[5] * l_b[7];
    r_out.basis[5] = p_b[3] * l_b[2] + p_b[4] * l_b[5] + p_b[5] * l_b[8];

    r_out.basis[6] = p_b[6] * l_b[0] + p_b[7] * l_b[3] + p_b[8] * l_b[6];
    r_out.basis[7] = p_b[6] * l_b[1] + p_b[7] * l_b[4] + p_b[8] * l_b[7];
    r_out.basis[8] = p_b[6] * l_b[2] + p_b[7] * l_b[5] + p_b[8] * l_b[8];

    r_out.origin[0] = p_b[0] * p_local.origin[0] + p_b[1] * p_local.origin[1] + p_b[2] * p_local.origin[2] + p_parent.origin[0];
    r_out.origin[1] = p_b[3] * p_local.origin[0] + p_b[4] * p_local.origin[1] + p_b[5] * p_local.origin[2] + p_parent.origin[1];
    r_out.origin[2] = p_b[6] * p_local.origin[0] + p_b[7] * p_local.origin[1] + p_b[8] * p_local.origin[2] + p_parent.origin[2];
#endif
}

FORCE_INLINE SIMDTransform3D CausalServerSIMD::_to_simd_transform(const Transform3D &p_t) {
    SIMDTransform3D simd;
    simd.basis[0] = p_t.basis.rows[0][0]; simd.basis[1] = p_t.basis.rows[0][1]; simd.basis[2] = p_t.basis.rows[0][2];
    simd.basis[3] = p_t.basis.rows[1][0]; simd.basis[4] = p_t.basis.rows[1][1]; simd.basis[5] = p_t.basis.rows[1][2];
    simd.basis[6] = p_t.basis.rows[2][0]; simd.basis[7] = p_t.basis.rows[2][1]; simd.basis[8] = p_t.basis.rows[2][2];
    simd.pad0 = 0.0f;

    simd.origin[0] = p_t.origin.x;
    simd.origin[1] = p_t.origin.y;
    simd.origin[2] = p_t.origin.z;
    simd.pad1 = 0.0f;
    return simd;
}

CausalServerSIMD::CausalServerSIMD() {
    singleton = this;
    if (RenderingServer::get_singleton()) {
        world_scenario = RenderingServer::get_singleton()->scenario_create();
    }
}

CausalServerSIMD::~CausalServerSIMD() {
    singleton = nullptr;
}

int32_t CausalServerSIMD::register_causal_cell_simd(RID p_mesh, int32_t p_parent_idx, const Transform3D &p_initial_transform) {
    int32_t new_idx = local_transforms.size();

    int32_t depth = 0;
    if (p_parent_idx >= 0 && p_parent_idx < new_idx) {
        depth = cell_depths[p_parent_idx] + 1;
    }

    RID instance_rid;
    if (RenderingServer::get_singleton() && p_mesh.is_valid()) {
        instance_rid = RenderingServer::get_singleton()->instance_create();
        RenderingServer::get_singleton()->instance_set_base(instance_rid, p_mesh);
        RenderingServer::get_singleton()->instance_set_scenario(instance_rid, world_scenario);
    }

    SIMDTransform3D simd_tf = _to_simd_transform(p_initial_transform);

    local_transforms.push_back(simd_tf);
    global_transforms.push_back(simd_tf);
    parent_indices.push_back(p_parent_idx);
    dirty_flags.push_back(1);
    rendering_instances.push_back(instance_rid);
    cell_depths.push_back(depth);

    if ((int32_t)depth_layers.size() <= depth) {
        depth_layers.resize(depth + 1);
    }
    depth_layers[depth].push_back(new_idx);

    return new_idx;
}

void CausalServerSIMD::shift_variable(int32_t p_cell_idx, const Vector3 &p_delta_shift) {
    if (p_cell_idx < 0 || p_cell_idx >= (int32_t)local_transforms.size()) return;

    local_transforms[p_cell_idx].origin[0] += p_delta_shift.x;
    local_transforms[p_cell_idx].origin[1] += p_delta_shift.y;
    local_transforms[p_cell_idx].origin[2] += p_delta_shift.z;
    dirty_flags[p_cell_idx] = 1;
}

void CausalServerSIMD::_process_depth_layer_chunk(uint32_t p_layer_idx, uint32_t p_start_idx, uint32_t p_end_idx) {
    const LocalVector<int32_t> &layer_nodes = depth_layers[p_layer_idx];

    for (uint32_t i = p_start_idx; i < p_end_idx; i++) {
        int32_t idx = layer_nodes[i];
        int32_t parent_idx = parent_indices[idx];

        if (dirty_flags[idx] || (parent_idx >= 0 && dirty_flags[parent_idx])) {
            if (parent_idx >= 0) {
                simd_transform_multiply(global_transforms[parent_idx], local_transforms[idx], global_transforms[idx]);
            } else {
                global_transforms[idx] = local_transforms[idx];
            }

            if (RenderingServer::get_singleton() && rendering_instances[idx].is_valid()) {
                const SIMDTransform3D &gt = global_transforms[idx];
                Transform3D godot_transform(
                    Basis(gt.basis[0], gt.basis[1], gt.basis[2],
                          gt.basis[3], gt.basis[4], gt.basis[5],
                          gt.basis[6], gt.basis[7], gt.basis[8]),
                    Vector3(gt.origin[0], gt.origin[1], gt.origin[2])
                );
                RenderingServer::get_singleton()->instance_set_transform(rendering_instances[idx], godot_transform);
            }
            dirty_flags[idx] = 1;
        }
    }
}

void CausalServerSIMD::propagate_and_project_parallel() {
    WorkerThreadPool *pool = WorkerThreadPool::get_singleton();

    for (uint32_t layer = 0; layer < depth_layers.size(); layer++) {
        uint32_t total_elements = depth_layers[layer].size();
        if (total_elements == 0) continue;

        uint32_t chunk_size = 1024;
        if (pool && total_elements > chunk_size) {
            WorkerThreadPool::GroupID group_id = pool->add_template_group_task(
                this,
                &CausalServerSIMD::_process_depth_layer_chunk,
                layer,
                total_elements,
                chunk_size,
                true,
                SNAME("CausalSIMDPropagation")
            );
            pool->wait_for_group_task_completion(group_id);
        } else {
            _process_depth_layer_chunk(layer, 0, total_elements);
        }
    }

    memset(dirty_flags.ptr(), 0, dirty_flags.size() * sizeof(uint8_t));
}

void CausalServerSIMD::reparent_cell(int32_t p_cell_idx, int32_t p_new_parent_idx) {
    if (p_cell_idx < 0 || p_cell_idx >= (int32_t)local_transforms.size()) return;

    int32_t old_parent = parent_indices[p_cell_idx];
    if (old_parent == p_new_parent_idx) return;

    parent_indices[p_cell_idx] = p_new_parent_idx;
    dirty_flags[p_cell_idx] = 1;

    rebuild_depth_layers();
}

void CausalServerSIMD::rebuild_depth_layers() {
    uint32_t total_cells = local_transforms.size();
    if (total_cells == 0) return;

    for (uint32_t i = 0; i < depth_layers.size(); i++) {
        depth_layers[i].clear();
    }

    for (uint32_t i = 0; i < total_cells; i++) {
        int32_t parent_idx = parent_indices[i];
        int32_t depth = 0;

        if (parent_idx >= 0 && parent_idx < (int32_t)i) {
            depth = cell_depths[parent_idx] + 1;
        }

        cell_depths[i] = depth;

        if ((int32_t)depth_layers.size() <= depth) {
            depth_layers.resize(depth + 1);
        }
        depth_layers[depth].push_back(i);
    }
}
