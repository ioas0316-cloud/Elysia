#ifndef CAUSAL_SERVER_SIMD_H
#define CAUSAL_SERVER_SIMD_H

#include "core/object/class_db.h"
#include "core/os/worker_thread_pool.h"
#include "core/templates/local_vector.h"
#include "servers/rendering_server.h"

struct alignas(64) SIMDTransform3D {
    float basis[9];
    float pad0;
    float origin[3];
    float pad1;
};

class CausalServerSIMD : public Object {
    GDCLASS(CausalServerSIMD, Object);

    static CausalServerSIMD *singleton;

private:
    LocalVector<SIMDTransform3D> local_transforms;
    LocalVector<SIMDTransform3D> global_transforms;
    LocalVector<int32_t> parent_indices;
    LocalVector<uint8_t> dirty_flags;
    LocalVector<RID> rendering_instances;
    LocalVector<int32_t> cell_depths;

    LocalVector<LocalVector<int32_t>> depth_layers;
    RID world_scenario;

    FORCE_INLINE SIMDTransform3D _to_simd_transform(const Transform3D &p_t);
    void _process_depth_layer_chunk(uint32_t p_layer_idx, uint32_t p_start_idx, uint32_t p_end_idx);

protected:
    static void _bind_methods();

public:
    static CausalServerSIMD *get_singleton() { return singleton; }

    CausalServerSIMD();
    ~CausalServerSIMD();

    int32_t register_causal_cell_simd(RID p_mesh, int32_t p_parent_idx = -1, const Transform3D &p_initial_transform = Transform3D());
    void shift_variable(int32_t p_cell_idx, const Vector3 &p_delta_shift);
    void propagate_and_project_parallel();

    void reparent_cell(int32_t p_cell_idx, int32_t p_new_parent_idx);
    void rebuild_depth_layers();

    size_t get_cell_count() const { return local_transforms.size(); }
    size_t get_depth_layer_count() const { return depth_layers.size(); }
    const SIMDTransform3D &get_global_transform(int32_t p_idx) const { return global_transforms[p_idx]; }
    int32_t get_cell_depth(int32_t p_idx) const { return cell_depths[p_idx]; }
};

#endif // CAUSAL_SERVER_SIMD_H
