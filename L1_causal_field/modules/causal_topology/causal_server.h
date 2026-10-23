#ifndef CAUSAL_SERVER_H
#define CAUSAL_SERVER_H

#include "core/object/class_db.h"
#include "core/templates/local_vector.h"
#include "servers/rendering_server.h"
#include "servers/physics_server_3d.h"

struct CausalCell {
    RID rendering_instance;
    RID physics_body;

    Transform3D local_transform;
    Transform3D global_transform;

    int32_t parent_index = -1;
    LocalVector<int32_t> child_indices;

    Vector3 state_variable;
    bool is_dirty = false;
};

class CausalServer : public Object {
    GDCLASS(CausalServer, Object);

    static CausalServer *singleton;

private:
    LocalVector<CausalCell> causal_matrix;
    RID world_scenario;

protected:
    static void _bind_methods();

public:
    static CausalServer *get_singleton() { return singleton; }

    CausalServer();
    ~CausalServer();

    int32_t register_causal_cell(RID p_mesh, int32_t p_parent_idx = -1, const Transform3D &p_initial_transform = Transform3D());
    void shift_variable(int32_t p_cell_idx, const Vector3 &p_delta_shift);
    void propagate_and_project();

    RID get_world_scenario() const { return world_scenario; }
    size_t get_cell_count() const { return causal_matrix.size(); }
    const CausalCell &get_cell(int32_t p_idx) const { return causal_matrix[p_idx]; }
};

#endif // CAUSAL_SERVER_H
