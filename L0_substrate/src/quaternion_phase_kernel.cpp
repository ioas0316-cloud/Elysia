#include <cmath>
#include <vector>

// 128-bit aligned structure for GPU/CPU cache efficiency
struct alignas(16) QuaternionPhasor {
    float q_w, q_x, q_y, q_z;         // Quaternion Phase Angle
    float pos_x, pos_y, pos_z, amp;   // 3D Position Anchor & Amplitude
    float scale_x, scale_y, scale_z, cutoff_radius; // Spatial extent
};

// Inline FMA-based Hamilton Product
inline void quaternion_mul_cpu(
    float a_w, float a_x, float a_y, float a_z,
    float b_w, float b_x, float b_y, float b_z,
    float& r_w, float& r_x, float& r_y, float& r_z
) {
    r_w = a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z;
    r_x = a_w * b_x + a_x * b_w + a_y * b_z - a_z * b_y;
    r_y = a_w * b_y - a_x * b_z + a_y * b_w + a_z * b_x;
    r_z = a_w * b_z + a_x * b_y - a_y * b_x + a_z * b_w;
}

// C++ High-Throughput Quaternion Torque Integration Loop
extern "C" void update_quaternion_phase_cpp(
    float* phasors_flat,
    const float* target_field_flat,
    float dt,
    float gamma,
    int num_nodes
) {
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; ++i) {
        int p_idx = i * 12; // 12 floats per QuaternionPhasor
        int t_idx = i * 4;  // 4 floats per target_q

        float q_w = phasors_flat[p_idx + 0];
        float q_x = phasors_flat[p_idx + 1];
        float q_y = phasors_flat[p_idx + 2];
        float q_z = phasors_flat[p_idx + 3];

        float target_w = target_field_flat[t_idx + 0];
        float target_x = target_field_flat[t_idx + 1];
        float target_y = target_field_flat[t_idx + 2];
        float target_z = target_field_flat[t_idx + 3];

        // Conjugate quaternion for inverse rotation: q_inv = (w, -x, -y, -z)
        float inv_w = q_w;
        float inv_x = -q_x;
        float inv_y = -q_y;
        float inv_z = -q_z;

        // err_q = target_q x q_inv
        float err_w, err_x, err_y, err_z;
        quaternion_mul_cpu(
            target_w, target_x, target_y, target_z,
            inv_w, inv_x, inv_y, inv_z,
            err_w, err_x, err_y, err_z
        );

        // Torque vector
        float sign_w = (err_w >= 0.0f) ? 1.0f : -1.0f;
        float torque_x = 2.0f * sign_w * err_x;
        float torque_y = 2.0f * sign_w * err_y;
        float torque_z = 2.0f * sign_w * err_z;

        // Direct exponent map rotation integration: dq = (0, torque) x q
        float dq_w, dq_x, dq_y, dq_z;
        quaternion_mul_cpu(
            0.0f, torque_x, torque_y, torque_z,
            q_w, q_x, q_y, q_z,
            dq_w, dq_x, dq_y, dq_z
        );

        q_w += gamma * dt * 0.5f * dq_w;
        q_x += gamma * dt * 0.5f * dq_x;
        q_y += gamma * dt * 0.5f * dq_y;
        q_z += gamma * dt * 0.5f * dq_z;

        // Normalization
        float norm = std::sqrt(q_w * q_w + q_x * q_x + q_y * q_y + q_z * q_z) + 1e-8f;
        phasors_flat[p_idx + 0] = q_w / norm;
        phasors_flat[p_idx + 1] = q_x / norm;
        phasors_flat[p_idx + 2] = q_y / norm;
        phasors_flat[p_idx + 3] = q_z / norm;
    }
}
