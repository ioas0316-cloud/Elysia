// ============================================================================
// WebGPU WGSL Compute Shader: Resonant SDF & Analytic Normal Generator
// ============================================================================

struct FieldUniforms {
    time : f32,
    amplitude : f32,
    sigma : f32,
    omega : f32,
    num_points : u32,
    _padding0 : u32,
    _padding1 : u32,
    _padding2 : u32,
    k_audio : vec4<f32>,     // 3D Audio Wavevector + padding
    imu_accel : vec4<f32>,   // 3D IMU Acceleration + padding
    z_text : vec4<f32>,      // Text Latent Phase + padding
}

@group(0) @binding(0) var<uniform> params : FieldUniforms;
@group(0) @binding(1) var<storage, read> query_positions : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out_distances : array<f32>;
@group(0) @binding(3) var<storage, read_write> out_normals : array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_points) {
        return;
    }

    // 1. Position & IMU Gravity Warp
    let raw_pos = query_positions[idx].xyz;
    let warped_pos = raw_pos + params.imu_accel.xyz * 0.1;

    // 2. Base Distance & Wave Phase
    let norm_pos = length(warped_pos) + 1e-8;
    let d_base = norm_pos - 1.0;

    let k_dot_x = dot(warped_pos, params.k_audio.xyz);
    let z_dot_x = dot(warped_pos, params.z_text.xyz);
    let phase = k_dot_x + z_dot_x - (params.omega * params.time % 6.28318530718);

    // 3. Perturbed Distance Calculation
    let cos_p = cos(phase);
    let sin_p = sin(phase);
    let mask = exp(-abs(d_base) / params.sigma);

    out_distances[idx] = d_base + params.amplitude * mask * cos_p;

    // 4. Zero-Extra-Sample Analytic Normal Computation
    let grad_base = warped_pos / norm_pos;
    let sgn_d = select(-1.0, 1.0, d_base > 0.0);

    let d_mask_term = -(sgn_d / params.sigma) * mask * cos_p;
    let d_phase_term = -mask * sin_p;

    let grad_perturbed = grad_base * (1.0 + params.amplitude * d_mask_term)
                       + params.k_audio.xyz * (params.amplitude * d_phase_term);

    let norm_vector = normalize(grad_perturbed);
    out_normals[idx] = vec4<f32>(norm_vector, 0.0);
}
