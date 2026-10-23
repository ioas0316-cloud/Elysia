r"""
Elysia Core Physics: Emergent Phase Viscosity & Thermodynamic Operator Engine
==============================================================================
Reinterprets classical static mathematical operators (\sum, \log, \nabla, \int)
as dynamic state space transformers on the S^3 unit quaternion manifold:

1. Derivative (\nabla \times): Spatial shear / vorticity differential operator generating micro-rotor rotation.
2. Sigma (\sum_{phase}): Kuramoto-style S^3 phase-locking & topological synchronization torque.
3. Log (\log_{scale}): Logarithmic scale compression & energy dissipation factor \alpha.
4. Integral (\int / Divergence): Continuous thermodynamic phase transition & unified stress tensor feedback.

Also provides unified optical raymarching & HLSL/GLSL Compute shader generation
mapping order parameter \Phi and Rotor field Q directly into physical optical phenomena.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Helper Quaternion Functions (S^3 Operations)
# =============================================================================

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product for quaternions q = [w, x, y, z].
    q1, q2 shape: [B, 4, H, W, D] or [..., 4]
    """
    w1, x1, y1, z1 = q1[:, 0:1], q1[:, 1:2], q1[:, 2:3], q1[:, 3:4]
    w2, x2, y2, z2 = q2[:, 0:1], q2[:, 1:2], q2[:, 2:3], q2[:, 3:4]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.cat([w, x, y, z], dim=1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Quaternion conjugate q* = [w, -x, -y, -z].
    q shape: [B, 4, ...]
    """
    conj_mask = torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype).view(1, 4, 1, 1, 1)
    return q * conj_mask


def vector_to_quaternion_exp(v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from 3D torque / angular velocity vector to S^3 increment quaternion.
    v shape: [B, 3, H, W, D]
    Returns dq = [w, x, y, z] shape [B, 4, H, W, D]
    """
    v_norm = torch.norm(v, dim=1, keepdim=True) + 1e-8
    theta = v_norm
    axis = v / v_norm

    dq_w = torch.cos(theta / 2.0)
    dq_xyz = axis * torch.sin(theta / 2.0)
    dq = torch.cat([dq_w, dq_xyz], dim=1)
    return F.normalize(dq, p=2, dim=1)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Converts unit quaternion [B, 4, H, W, D] to 3x3 rotation matrices [B, 3, 3, H, W, D].
    """
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    r00 = 1.0 - 2.0 * (y ** 2 + z ** 2)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)

    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x ** 2 + z ** 2)
    r12 = 2.0 * (y * z - x * w)

    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x ** 2 + y ** 2)

    row0 = torch.stack([r00, r01, r02], dim=1)  # [B, 3, H, W, D]
    row1 = torch.stack([r10, r11, r12], dim=1)
    row2 = torch.stack([r20, r21, r22], dim=1)

    R = torch.stack([row0, row1, row2], dim=2)  # [B, 3, 3, H, W, D]
    return R


# =============================================================================
# Spatial Calculus Utilities (Central Differences)
# =============================================================================

def compute_spatial_gradient(V: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """
    Computes 3x3 velocity gradient tensor L = grad(V) in 3D.
    V shape: [B, 3, H, W, D]
    Returns grad_v shape: [B, 3, 3, H, W, D] where grad_v[:, i, j] = dV_i / dx_j
    """
    vx, vy, vz = V[:, 0:1], V[:, 1:2], V[:, 2:3]

    def grad_component(field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_x = F.pad(field, (0, 0, 0, 0, 1, 1), mode='replicate')
        df_dx = (p_x[:, :, 2:, :, :] - p_x[:, :, :-2, :, :]) / (2.0 * dx)

        p_y = F.pad(field, (0, 0, 1, 1, 0, 0), mode='replicate')
        df_dy = (p_y[:, :, :, 2:, :] - p_y[:, :, :, :-2, :]) / (2.0 * dx)

        p_z = F.pad(field, (1, 1, 0, 0, 0, 0), mode='replicate')
        df_dz = (p_z[:, :, :, :, 2:] - p_z[:, :, :, :, :-2]) / (2.0 * dx)

        return df_dx, df_dy, df_dz

    dvx_dx, dvx_dy, dvx_dz = grad_component(vx)
    dvy_dx, dvy_dy, dvy_dz = grad_component(vy)
    dvz_dx, dvz_dy, dvz_dz = grad_component(vz)

    row_x = torch.cat([dvx_dx, dvx_dy, dvx_dz], dim=1)  # [B, 3, H, W, D]
    row_y = torch.cat([dvy_dx, dvy_dy, dvy_dz], dim=1)
    row_z = torch.cat([dvz_dx, dvz_dy, dvz_dz], dim=1)

    grad_v = torch.stack([row_x, row_y, row_z], dim=2)  # [B, 3, 3, H, W, D]
    return grad_v


def extract_curl(grad_v: torch.Tensor) -> torch.Tensor:
    """
    Extracts vorticity curl vector w = grad x V from velocity gradient L.
    w_x = dV_z/dy - dV_y/dz
    w_y = dV_x/dz - dV_z/dx
    w_z = dV_y/dx - dV_x/dy
    """
    dvz_dy = grad_v[:, 2, 1:2]
    dvy_dz = grad_v[:, 1, 2:3]
    w_x = dvz_dy - dvy_dz

    dvx_dz = grad_v[:, 0, 2:3]
    dvz_dx = grad_v[:, 2, 0:1]
    w_y = dvx_dz - dvz_dx

    dvy_dx = grad_v[:, 1, 0:1]
    dvx_dy = grad_v[:, 0, 1:2]
    w_z = dvy_dx - dvx_dy

    vorticity = torch.cat([w_x, w_y, w_z], dim=1)  # [B, 3, H, W, D]
    return vorticity


def gather_6neighbors(Q: torch.Tensor) -> torch.Tensor:
    """
    Gathers 6-connected neighbor quaternions.
    Q shape: [B, 4, H, W, D]
    Returns Q_neighbors shape: [B, 6, 4, H, W, D]
    """
    neighbors = []
    shifts = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    for dx, dy, dz in shifts:
        p_x = (max(0, dx), max(0, -dx))
        p_y = (max(0, dy), max(0, -dy))
        p_z = (max(0, dz), max(0, -dz))

        padded = F.pad(Q, (p_z[0], p_z[1], p_y[0], p_y[1], p_x[0], p_x[1]), mode='replicate')

        sliced = padded[:, :,
                 p_x[1]:p_x[1] + Q.shape[2],
                 p_y[1]:p_y[1] + Q.shape[3],
                 p_z[1]:p_z[1] + Q.shape[4]]
        neighbors.append(sliced)

    Q_neighbors = torch.stack(neighbors, dim=1)  # [B, 6, 4, H, W, D]
    return Q_neighbors


def compute_tensor_divergence(stress: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """
    Computes divergence of 3x3 stress tensor sigma.
    stress shape: [B, 3, 3, H, W, D]
    Returns acceleration a = div(sigma) shape: [B, 3, H, W, D]
    """
    acc_components = []
    for i in range(3):
        sig_i0 = stress[:, i, 0:1]
        sig_i1 = stress[:, i, 1:2]
        sig_i2 = stress[:, i, 2:3]

        p_x = F.pad(sig_i0, (0, 0, 0, 0, 1, 1), mode='replicate')
        dsig_x = (p_x[:, :, 2:, :, :] - p_x[:, :, :-2, :, :]) / (2.0 * dx)

        p_y = F.pad(sig_i1, (0, 0, 1, 1, 0, 0), mode='replicate')
        dsig_y = (p_y[:, :, :, 2:, :] - p_y[:, :, :, :-2, :]) / (2.0 * dx)

        p_z = F.pad(sig_i2, (1, 1, 0, 0, 0, 0), mode='replicate')
        dsig_z = (p_z[:, :, :, :, 2:] - p_z[:, :, :, :, :-2]) / (2.0 * dx)

        div_i = dsig_x + dsig_y + dsig_z
        acc_components.append(div_i)

    acceleration = torch.cat(acc_components, dim=1)  # [B, 3, H, W, D]
    return acceleration


# =============================================================================
# Core Thermodynamic Phase Viscosity Engine Class
# =============================================================================

class EmergentPhaseViscosityEngine:
    r"""
    Thermodynamic Phase Engine operating on S^3 quaternion manifolds.
    Reinterprets \nabla, \sum, \log, \int as continuous topological state operations.
    """

    def __init__(self,
                 K_0: float = 10.0,
                 gamma: float = 0.5,
                 beta: float = 1.0,
                 D_R: float = 0.1,
                 shear_mode: str = "newtonian"):
        """
        K_0: Phase synchronization coupling strength (Kuramoto parameter).
        gamma, beta: Logarithmic scale compression parameters.
        D_R: Rotational Brownian diffusion constant.
        shear_mode: 'newtonian', 'shear_thinning', or 'shear_thickening'.
        """
        self.K_0 = K_0
        self.gamma = gamma
        self.beta = beta
        self.D_R = D_R
        self.shear_mode = shear_mode

    def step(self,
             V: torch.Tensor,
             Q: torch.Tensor,
             T: torch.Tensor,
             dt: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        r"""
        Executes 1 thermodynamic phase transition integration step:

        Step 1 (\nabla \times): Spatial shear vorticity extraction and micro-rotor acceleration.
        Step 2 (\sum_{phase}): S^3 quaternion relative phase-locking restoration torque \tau_{sync}.
        Step 3 (\log_{scale}): Logarithmic energy scale compression and dissipation factor \alpha.
        Step 4 (\int / Divergence): Order parameter \Phi & unified stress tensor calculation for momentum feedback.

        V: Speed / Velocity Field Tensor [B, 3, H, W, D]
        Q: Unit Quaternion Rotor Field [B, 4, H, W, D]
        T: Thermal Temperature Field [B, 1, H, W, D]
        dt: Time step size
        """
        B, _, H, W, D = V.shape
        device = V.device

        # ---------------------------------------------------------------------
        # Step 1: Derivative (\nabla \times) - Spatial Vorticity Extraction
        # ---------------------------------------------------------------------
        grad_v = compute_spatial_gradient(V)  # [B, 3, 3, H, W, D]
        vorticity = extract_curl(grad_v)       # [B, 3, H, W, D]

        # ---------------------------------------------------------------------
        # Step 2: Sigma (\sum_{phase}) - Phase-Locking Kuramoto Torque on S^3
        # ---------------------------------------------------------------------
        Q_neighbors = gather_6neighbors(Q)  # [B, 6, 4, H, W, D]
        Q_self_conj = quaternion_conjugate(Q).unsqueeze(1)  # [B, 1, 4, H, W, D]

        q_rel_list = []
        for n_idx in range(6):
            q_n = Q_neighbors[:, n_idx]
            q_rel_n = quaternion_multiply(q_n, Q_self_conj[:, 0])
            q_rel_list.append(q_rel_n)

        Q_rel = torch.stack(q_rel_list, dim=1)  # [B, 6, 4, H, W, D]

        # Tau_sync = K_0 * sum(Im(Q_rel))
        im_q_rel = Q_rel[:, :, 1:]  # [B, 6, 3, H, W, D]
        tau_sync = self.K_0 * torch.sum(im_q_rel, dim=1)  # [B, 3, H, W, D]

        # ---------------------------------------------------------------------
        # Step 2.1: Thermal Brownian Noise \eta(T) (Langevin Dynamics on S^3)
        # ---------------------------------------------------------------------
        noise_std = torch.sqrt(torch.clamp(2.0 * self.D_R * T * dt, min=1e-12))
        eta_thermal = torch.randn_like(tau_sync) * noise_std  # [B, 3, H, W, D]

        # Total Rotational Vector on Lie Algebra su(2)
        total_torque = vorticity + tau_sync + eta_thermal  # [B, 3, H, W, D]

        # Update Rotor Q via Exponential Map
        dq = vector_to_quaternion_exp(total_torque * dt)
        Q_updated = F.normalize(quaternion_multiply(dq, Q), p=2, dim=1)

        # ---------------------------------------------------------------------
        # Step 3: Log (\log_{scale}) - Logarithmic Energy Dissipation Factor
        # ---------------------------------------------------------------------
        E_sync = torch.sum(tau_sync ** 2, dim=1, keepdim=True)  # [B, 1, H, W, D]

        if self.shear_mode == "shear_thinning":
            alpha = 1.0 / (1.0 + self.gamma * torch.log(1.0 + self.beta * E_sync ** 2))
        elif self.shear_mode == "shear_thickening":
            alpha = 1.0 - 1.0 / (1.0 + self.beta * torch.log(1.0 + self.gamma * E_sync))
        else:  # standard Newtonian
            alpha = 1.0 / (1.0 + self.gamma * torch.log(1.0 + self.beta * E_sync))

        alpha = torch.clamp(alpha, min=1e-4, max=1.0)

        # ---------------------------------------------------------------------
        # Step 4: Integral (\int / Divergence) - Order Parameter & Stress Tensor
        # ---------------------------------------------------------------------
        # Local Order Parameter Phi (Kuramoto Phase Coherence on S^3)
        Q_mean_neighbor = torch.mean(Q_neighbors, dim=1)  # [B, 4, H, W, D]
        Phi = torch.norm(Q_mean_neighbor, dim=1, keepdim=True)  # [B, 1, H, W, D]
        Phi = torch.clamp(Phi, 0.0, 1.0)

        # Dynamic State Regime Weights
        w_solid = torch.sigmoid(20.0 * (Phi - 0.75))
        w_gas = 1.0 - torch.sigmoid(20.0 * (Phi - 0.25))
        w_liquid = torch.clamp(1.0 - w_solid - w_gas, min=0.0)

        # Component Stress Tensors
        # 1. Viscous Stress Tensor: \sigma_{viscous} = \alpha * R * S * R^T
        R_matrix = quaternion_to_rotation_matrix(Q_updated)  # [B, 3, 3, H, W, D]
        S_tensor = 0.5 * (grad_v + grad_v.transpose(1, 2))    # [B, 3, 3, H, W, D]

        # R * S * R^T
        # R_matrix: [B, 3, 3, H, W, D] -> bijxyz
        # S_tensor: [B, 3, 3, H, W, D] -> bjkxyz
        R_S = torch.einsum('bijxyz,bjkxyz->bikxyz', R_matrix, S_tensor)
        viscous_stress = alpha.unsqueeze(2) * torch.einsum('bikxyz,blkxyz->bilxyz', R_S, R_matrix)

        # 2. Solid Elastic Strain Stress Tensor
        identity_3x3 = torch.eye(3, device=device, dtype=V.dtype).view(1, 3, 3, 1, 1, 1)
        elastic_stress = Phi.unsqueeze(2) * (R_matrix - identity_3x3)

        # 3. Gas Thermodynamic Expansion Pressure Stress Tensor
        P_gas = T * (1.0 - Phi) * 0.5  # Ideal Gas Law P ~ T
        gas_stress = -P_gas.unsqueeze(2) * identity_3x3

        # Unified Stress Tensor Integration
        sigma_unified = (w_solid.unsqueeze(2) * elastic_stress +
                         w_liquid.unsqueeze(2) * viscous_stress +
                         w_gas.unsqueeze(2) * gas_stress)

        # Momentum Feedback via Divergence
        acceleration = compute_tensor_divergence(sigma_unified)  # [B, 3, H, W, D]
        V_next = V + acceleration * dt

        metrics = {
            "mean_order_parameter_phi": float(torch.mean(Phi).item()),
            "solid_fraction": float(torch.mean(w_solid).item()),
            "liquid_fraction": float(torch.mean(w_liquid).item()),
            "gas_fraction": float(torch.mean(w_gas).item()),
            "mean_dissipation_alpha": float(torch.mean(alpha).item()),
            "mean_torque_sync": float(torch.mean(torch.norm(tau_sync, dim=1)).item()),
            "max_velocity": float(torch.max(torch.norm(V_next, dim=1)).item()),
        }

        return V_next, Q_updated, metrics


# =============================================================================
# Optical Raymarching & Shader Generator Utilities
# =============================================================================

def compute_phi_gradient_3d(Phi: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """
    Computes spatial gradient nabla Phi for surface interface detection in volume rendering.
    """
    p_x = F.pad(Phi, (0, 0, 0, 0, 1, 1), mode='replicate')
    dphi_dx = (p_x[:, :, 2:, :, :] - p_x[:, :, :-2, :, :]) / (2.0 * dx)

    p_y = F.pad(Phi, (0, 0, 1, 1, 0, 0), mode='replicate')
    dphi_dy = (p_y[:, :, :, 2:, :] - p_y[:, :, :, :-2, :]) / (2.0 * dx)

    p_z = F.pad(Phi, (1, 1, 0, 0, 0, 0), mode='replicate')
    dphi_dz = (p_z[:, :, :, :, 2:] - p_z[:, :, :, :, :-2]) / (2.0 * dx)

    return torch.cat([dphi_dx, dphi_dy, dphi_dz], dim=1)


def generate_max_mipmaps_3d(Phi: torch.Tensor, brick_size: int = 4) -> torch.Tensor:
    """
    Generates hierarchical 3D Max-Mipmap tensor for Empty Space Skipping in Compute Shaders.
    """
    return F.max_pool3d(Phi, kernel_size=brick_size, stride=brick_size)


def generate_glsl_volume_raymarch_shader() -> str:
    r"""
    Generates GLSL volume raymarching shader source code.
    Maps Order Parameter \Phi and Rotor field Q directly into physical optical phenomena.
    """
    return """// ============================================================================
// Elysia Unified Optical Volume Raymarching GLSL Shader
// Maps Topological Order Parameter \\Phi and Rotor Q directly to optics.
// ============================================================================

#version 450 core

in vec2 v_TexCoord;
out vec4 FragColor;

uniform sampler3D u_PhiTexture;    // [0, 1] Order Parameter Field
uniform sampler3D u_QTexture;      // Unit Quaternion Rotor Field (w, x, y, z)
uniform sampler3D u_PhiMaxMipMap;  // Max Mipmap for Empty Space Skipping

uniform vec3 u_CameraPos;
uniform vec3 u_LightPos;
uniform mat4 u_InvProjection;
uniform mat4 u_InvView;

const float AIR_THRESHOLD = 0.005;
const int MAX_STEPS = 128;
const float STEP_SIZE = 0.01;

// Quaternion to Rotation Matrix
mat3 quaternion_to_matrix(vec4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return mat3(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w),
        2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w),
        2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)
    );
}

void main() {
    // Generate camera ray
    vec4 target = u_InvProjection * vec4(v_TexCoord * 2.0 - 1.0, 1.0, 1.0);
    vec3 rd = normalize((u_InvView * vec4(target.xyz, 0.0)).xyz);
    vec3 ro = u_CameraPos;

    float t = 0.0;
    float transmittance = 1.0;
    vec3 accumulated_color = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; i++) {
        if (transmittance < 0.01) break; // Early Ray Termination

        vec3 pos = ro + rd * t;
        vec3 uvw = pos * 0.5 + 0.5; // Map [-1, 1] to [0, 1] grid

        if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) {
            t += STEP_SIZE;
            continue;
        }

        // 1. EMPTY SPACE SKIPPING via Max Mipmap
        float phi_max = texture(u_PhiMaxMipMap, uvw).r;
        if (phi_max < AIR_THRESHOLD) {
            t += STEP_SIZE * 4.0; // Skip empty brick space
            continue;
        }

        // 2. High Resolution Field Sampling
        float phi = texture(u_PhiTexture, uvw).r;
        vec4 Q = texture(u_QTexture, uvw);

        if (phi > AIR_THRESHOLD) {
            // Compute Phi Gradient (Surface Normal Interface)
            vec3 eps = vec3(0.005, 0.0, 0.0);
            float dpx = texture(u_PhiTexture, uvw + eps.xyy).r - texture(u_PhiTexture, uvw - eps.xyy).r;
            float dpy = texture(u_PhiTexture, uvw + eps.yxy).r - texture(u_PhiTexture, uvw - eps.yxy).r;
            float dpz = texture(u_PhiTexture, uvw + eps.yyx).r - texture(u_PhiTexture, uvw - eps.yyx).r;
            vec3 grad = vec3(dpx, dpy, dpz);
            float grad_len = length(grad);

            vec3 N = (grad_len > 1e-4) ? normalize(-grad) : rd;

            // Optical Spectrum Mapping based on Regime \\Phi
            float density = pow(phi, 2.0) * 8.0 + grad_len * 15.0;
            float ior = 1.0 + 0.45 * phi * phi;

            // Photoelastic Birefringence Rainbow Spectrum from Rotor Q Spin
            float phase_angle = 2.0 * acos(clamp(Q.x, -1.0, 1.0));
            vec3 birefringence = 0.5 + 0.5 * cos(phase_angle + vec3(0.0, 2.094, 4.188));

            // Shading & Lighting
            vec3 L = normalize(u_LightPos - pos);
            mat3 R_rot = quaternion_to_matrix(Q);
            vec3 aniso_axis = R_rot * vec3(0.0, 0.0, 1.0);

            float spec = pow(max(0.0, dot(reflect(-L, N), -rd)), 16.0) * abs(dot(aniso_axis, N));
            float diff = max(0.0, dot(N, L));

            vec3 light_color = vec3(1.0, 0.95, 0.85);
            vec3 step_color = mix(birefringence, light_color, 0.5) * (diff + spec);

            // Beer-Lambert Absorption
            float alpha = 1.0 - exp(-density * STEP_SIZE);
            accumulated_color += transmittance * alpha * step_color;
            transmittance *= (1.0 - alpha);

            t += max(0.002, STEP_SIZE / (grad_len + 1.0)); // Adaptive step
        } else {
            t += STEP_SIZE;
        }
    }

    FragColor = vec4(accumulated_color, 1.0 - transmittance);
}
"""


def generate_hlsl_compute_shader() -> str:
    r"""
    Generates HLSL Compute shader for DirectX / Vulkan Pipelined Ray Compaction.
    """
    return """// ============================================================================
// Elysia Wavefront Ray Compaction & Volume Raymarch HLSL Compute Shader
// ============================================================================

#define TILE_SIZE 8
#define AIR_THRESHOLD 0.005f

Texture3D<float> g_PhiMaxMipMap : register(t0);
Texture3D<float> g_PhiTexture   : register(t1);
Texture3D<float4> g_QTexture   : register(t2);

RWTexture2D<float4> g_OutputImage : register(u0);

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void CS_RaymarchVolume(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixelPos = dispatchThreadID.xy;
    float3 ro = vec3(0.0, 0.0, -2.0);
    float3 rd = normalize(vec3((pixelPos / 512.0) * 2.0 - 1.0, 1.0));

    float t = 0.0f;
    float transmittance = 1.0f;
    float3 color = float3(0, 0, 0);

    while (t < 4.0f && transmittance > 0.01f) {
        float3 pos = ro + rd * t;
        float3 uvw = pos * 0.5f + 0.5f;

        float phiMax = g_PhiMaxMipMap.SampleLevel(g_LinearSampler, uvw, 0);
        if (phiMax < AIR_THRESHOLD) {
            t += 0.08f; // Empty space leap
            continue;
        }

        float phi = g_PhiTexture.SampleLevel(g_LinearSampler, uvw, 0);
        if (phi > AIR_THRESHOLD) {
            float alpha = 1.0f - exp(-phi * phi * 10.0f * 0.02f);
            color += transmittance * alpha * float3(phi, 0.5f * phi, 1.0f - phi);
            transmittance *= (1.0f - alpha);
            t += 0.02f;
        } else {
            t += 0.04f;
        }
    }

    g_OutputImage[pixelPos] = float4(color, 1.0f - transmittance);
}
"""
