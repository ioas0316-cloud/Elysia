import torch
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

# Dummy jit decorator to prevent import failures when triton is missing
def dummy_jit(fn):
    return fn

jit_decorator = triton.jit if HAS_TRITON else dummy_jit

@jit_decorator
def rotor_vector_kernel(
    v_ptr,         # Pointer to input vectors (batch_size, dim)
    bv_ptr,        # Pointer to bivectors (batch_size, dim)
    out_ptr,       # Pointer to output vectors (batch_size, dim)
    s_val_ptr,     # Pointer to scalar values representing theta (batch_size)
    dt,            # Time step delta
    stride_v_b, stride_v_d, # Strides for v
    stride_bv_b, stride_bv_d, # Strides for bv
    stride_out_b, stride_out_d, # Strides for out
    dim, # Number of dimensions (e.g. 21)
    BLOCK_SIZE
):
    if not HAS_TRITON:
        return
    # Triton jit kernel code below is parsed by Triton compiler only when compiled
    pid = tl.program_id(axis=0)
    batch_offset = pid * BLOCK_SIZE
    offsets = batch_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < stride_v_b
    dim_offsets = tl.arange(0, dim)
    s_val = tl.load(s_val_ptr + offsets, mask=mask)
    theta = tl.math.acos(tl.math.max(tl.math.min(s_val, 1.0), -1.0))
    angle = theta * dt
    cos_a = tl.math.cos(angle)
    sin_a = tl.math.sin(angle)
    # Placeholder loop
    pass

@jit_decorator
def rotor_plane_kernel(
    v_ptr,
    out_ptr,
    theta,
    p1,
    p2,
    dt,
    batch_size,
    dim,
    BLOCK_SIZE
):
    if not HAS_TRITON:
        return
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    angle = theta * dt
    cos_t = tl.math.cos(angle)
    sin_t = tl.math.sin(angle)

    v_p1_ptrs = v_ptr + offsets * dim + p1
    v_p2_ptrs = v_ptr + offsets * dim + p2

    x = tl.load(v_p1_ptrs, mask=mask)
    y = tl.load(v_p2_ptrs, mask=mask)

    new_x = x * cos_t - y * sin_t
    new_y = x * sin_t + y * cos_t

    out_p1_ptrs = out_ptr + offsets * dim + p1
    out_p2_ptrs = out_ptr + offsets * dim + p2

    tl.store(out_p1_ptrs, new_x, mask=mask)
    tl.store(out_p2_ptrs, new_y, mask=mask)


class ElysiaFastCore:
    """
    Triton-accelerated core engine for Elysia.
    If Triton is not available (e.g. Windows), it falls back to standard PyTorch matrix operations.
    """
    @staticmethod
    def apply_plane_rotors_batch(v: torch.Tensor, theta: float, p1: int, p2: int, dt: float = 1.0):
        """
        Applies a plane rotation (Givens rotation) on a batch of vectors.
        v: (batch_size, dim) tensor
        """
        if not HAS_TRITON or not v.is_cuda:
            # Fallback to pure PyTorch matrix operation (Works on CPU/GPU seamlessly)
            cos_t = math.cos(theta * dt)
            sin_t = math.sin(theta * dt)
            out = v.clone()
            if p1 < out.shape[1] and p2 < out.shape[1]:
                x = out[:, p1].clone()
                y = out[:, p2].clone()
                out[:, p1] = x * cos_t - y * sin_t
                out[:, p2] = x * sin_t + y * cos_t
            return out

        batch_size, dim = v.shape
        out = v.clone()

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(batch_size, BLOCK_SIZE),)

        rotor_plane_kernel[grid](
            v_ptr=v,
            out_ptr=out,
            theta=theta,
            p1=p1,
            p2=p2,
            dt=dt,
            batch_size=batch_size,
            dim=dim,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return out


@jit_decorator
def _complex_rotor_step_kernel(
    x_ptr, v_ptr, a_ptr, F_ptr,
    M, D, G, K, N, dt,
    n_elements,
    BLOCK_SIZE
):
    if not HAS_TRITON:
        return
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    real_offsets = offsets * 2
    imag_offsets = offsets * 2 + 1

    x_r = tl.load(x_ptr + real_offsets, mask=mask)
    x_i = tl.load(x_ptr + imag_offsets, mask=mask)

    v_r = tl.load(v_ptr + real_offsets, mask=mask)
    v_i = tl.load(v_ptr + imag_offsets, mask=mask)

    F_r = tl.load(F_ptr + real_offsets, mask=mask)
    F_i = tl.load(F_ptr + imag_offsets, mask=mask)

    fric_r = D * v_r - G * v_i
    fric_i = D * v_i + G * v_r

    stiff_r = K * x_r - N * x_i
    stiff_i = K * x_i + N * x_r

    a_r = (F_r - fric_r - stiff_r) / M
    a_i = (F_i - fric_i - stiff_i) / M

    v_new_r = v_r + a_r * dt
    v_new_i = v_i + a_i * dt

    x_new_r = x_r + v_new_r * dt
    x_new_i = x_i + v_new_i * dt

    tl.store(a_ptr + real_offsets, a_r, mask=mask)
    tl.store(a_ptr + imag_offsets, a_i, mask=mask)

    tl.store(v_ptr + real_offsets, v_new_r, mask=mask)
    tl.store(v_ptr + imag_offsets, v_new_i, mask=mask)

    tl.store(x_ptr + real_offsets, x_new_r, mask=mask)
    tl.store(x_ptr + imag_offsets, x_new_i, mask=mask)


def fast_complex_rotor_step(M, D, G, K, N, x: torch.Tensor, v: torch.Tensor, F: torch.Tensor, dt: float):
    """
    Triton-accelerated version of SovereignMath.complex_rotor_step.
    Operates on batched complex tensors. Falls back to pure PyTorch if Triton is not available.
    """
    if not HAS_TRITON or not x.is_cuda:
        # Fallback to pure PyTorch complex differential solver
        friction = (D + 1j * G) * v
        stiffness = (K + 1j * N) * x
        a_new = (F - friction - stiffness) / M
        v_new = v + a_new.real * dt
        x_new = x + v_new * dt
        return x_new, v_new, a_new

    assert x.is_complex() and v.is_complex() and F.is_complex()
    assert x.is_cuda and v.is_cuda and F.is_cuda

    x_f = torch.view_as_real(x).view(-1)
    v_f = torch.view_as_real(v).view(-1)
    F_f = torch.view_as_real(F).view(-1)

    a_f = torch.empty_like(x_f)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _complex_rotor_step_kernel[grid](
        x_f, v_f, a_f, F_f,
        M, D, G, K, N, dt,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    x_new = torch.view_as_complex(x_f.view(-1, 2)).view(x.shape)
    v_new = torch.view_as_complex(v_f.view(-1, 2)).view(v.shape)
    a_new = torch.view_as_complex(a_f.view(-1, 2)).view(x.shape)

    return x_new, v_new, a_new

ElysiaFastCore.fast_complex_rotor_step = fast_complex_rotor_step
