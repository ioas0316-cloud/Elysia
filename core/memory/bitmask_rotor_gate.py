import numpy as np
import math

try:
    from numba import cuda, float32, uint64
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @cuda.jit
    def _wye_mirror_match_kernel(device_ground, input_wave, mask_tensor, output_ptr):
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        idx = bx * cuda.blockDim.x + tx

        if idx < device_ground.shape[0]:
            val = input_wave[idx]
            mask = mask_tensor[idx]
            guard_bias = 0xF
            vibrant = (val & (~mask)) | (val & guard_bias)
            if vibrant != 0:
                output_ptr[idx] = (vibrant ^ mask) | (val & guard_bias)
            else:
                output_ptr[idx] = val & guard_bias

    @cuda.jit
    def _logic_to_resonance_kernel(d_bitmask_stream, d_quaternion_field, d_resonance_out, num_elements, base_tension):
        """
        [Device Kernel] Logic to Resonance
        분기문(If) 대신 위상 간섭(Phase Shift)으로 결론을 도출하는 하드웨어 직결 코어.
        """
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if idx >= num_elements:
            return

        mask = d_bitmask_stream[idx]

        # d_quaternion_field is Nx4 array [x, y, z, w]
        qx = d_quaternion_field[idx, 0]
        qy = d_quaternion_field[idx, 1]
        qz = d_quaternion_field[idx, 2]
        qw = d_quaternion_field[idx, 3]

        # __popcll(mask) % 2 == 0 -> 0.0f else 3.14159265f
        # numba doesn't have __popcll directly exposed in python level easily without trick,
        # so we implement a fast popcount for uint64
        m = mask
        m = m - ((m >> 1) & 0x5555555555555555)
        m = (m & 0x3333333333333333) + ((m >> 2) & 0x3333333333333333)
        m = (m + (m >> 4)) & 0x0f0f0f0f0f0f0f0f
        m = (m * 0x0101010101010101) >> 56
        popcount = m

        phase_shift = 0.0
        if popcount % 2 != 0:
            phase_shift = 3.14159265

        sin_p = math.sin(phase_shift * base_tension)
        cos_p = math.cos(phase_shift * base_tension)

        rotated_qx = qx * cos_p - qy * sin_p
        rotated_qy = qx * sin_p + qy * cos_p

        resonance_force = (rotated_qx * qx) + (rotated_qy * qy)
        d_resonance_out[idx] = resonance_force

class BitmaskRotorGate:
    def __init__(self, matrix_dimension: int):
        self.dimension = matrix_dimension
        self.ground_topology = None
        self.device_ground = None

    def upload_to_device(self):
        if HAS_NUMBA and cuda.is_available():
            self.device_ground = cuda.to_device(self.ground_topology)
        else:
            self.device_ground = self.ground_topology

    def bypass_trigger(self, input_wave_ptr, mask_tensor_ptr, output_ptr):
        """기존 레거시 바이패스"""
        if not (HAS_NUMBA and cuda.is_available()):
            guard_bias = 0xF
            for i in range(self.dimension):
                val = input_wave_ptr[i]
                mask = mask_tensor_ptr[i]
                vibrant = (val & (~mask)) | (val & guard_bias)
                if vibrant != 0:
                    output_ptr[i] = (vibrant ^ mask) | (val & guard_bias)
                else:
                    output_ptr[i] = val & guard_bias
            return

        threads_per_block = 32
        blocks_per_grid = (self.dimension + 31) // 32
        _wye_mirror_match_kernel[blocks_per_grid, threads_per_block](
            self.device_ground,
            input_wave_ptr,
            mask_tensor_ptr,
            output_ptr
        )

    def logic_to_resonance_bypass(self, bitmask_stream_ptr, quaternion_field_ptr, resonance_out_ptr, base_tension=1.0):
        """
        [Phase 2.5] 논리 분기(If)를 쿼터니언 위상 간섭으로 치환하는 CUDA 커널 트리거.
        """
        if not (HAS_NUMBA and cuda.is_available()):
            # CPU Fallback
            for i in range(self.dimension):
                mask = bitmask_stream_ptr[i]
                qx = quaternion_field_ptr[i, 0]
                qy = quaternion_field_ptr[i, 1]
                qz = quaternion_field_ptr[i, 2]
                qw = quaternion_field_ptr[i, 3]

                popcount = bin(int(mask)).count('1')
                phase_shift = 0.0 if popcount % 2 == 0 else 3.14159265

                sin_p = math.sin(phase_shift * base_tension)
                cos_p = math.cos(phase_shift * base_tension)

                rotated_qx = qx * cos_p - qy * sin_p
                rotated_qy = qx * sin_p + qy * cos_p

                resonance_force = (rotated_qx * qx) + (rotated_qy * qy)
                resonance_out_ptr[i] = resonance_force
            return

        threads_per_block = 32
        blocks_per_grid = (self.dimension + 31) // 32

        # num_elements = self.dimension
        _logic_to_resonance_kernel[blocks_per_grid, threads_per_block](
            bitmask_stream_ptr,
            quaternion_field_ptr,
            resonance_out_ptr,
            self.dimension,
            base_tension
        )


    @staticmethod
    def pack_64bit(phase_state: np.uint32, token_val: np.uint32) -> np.uint64:
        return (np.uint64(phase_state) << np.uint64(32)) | np.uint64(token_val)

    @staticmethod
    def unpack_64bit(packed_data: np.uint64):
        phase_state = np.uint32(packed_data >> np.uint64(32))
        token_val = np.uint32(packed_data & np.uint64(0xFFFFFFFF))
        return phase_state, token_val

    @staticmethod
    def create_mask(target_phase: np.uint32, rotor_shift: int) -> np.uint64:
        shifted_phase = np.uint32((target_phase << rotor_shift) | (target_phase >> (32 - rotor_shift)))
        return BitmaskRotorGate.pack_64bit(shifted_phase, np.uint32(0xFFFFFFFF))

    @staticmethod
    def project_to_hologram(packed_data: np.uint64, base_dimension: int = 128) -> np.ndarray:
        phase_state, token_val = BitmaskRotorGate.unpack_64bit(packed_data)
        guard_anchor = token_val & 0xF
        
        np.random.seed(int(phase_state ^ guard_anchor))
        curvature = 1.0 + (guard_anchor / 15.0)
        holographic_surface = np.random.normal(loc=0.0, scale=curvature, size=base_dimension)
        return holographic_surface
