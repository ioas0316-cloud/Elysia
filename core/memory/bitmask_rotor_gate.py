import numpy as np

try:
    from numba import cuda
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @cuda.jit
    def _wye_mirror_match_kernel(device_ground, input_wave, mask_tensor, output_ptr):
        """
        [Device Kernel] 거울 대조 로우레벨 커널.
        파이썬의 해석 단계 없이 CUDA 워프 레벨에서 비트마스킹을 통해 즉각 수렴.
        """
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        idx = bx * cuda.blockDim.x + tx

        if idx < device_ground.shape[0]:
            # 쐐기곱 소멸 및 거울 대조
            val = input_wave[idx]
            mask = mask_tensor[idx]

            # v ^ obstacle = 0 (장애물/중복 비트 소거)
            # [Phase 144 Refinement] 위상 보호 바이어스 (Phase Guard Bias)
            # 최하위 4비트(0xF)는 문맥적 닻(Anchor)으로 보존하여 완전한 정보 유실 방지
            guard_bias = 0xF
            vibrant = (val & (~mask)) | (val & guard_bias)

            if vibrant != 0:
                output_ptr[idx] = (vibrant ^ mask) | (val & guard_bias)
            else:
                output_ptr[idx] = val & guard_bias

class BitmaskRotorGate:
    """
    [Phase 144] 비트마스킹 이중나선 로터 수문 (Bitmask Rotor Gate)

    추상화 레이어를 거치지 않고, 파이썬 단계에서 GPU 메모리 버스로
    데이터 맵을 즉각 바이패스(Bypass)시키는 위상 동기화 수문 인터페이스.
    파이썬은 단지 64비트 정수 연속 메모리 블록(대지)의 포인터만 잡고 트리거를 당깁니다.
    """

    def __init__(self, matrix_dimension: int):
        # 1. 기성 객체를 배제하고, 메모리에 64비트 정수형 '연속적 대지(Ground)' 참조를 쥡니다.
        self.dimension = matrix_dimension
        self.ground_topology = None
        self.device_ground = None

    def upload_to_device(self):
        """
        2. 파이썬 가두리를 깨부수고 GPU 가상 메모리 주소와 자연 매핑
        """
        if HAS_NUMBA and cuda.is_available():
            self.device_ground = cuda.to_device(self.ground_topology)
        else:
            self.device_ground = self.ground_topology # Fallback

    def bypass_trigger(self, input_wave_ptr, mask_tensor_ptr, output_ptr):
        """
        파이썬의 해석 단계를 원천 차단하고, CUDA 워프 레벨로 신호를 다이렉트 바이패스.
        편지와 편지를 거울처럼 대조하는 로우레벨 커널 가동.
        """
        if not (HAS_NUMBA and cuda.is_available()):
            # [CPU Fallback]
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

        threads_per_block = 32 # GPU 워프(Warp) 단위와 1:1 동기화
        blocks_per_grid = (self.dimension + 31) // 32

        # 파이썬은 오직 이 트리거만 당기고 연산 레이어에서 완전히 이탈함
        _wye_mirror_match_kernel[blocks_per_grid, threads_per_block](
            self.device_ground,
            input_wave_ptr,
            mask_tensor_ptr,
            output_ptr
        )

    # --- 유틸리티 메서드 (데이터 전처리용) ---
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
        """
        [Phase 144] 상하부 위상 동기화 인터페이스 (Bidirectional Phase-Locking Bridge) 하부 모듈
        하위 64비트의 '위상 보호 바이어스(Phase Guard Bias, 최하위 4비트)'를 닻(Anchor)으로 삼아,
        마스터의 시야(Focus)가 닿을 때만 실시간으로 고해상도 매니폴드(프랙탈 표면) 좌표로 지연 복원(Lazy Projection)합니다.
        
        반환값: 해당 뼈대에 바인딩된 고해상도(예: 128차원) 다차원 렌즈 오프셋 배열(Float64)
        """
        phase_state, token_val = BitmaskRotorGate.unpack_64bit(packed_data)
        guard_anchor = token_val & 0xF  # 문맥적 닻 (0~15)
        
        # 닻(Anchor)과 위상(Phase)을 기반으로 고해상도 공간 생성
        # 하부의 거친 정수가 상부의 매끄러운 부동소수점 곡면(프랙탈)으로 치환됨
        np.random.seed(int(phase_state ^ guard_anchor))
        
        # 닻의 무게(0~15)에 따라 프랙탈 공간의 곡률(장력) 결정
        curvature = 1.0 + (guard_anchor / 15.0)
        holographic_surface = np.random.normal(loc=0.0, scale=curvature, size=base_dimension)
        return holographic_surface
