"""
SpectralDescription — 픽셀인과(Pixel Causality) 구조화
======================================================
픽셀 자체가 아닌, 픽셀을 낳는 인과 원리를 기술한다.

RGB가 결합 원리로 모든 색을 표현하듯,
픽셀인과 구조를 스케일 로터화 하면
4K, 8K, 게임영상 등 어떤 해상도/매체든
동일한 원리로 재현·확장 가능하다.

핵심 순서:
  1. 원본 → SpectralDescription 추출 (원본 손대지 않음)
  2. SpectralDescription → 재현 (reconstruct)
  3. 원본 == 재현 비교대조 검증 (verify_against)
  4. 검증 통과 후에만 스케일 로터 사용 가능

4 Continuities:
  - Relationship:   픽셀 경계는 hard cut이 아닌 인과 포텐셜 필드
  - Connectivity:   인트라/인터/템포럴 인과가 ConnectivityBeam으로 결합
  - Mobility:       스케일 로터 θ는 보존된 모멘텀으로 회전
  - Informational:  이산 프레임 간 인과 보간으로 연속성 유지
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from enum import Enum
from scipy.ndimage import sobel, gaussian_filter
from scipy.fft import fft2, ifft2


# ---------------------------------------------------------------------------
# 전달함수 (θ₃): "어떤 원리로 변화하는가"
# ---------------------------------------------------------------------------

class TransferFunction(Enum):
    """스케일 로터의 θ₃ 축 — 값이 어떤 곡선으로 변화하는가"""
    LINEAR    = "linear"      # 단순 선형: Δstep = 균등
    GAMMA_22  = "gamma_2.2"   # sRGB 표준 (인간 눈 감마 보정)
    PQ        = "pq"          # HDR10 Perceptual Quantizer
    HLG       = "hlg"         # Hybrid Log-Gamma (방송 HDR)
    LOG_C     = "log_c"       # ARRI Log C (시네마 RAW)


def apply_transfer(values: np.ndarray, fn: TransferFunction) -> np.ndarray:
    """전달함수 적용: 선형 빛값 → 인코딩된 픽셀값"""
    v = np.clip(values, 0.0, 1.0)
    if fn == TransferFunction.LINEAR:
        return v
    elif fn == TransferFunction.GAMMA_22:
        # sRGB 표준 인코딩
        return np.where(v <= 0.0031308,
                        12.92 * v,
                        1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    elif fn == TransferFunction.PQ:
        # ST 2084 Perceptual Quantizer
        m1, m2 = 0.1593017578125, 78.84375
        c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
        vp = np.power(v, 1.0 / m2)
        return np.power(np.maximum(vp - c1, 0) / (c2 - c3 * vp), 1.0 / m1)
    elif fn == TransferFunction.HLG:
        a, b, c = 0.17883277, 0.28466892, 0.55991073
        return np.where(v <= 1.0 / 12.0,
                        np.sqrt(3.0 * v),
                        a * np.log(12.0 * v - b) + c)
    return v


def invert_transfer(encoded: np.ndarray, fn: TransferFunction) -> np.ndarray:
    """전달함수 역적용: 인코딩된 픽셀값 → 선형 빛값"""
    v = np.clip(encoded, 0.0, 1.0)
    if fn == TransferFunction.LINEAR:
        return v
    elif fn == TransferFunction.GAMMA_22:
        return np.where(v <= 0.04045,
                        v / 12.92,
                        np.power((v + 0.055) / 1.055, 2.4))
    elif fn == TransferFunction.PQ:
        m1, m2 = 0.1593017578125, 78.84375
        c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
        vp = np.power(v, m1)
        return np.power((c1 + c2 * vp) / (1.0 + c3 * vp), m2)
    elif fn == TransferFunction.HLG:
        a, b, c = 0.17883277, 0.28466892, 0.55991073
        return np.where(v <= 0.5,
                        v ** 2 / 3.0,
                        (np.exp((v - c) / a) + b) / 12.0)
    return v


# ---------------------------------------------------------------------------
# 인트라픽셀 인과 (IntraPixelCausality)
# "픽셀 내부: R,G,B 채널의 결합 원리"
# ---------------------------------------------------------------------------

# sRGB 표준 원색 (CIE xy 좌표)
_SRGB_PRIMARIES = np.array([
    [0.640, 0.330],  # R
    [0.300, 0.600],  # G
    [0.150, 0.060],  # B
], dtype=np.float64)

# D65 백색점
_D65_WHITE = np.array([0.3127, 0.3290], dtype=np.float64)

# sRGB → XYZ (D65) 변환 행렬 (IEC 61966-2-1)
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ)


@dataclass
class IntraPixelCausality:
    """
    픽셀 내부 인과: R,G,B 채널의 결합 원리.

    θ₁ black_point  : 어디를 "없음"으로 볼 것인가
    θ₂ white_level  : 어디를 "최대"로 볼 것인가
    θ₃ transfer_fn  : 변화가 어떤 곡선인가
    θ₄ bit_depth    : 얼마나 촘촘히 나누는가

    RGB 값이 아닌 RGB가 결합하는 방식을 기술.
    """
    primaries:    np.ndarray       # (3,2) CIE xy [R,G,B]
    white_point:  np.ndarray       # (2,)  CIE xy
    black_point:  float            # θ₁
    white_level:  float            # θ₂
    transfer_fn:  TransferFunction # θ₃
    bit_depth:    int              # θ₄
    rgb_to_xyz:   np.ndarray       # (3,3) 선형 변환 행렬
    xyz_to_rgb:   np.ndarray       # (3,3) 역변환 행렬

    @classmethod
    def from_image(cls, img_linear: np.ndarray,
                   transfer_fn: TransferFunction = TransferFunction.GAMMA_22,
                   bit_depth: int = 8) -> "IntraPixelCausality":
        """
        원본 이미지(0~1 선형)에서 인트라픽셀 인과 추출.
        히스토그램으로 black/white point를 실측.
        """
        flat = img_linear.reshape(-1, img_linear.shape[-1]) if img_linear.ndim == 3 else img_linear.ravel()
        bp = float(np.percentile(flat, 0.5))   # θ₁: 하위 0.5% = 실질적 흑점
        wp = float(np.percentile(flat, 99.5))  # θ₂: 상위 0.5% = 실질적 백점
        return cls(
            primaries   = _SRGB_PRIMARIES.copy(),
            white_point = _D65_WHITE.copy(),
            black_point = bp,
            white_level = wp,
            transfer_fn = transfer_fn,
            bit_depth   = bit_depth,
            rgb_to_xyz  = _SRGB_TO_XYZ.copy(),
            xyz_to_rgb  = _XYZ_TO_SRGB.copy(),
        )

    def normalize(self, img_linear: np.ndarray) -> np.ndarray:
        """선형 빛값을 [black_point, white_level] 범위로 정규화"""
        span = max(self.white_level - self.black_point, 1e-9)
        return np.clip((img_linear - self.black_point) / span, 0.0, 1.0)

    def denormalize(self, img_norm: np.ndarray) -> np.ndarray:
        """정규화된 값을 다시 선형 빛값으로"""
        span = self.white_level - self.black_point
        return img_norm * span + self.black_point


# ---------------------------------------------------------------------------
# 인터픽셀 인과 (InterPixelCausality)
# "픽셀 간 인과: 인접 픽셀이 어떤 원리로 연결되는가"
# ---------------------------------------------------------------------------

@dataclass
class InterPixelCausality:
    """
    픽셀 간 인과 — 텍스처·명암·경계의 생성 원리.

    핵심 설계:
    - Laplacian(I)을 인과로 저장. 포아송 방정식 ΔI = L로
      FFT 역산 시 완전 복원(오차 ~1e-14).
    - 중앙차분 기울기(dy, dx)는 경계/텍스처 기술자로만 사용.
    - DC 오프셋으로 절대 밝기 기준 보존.

    해상도 독립성:
    Laplacian 필드는 스케일링 후에도
    동일한 인과 구조(변화율의 변화율)를 유지.
    """
    # 이산 Laplacian 필드 (H, W, 3) — 채널별 변화율의 변화율
    laplacian:  np.ndarray
    # 중앙차분 기울기장 (H, W, 3, 2) — [dy, dx] 경계/텍스처 기술
    chr_grad:   np.ndarray
    # 명도 기울기장 (H, W, 2) — 경계 위상 계산용
    lum_grad:   np.ndarray
    # 경계 위상 맵 (H, W)
    edge_topo:  np.ndarray
    # 주파수 기저 (FFT 진폭) — 텍스처 스케일 불변 기술자
    freq_basis: np.ndarray
    # DC 오프셋 (H,W 평균) — 채널별 절대 밝기 기준
    dc_offset:  np.ndarray  # shape (3,)

    @classmethod
    def from_image(cls, img_norm: np.ndarray) -> "InterPixelCausality":
        """
        정규화된 이미지(H,W,3, 0~1)에서 인터픽셀 인과 추출.

        Laplacian = 2cos(2πky/H)-2 + 2cos(2πkx/W)-2 eigenvalue로
        FFT 역산 시 완전 복원.
        """
        if img_norm.ndim == 2:
            img_norm = np.stack([img_norm] * 3, axis=-1)

        def discrete_laplacian(ch: np.ndarray) -> np.ndarray:
            """이산 Laplacian (중앙차분): L[i,j] = I[i+1,j]+I[i-1,j]+I[i,j+1]+I[i,j-1] - 4*I[i,j]"""
            return (np.roll(ch, -1, axis=0) + np.roll(ch, 1, axis=0)
                  + np.roll(ch, -1, axis=1) + np.roll(ch, 1, axis=1)
                  - 4.0 * ch)

        def central_grad(ch: np.ndarray):
            """중앙차분 기울기 (텍스처·경계 기술용)"""
            dy = (np.roll(ch, -1, axis=0) - np.roll(ch, 1, axis=0)) * 0.5
            dx = (np.roll(ch, -1, axis=1) - np.roll(ch, 1, axis=1)) * 0.5
            return dy, dx

        # 채널별 Laplacian
        laps = []
        chr_grads = []
        for c in range(3):
            lap = discrete_laplacian(img_norm[..., c])
            laps.append(lap)
            dy_c, dx_c = central_grad(img_norm[..., c])
            chr_grads.append(np.stack([dy_c, dx_c], axis=-1))

        laplacian = np.stack(laps, axis=-1)      # (H,W,3)
        chr_grad  = np.stack(chr_grads, axis=-2) # (H,W,3,2)

        # 명도 기울기 (경계 위상용)
        lum = (0.2126 * img_norm[..., 0]
               + 0.7152 * img_norm[..., 1]
               + 0.0722 * img_norm[..., 2])
        dy_lum, dx_lum = central_grad(lum)
        lum_grad  = np.stack([dy_lum, dx_lum], axis=-1)
        edge_topo = np.sqrt(dy_lum**2 + dx_lum**2)

        # FFT 스펙트럼 (텍스처 기술자)
        fft_amp   = np.abs(fft2(lum))
        freq_basis = np.fft.fftshift(fft_amp)

        dc_offset = np.mean(img_norm.reshape(-1, 3), axis=0)

        return cls(
            laplacian  = laplacian,
            chr_grad   = chr_grad,
            lum_grad   = lum_grad,
            edge_topo  = edge_topo,
            freq_basis = freq_basis,
            dc_offset  = dc_offset,
        )

    def reconstruct_channel(self, lap: np.ndarray, dc: float = 0.5) -> np.ndarray:
        """
        포아송 복원: Laplacian → 이미지.
        ΔI = L  →  FFT: λ(k) * F(I) = F(L)
        이산 Laplacian eigenvalue: λ = 2cos(2πky/H)-2 + 2cos(2πkx/W)-2
        오차: ~1e-14 (부동소수점 정밀도 한계)
        """
        H, W = lap.shape
        F_lap = fft2(lap)

        ky = np.fft.fftfreq(H).reshape(-1, 1) * 2.0 * np.pi
        kx = np.fft.fftfreq(W).reshape(1, -1) * 2.0 * np.pi
        denom = 2.0 * np.cos(ky) - 2.0 + 2.0 * np.cos(kx) - 2.0
        denom[0, 0] = 1.0  # DC 분리

        F_img = F_lap / denom
        F_img[0, 0] = dc * H * W  # DC 오프셋 삽입

        return np.real(ifft2(F_img))

    def reconstruct(self, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Laplacian 인과장으로부터 이미지 재현.
        target_shape 지정 시 해당 해상도로 스케일 재현.
        """
        channels = []
        for c in range(3):
            lap = self.laplacian[..., c]
            dc  = float(self.dc_offset[c])
            ch  = self.reconstruct_channel(lap, dc=dc)
            channels.append(ch)

        img = np.stack(channels, axis=-1)
        img = np.clip(img, 0.0, 1.0)

        if target_shape is not None and target_shape != img.shape[:2]:
            img = _resize_bilinear(img, target_shape)

        return img


# ---------------------------------------------------------------------------
# 템포럴 인과 (TemporalCausality) — 영상용
# ---------------------------------------------------------------------------

@dataclass
class TemporalCausality:
    """
    프레임 간 인과: 시간축 변화 원리.
    광학 플로우가 아닌 인과적 변화 기울기장.
    """
    motion_grad:           Optional[np.ndarray] = None   # (H,W,2) 시간 기울기
    scene_change_energy:   float = 0.0                   # 씬 단절 에너지
    temporal_dc:           float = 0.0                   # 시간 DC 오프셋


# ---------------------------------------------------------------------------
# SpectralDescription — 최종 인과 구조체
# ---------------------------------------------------------------------------

@dataclass
class SpectralDescription:
    """
    픽셀인과 구조체.

    원본 데이터를 손대지 않고 원본을 낳은 인과 원리를 기술.
    reconstruct() == original 이 성립해야 스케일 로터 사용 가능.

    스케일 독립성:
    - 동일한 인과 원리(θ₁~θ₄ + 기울기장)로
    - 4K → 8K → 임의 해상도 재현 가능
    - 게임 영상이든 시네마든 동일한 원리
    """
    intra:          IntraPixelCausality
    inter:          InterPixelCausality
    temporal:       Optional[TemporalCausality] = None
    source_shape:   Tuple[int, ...] = field(default_factory=tuple)
    source_space:   str = "sRGB"

    # ------------------------------------------------------------------
    # 추출
    # ------------------------------------------------------------------

    @classmethod
    def extract(cls,
                image: np.ndarray,
                transfer_fn: TransferFunction = TransferFunction.GAMMA_22
                ) -> "SpectralDescription":
        """
        원본 이미지(uint8 또는 float)에서 픽셀인과 추출.
        원본은 절대 수정하지 않는다.

        Parameters
        ----------
        image : np.ndarray
            원본 이미지. uint8 (H,W,3) 또는 float (H,W,3) 0~1.
        transfer_fn : TransferFunction
            원본의 전달함수 (기본 sRGB gamma 2.2).
        """
        orig = image.copy()  # 원본 복사본 — 원본 자체는 손대지 않음
        source_shape = orig.shape

        # 1. uint8 → float 정규화
        if orig.dtype == np.uint8:
            img_encoded = orig.astype(np.float64) / 255.0
        else:
            img_encoded = orig.astype(np.float64)

        # 2. 전달함수 역적용: 인코딩값 → 선형 빛값
        img_linear = invert_transfer(img_encoded, transfer_fn)

        # 3. IntraPixelCausality 추출 (결합 원리)
        intra = IntraPixelCausality.from_image(img_linear, transfer_fn)

        # 4. 정규화: [black_point, white_level] → [0, 1]
        img_norm = intra.normalize(img_linear)

        # 5. InterPixelCausality 추출 (기울기 인과장)
        inter = InterPixelCausality.from_image(img_norm)

        return cls(
            intra        = intra,
            inter        = inter,
            source_shape = source_shape,
            source_space = "sRGB",
        )

    # ------------------------------------------------------------------
    # 재현
    # ------------------------------------------------------------------

    def reconstruct(self,
                    target_shape: Optional[Tuple[int, int]] = None
                    ) -> np.ndarray:
        """
        인과 원리로부터 이미지 재현.
        target_shape 지정 시 해당 해상도로 스케일 재현.
        원본 해상도 미지정 시 원본 크기로 재현.
        """
        shape = target_shape or self.source_shape[:2]

        # 기울기장 → 이미지 (포아송 복원)
        img_norm = self.inter.reconstruct(target_shape=shape)

        # 역정규화: [0,1] → [black_point, white_level]
        img_linear = self.intra.denormalize(img_norm)

        # 전달함수 적용: 선형 빛값 → 인코딩값
        img_encoded = apply_transfer(img_linear, self.intra.transfer_fn)

        # float → uint8
        img_uint8 = np.clip(img_encoded * 255.0, 0, 255).astype(np.uint8)
        return img_uint8

    # ------------------------------------------------------------------
    # 비교대조 검증 — 핵심
    # ------------------------------------------------------------------

    def verify_against(self,
                       original: np.ndarray,
                       tolerance_8bit: float = 2.0
                       ) -> Dict[str, Any]:
        """
        비교대조 검증: SpectralDescription이 원본을 재현할 수 있는가.

        검증은 선형 빛값(Linear Light) 공간에서 수행.
        uint8 → 전달함수 역적용 → 선형 공간 비교.
        이렇게 해야 gamma 비선형성이 오차를 왜곡하지 않음.

        Parameters
        ----------
        original : np.ndarray
            원본 이미지 (uint8 H,W,3).
        tolerance_8bit : float
            허용 최대 오차 (0~255 스케일 기준).
        """
        reconstructed = self.reconstruct(target_shape=original.shape[:2])

        # 선형 공간 비교 (더 공정한 인과 검증)
        orig_lin  = invert_transfer(original.astype(np.float64) / 255.0,
                                    self.intra.transfer_fn)
        recon_lin = invert_transfer(reconstructed.astype(np.float64) / 255.0,
                                    self.intra.transfer_fn)

        abs_err = np.abs(orig_lin - recon_lin) * 255.0  # 0~255 스케일로 표현
        max_err  = float(np.max(abs_err))
        mean_err = float(np.mean(abs_err))
        psnr     = _compute_psnr(orig_lin * 255.0, recon_lin * 255.0)

        passes = max_err <= tolerance_8bit

        channel_errors = {
            "R": float(np.mean(abs_err[..., 0])),
            "G": float(np.mean(abs_err[..., 1])),
            "B": float(np.mean(abs_err[..., 2])),
        }

        return {
            "passes":          passes,
            "max_error_8bit":  max_err,
            "mean_error_8bit": mean_err,
            "psnr_db":         psnr,
            "channel_errors":  channel_errors,
            "tolerance":       tolerance_8bit,
            "source_shape":    self.source_shape,
            "reconstructed":   reconstructed,
        }

    # ------------------------------------------------------------------
    # 스케일 로터 — 검증 통과 후 사용 가능
    # ------------------------------------------------------------------

    def scale_to(self,
                 new_shape: Tuple[int, int],
                 original: Optional[np.ndarray] = None,
                 verify_first: bool = True
                 ) -> np.ndarray:
        """
        스케일 로터: 동일한 인과 원리로 다른 해상도에서 재현.
        verify_first=True 시 원본 검증 통과 후에만 스케일링.

        Parameters
        ----------
        new_shape : (H, W)
            목표 해상도.
        original : np.ndarray, optional
            검증용 원본. verify_first=True 시 필요.
        verify_first : bool
            True면 검증 통과 후에만 스케일링. False면 즉시 스케일링.
        """
        if verify_first:
            if original is None:
                raise ValueError(
                    "scale_to(verify_first=True) 는 원본 이미지가 필요합니다. "
                    "원본 재현도 못하면서 스케일을 논할 수 없습니다."
                )
            result = self.verify_against(original, tolerance_8bit=3.0)
            if not result["passes"]:
                raise RuntimeError(
                    f"비교대조 검증 실패 — 원본 재현 불가.\n"
                    f"  max_error = {result['max_error_8bit']:.2f} (허용: {result['tolerance']})\n"
                    f"  PSNR      = {result['psnr_db']:.2f} dB\n"
                    "  SpectralDescription을 먼저 개선하세요."
                )

        return self.reconstruct(target_shape=new_shape)

    def report(self, original: Optional[np.ndarray] = None) -> str:
        """인과 구조 요약 리포트"""
        lines = [
            "=== SpectralDescription Report ===",
            f"원본 형태  : {self.source_shape}",
            f"색공간     : {self.source_space}",
            "",
            "[IntraPixelCausality — 결합 원리]",
            f"  θ₁ black_point  : {self.intra.black_point:.6f}",
            f"  θ₂ white_level  : {self.intra.white_level:.6f}",
            f"  θ₃ transfer_fn  : {self.intra.transfer_fn.value}",
            f"  θ₄ bit_depth    : {self.intra.bit_depth} bit",
            "",
            "[InterPixelCausality — 공간 인과]",
            f"  lum_grad shape  : {self.inter.lum_grad.shape}",
            f"  chr_grad shape  : {self.inter.chr_grad.shape}",
            f"  edge_topo max   : {np.max(self.inter.edge_topo):.4f}",
            f"  dc_offset (RGB) : {self.inter.dc_offset}",
        ]

        if original is not None:
            v = self.verify_against(original)
            lines += [
                "",
                "[비교대조 검증]",
                f"  결과     : {'✅ PASS' if v['passes'] else '❌ FAIL'}",
                f"  max_err  : {v['max_error_8bit']:.3f} / 255",
                f"  mean_err : {v['mean_error_8bit']:.3f} / 255",
                f"  PSNR     : {v['psnr_db']:.2f} dB",
                f"  채널별   : R={v['channel_errors']['R']:.3f} "
                            f"G={v['channel_errors']['G']:.3f} "
                            f"B={v['channel_errors']['B']:.3f}",
            ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------

def _resize_bilinear(img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    """바이리니어 보간으로 이미지 크기 변환 (scipy 없이)"""
    from scipy.ndimage import zoom
    H_new, W_new = new_shape
    H_old, W_old = img.shape[:2]
    zy = H_new / H_old
    zx = W_new / W_old
    if img.ndim == 3:
        return zoom(img, (zy, zx, 1.0), order=1)
    return zoom(img, (zy, zx), order=1)


def _compute_psnr(orig: np.ndarray, recon: np.ndarray,
                  max_val: float = 255.0) -> float:
    """PSNR (dB) 계산"""
    mse = float(np.mean((orig - recon) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


# ---------------------------------------------------------------------------
# 편의 함수
# ---------------------------------------------------------------------------

def extract_and_verify(image: np.ndarray,
                       tolerance_8bit: float = 2.0,
                       verbose: bool = True
                       ) -> Tuple["SpectralDescription", Dict[str, Any]]:
    """
    원본 이미지에서 SpectralDescription 추출 후 즉시 비교대조 검증.

    Returns
    -------
    desc : SpectralDescription
    result : dict  (verify_against 결과)
    """
    desc   = SpectralDescription.extract(image)
    result = desc.verify_against(image, tolerance_8bit=tolerance_8bit)
    if verbose:
        print(desc.report(original=image))
    return desc, result
