"""
Test: SpectralDescription — 비교대조 검증 루프
================================================
핵심 검증 순서:
  1. 원본 → SpectralDescription 추출
  2. SpectralDescription → 재현
  3. 원본 == 재현 비교대조 (오차 허용치 이내)
  4. 검증 통과 후 스케일 로터 사용

원본을 재현조차 못하면 스케일을 논할 수 없다.
"""

import pytest
import numpy as np
import os
from PIL import Image

from core.ingestion.spectral_description import (
    SpectralDescription,
    TransferFunction,
    IntraPixelCausality,
    InterPixelCausality,
    extract_and_verify,
    apply_transfer,
    invert_transfer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_gradient_image():
    """합성 그라디언트 이미지 — 수학적으로 완전한 인과 구조"""
    H, W = 64, 64
    y = np.linspace(0, 1, H, dtype=np.float32)
    x = np.linspace(0, 1, W, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    R = yy
    G = xx
    B = 1.0 - (yy + xx) / 2.0
    img = np.stack([R, G, B], axis=-1)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


@pytest.fixture
def real_apple_image():
    """실제 사과 이미지 (존재할 때만)"""
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "core", "ingestion", "apple_test.jpg"
    ))
    if not os.path.exists(path):
        pytest.skip(f"실제 이미지 없음: {path}")
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


@pytest.fixture
def uniform_color_image():
    """단색 이미지 — 가장 단순한 인과 구조"""
    img = np.full((32, 32, 3), fill_value=[128, 64, 200], dtype=np.uint8)
    return img


# ---------------------------------------------------------------------------
# 1. 전달함수 가역성 검증 (θ₃ 정확성)
# ---------------------------------------------------------------------------

class TestTransferFunction:
    """전달함수 적용 → 역적용이 항등(identity)인가"""

    @pytest.mark.parametrize("fn", [
        TransferFunction.LINEAR,
        TransferFunction.GAMMA_22,
        TransferFunction.HLG,
    ])
    def test_transfer_invertible(self, fn):
        """f⁻¹(f(x)) ≈ x 이어야 한다"""
        x = np.linspace(0.0, 1.0, 1000, dtype=np.float64)
        encoded = apply_transfer(x, fn)
        decoded = invert_transfer(encoded, fn)
        max_err = float(np.max(np.abs(x - decoded)))
        assert max_err < 1e-6, (
            f"{fn.value}: 전달함수 역가역성 실패. max_err={max_err:.2e}"
        )


# ---------------------------------------------------------------------------
# 2. IntraPixelCausality 추출 검증
# ---------------------------------------------------------------------------

class TestIntraPixelCausality:

    def test_black_white_point_range(self, synthetic_gradient_image):
        """추출된 θ₁, θ₂가 원본 범위 내에 있어야 한다"""
        img_f = synthetic_gradient_image.astype(np.float64) / 255.0
        img_linear = invert_transfer(img_f, TransferFunction.GAMMA_22)
        intra = IntraPixelCausality.from_image(img_linear)

        assert 0.0 <= intra.black_point < intra.white_level <= 1.0, (
            f"black={intra.black_point:.4f}, white={intra.white_level:.4f}"
        )

    def test_normalize_denormalize_identity(self, synthetic_gradient_image):
        """normalize → denormalize 가 항등이어야 한다"""
        img_f = synthetic_gradient_image.astype(np.float64) / 255.0
        img_linear = invert_transfer(img_f, TransferFunction.GAMMA_22)
        intra = IntraPixelCausality.from_image(img_linear)

        normalized = intra.normalize(img_linear)
        recovered  = intra.denormalize(normalized)
        max_err = float(np.max(np.abs(img_linear - recovered)))
        assert max_err < 1e-10, f"정규화 가역성 실패. max_err={max_err:.2e}"

    def test_rgb_to_xyz_matrix_determinant(self, synthetic_gradient_image):
        """RGB→XYZ 행렬이 가역적이어야 한다 (det ≠ 0)"""
        img_f = synthetic_gradient_image.astype(np.float64) / 255.0
        img_linear = invert_transfer(img_f, TransferFunction.GAMMA_22)
        intra = IntraPixelCausality.from_image(img_linear)

        det = np.linalg.det(intra.rgb_to_xyz)
        assert abs(det) > 1e-6, f"RGB→XYZ 행렬이 특이행렬. det={det:.6f}"


# ---------------------------------------------------------------------------
# 3. InterPixelCausality 포아송 복원 검증
# ---------------------------------------------------------------------------

class TestInterPixelCausality:

    def test_gradient_shape(self, synthetic_gradient_image):
        """Laplacian 및 기울기장 형태가 원본과 일치해야 한다"""
        img_f = synthetic_gradient_image.astype(np.float64) / 255.0
        inter = InterPixelCausality.from_image(img_f)
        H, W = img_f.shape[:2]
        assert inter.lum_grad.shape  == (H, W, 2)
        assert inter.chr_grad.shape  == (H, W, 3, 2)
        assert inter.edge_topo.shape == (H, W)
        assert inter.laplacian.shape == (H, W, 3)

    def test_poisson_reconstruction_uniform(self, uniform_color_image):
        """단색 이미지: 기울기=0, 복원이 DC 오프셋과 일치해야 한다"""
        img_f = uniform_color_image.astype(np.float64) / 255.0
        inter = InterPixelCausality.from_image(img_f)

        # 단색이면 기울기는 거의 0
        assert np.max(np.abs(inter.lum_grad)) < 0.1, \
            "단색 이미지의 기울기가 0이어야 한다"

        recon = inter.reconstruct()
        assert recon.shape == img_f.shape
        mean_err = float(np.mean(np.abs(recon - img_f)))
        assert mean_err < 0.05, f"단색 복원 평균 오차: {mean_err:.4f}"

    def test_dc_offset_matches_image_mean(self, synthetic_gradient_image):
        """DC 오프셋이 채널 평균과 근사해야 한다"""
        img_f = synthetic_gradient_image.astype(np.float64) / 255.0
        inter = InterPixelCausality.from_image(img_f)
        expected_dc = np.mean(img_f.reshape(-1, 3), axis=0)
        err = np.abs(inter.dc_offset - expected_dc)
        assert np.all(err < 0.01), f"DC 오프셋 불일치: {err}"


# ---------------------------------------------------------------------------
# 4. 핵심: SpectralDescription 비교대조 검증 루프
# ---------------------------------------------------------------------------

class TestSpectralDescriptionVerification:
    """
    원본 → 추출 → 재현 → 비교대조
    이것이 통과해야 스케일 로터를 쓸 수 있다.
    """

    def test_extract_returns_spectral_description(self, synthetic_gradient_image):
        """추출이 SpectralDescription을 반환해야 한다"""
        desc = SpectralDescription.extract(synthetic_gradient_image)
        assert isinstance(desc, SpectralDescription)
        assert desc.source_shape == synthetic_gradient_image.shape

    def test_reconstruct_returns_same_shape(self, synthetic_gradient_image):
        """재현 결과가 원본과 동일 형태여야 한다"""
        desc = SpectralDescription.extract(synthetic_gradient_image)
        recon = desc.reconstruct()
        assert recon.shape == synthetic_gradient_image.shape
        assert recon.dtype == np.uint8

    def test_verify_against_reports_all_fields(self, synthetic_gradient_image):
        """verify_against가 필요한 모든 필드를 반환해야 한다"""
        desc = SpectralDescription.extract(synthetic_gradient_image)
        result = desc.verify_against(synthetic_gradient_image)

        for key in ["passes", "max_error_8bit", "mean_error_8bit",
                    "psnr_db", "channel_errors", "tolerance", "reconstructed"]:
            assert key in result, f"누락된 필드: {key}"

    def test_gradient_image_psnr_acceptable(self, synthetic_gradient_image):
        """
        합성 그라디언트: 포아송 복원 PSNR이 20dB 이상이어야 한다.
        (완벽한 재현은 ∞ dB, 20dB = 지각적으로 유사한 수준)
        """
        desc, result = extract_and_verify(
            synthetic_gradient_image, tolerance_8bit=10.0, verbose=True
        )
        assert result["psnr_db"] > 20.0, (
            f"PSNR 불충분: {result['psnr_db']:.2f} dB\n"
            f"max_err={result['max_error_8bit']:.2f}"
        )

    def test_original_not_mutated(self, synthetic_gradient_image):
        """SpectralDescription 추출 과정에서 원본이 변경되면 안 된다"""
        original_copy = synthetic_gradient_image.copy()
        _ = SpectralDescription.extract(synthetic_gradient_image)
        assert np.array_equal(synthetic_gradient_image, original_copy), \
            "원본 이미지가 추출 과정에서 변경되었습니다 — 인과 무결성 위반"

    def test_uniform_color_high_psnr(self, uniform_color_image):
        """단색 이미지: 재현이 거의 완벽해야 한다 (PSNR > 30dB)"""
        desc, result = extract_and_verify(
            uniform_color_image, tolerance_8bit=5.0, verbose=True
        )
        assert result["psnr_db"] > 30.0, (
            f"단색 복원 PSNR 불충분: {result['psnr_db']:.2f} dB"
        )

    def test_real_image_when_available(self, real_apple_image):
        """실제 이미지: 추출→재현→검증이 오류 없이 완료되어야 한다"""
        desc = SpectralDescription.extract(real_apple_image)
        result = desc.verify_against(real_apple_image)

        # 실제 이미지는 PSNR > 15dB 이상이어야 기본 인과 구조 포착한 것
        assert result["psnr_db"] > 15.0, (
            f"실제 이미지 PSNR 불충분: {result['psnr_db']:.2f} dB\n"
            "SpectralDescription이 이 이미지의 인과 구조를 충분히 포착하지 못했습니다."
        )
        print(desc.report(original=real_apple_image))


# ---------------------------------------------------------------------------
# 5. 스케일 로터 검증 (검증 통과 후에만 가능)
# ---------------------------------------------------------------------------

class TestScaleRotor:

    def test_scale_up_produces_larger_image(self, synthetic_gradient_image):
        """스케일 업: 더 큰 해상도가 생성되어야 한다"""
        desc = SpectralDescription.extract(synthetic_gradient_image)
        H, W = synthetic_gradient_image.shape[:2]
        # tolerance 3.0: gamma 2.2 경계 반올림 오차 허용 (인간 지각 임계 이하)
        scaled = desc.scale_to(
            new_shape=(H * 2, W * 2),
            original=synthetic_gradient_image,
            verify_first=True,
        )
        assert scaled.shape[:2] == (H * 2, W * 2), \
            f"스케일 업 실패: {scaled.shape[:2]} != {(H*2, W*2)}"

    def test_scale_requires_verification(self, synthetic_gradient_image):
        """검증 없이(verify_first=False) 스케일링은 원본 불필요"""
        desc = SpectralDescription.extract(synthetic_gradient_image)
        H, W = synthetic_gradient_image.shape[:2]
        # original 없이도 verify_first=False면 동작해야 함
        scaled = desc.scale_to(
            new_shape=(H * 2, W * 2),
            original=None,
            verify_first=False
        )
        assert scaled.shape[:2] == (H * 2, W * 2)

    def test_scale_verify_first_raises_without_original(self, synthetic_gradient_image):
        """verify_first=True인데 original 없으면 ValueError"""
        desc = SpectralDescription.extract(synthetic_gradient_image)
        with pytest.raises(ValueError, match="원본 이미지가 필요"):
            desc.scale_to((128, 128), original=None, verify_first=True)

    def test_same_causal_principle_different_scale(self, synthetic_gradient_image):
        """
        동일한 SpectralDescription으로 4K 크기와 원본 크기를 재현.
        인과 원리는 같되 해상도만 다르다.
        """
        desc = SpectralDescription.extract(synthetic_gradient_image)
        H, W = synthetic_gradient_image.shape[:2]

        recon_orig  = desc.reconstruct(target_shape=(H,     W    ))
        recon_large = desc.reconstruct(target_shape=(H * 4, W * 4))
        recon_small = desc.reconstruct(target_shape=(H // 2, W // 2))

        assert recon_orig.shape[:2]  == (H, W)
        assert recon_large.shape[:2] == (H * 4, W * 4)
        assert recon_small.shape[:2] == (H // 2, W // 2)

        # 모두 유효한 uint8 이미지여야 함
        for r in [recon_orig, recon_large, recon_small]:
            assert r.dtype == np.uint8
            assert np.all(r >= 0) and np.all(r <= 255)


# ---------------------------------------------------------------------------
# 6. 리포트 출력
# ---------------------------------------------------------------------------

class TestReport:

    def test_report_without_original(self, synthetic_gradient_image):
        """원본 없이도 리포트가 생성되어야 한다"""
        desc = SpectralDescription.extract(synthetic_gradient_image)
        report = desc.report()
        assert "SpectralDescription Report" in report
        assert "θ₁" in report and "θ₃" in report

    def test_report_with_original_contains_verification(self, synthetic_gradient_image):
        """원본과 함께 리포트하면 검증 결과가 포함되어야 한다"""
        desc = SpectralDescription.extract(synthetic_gradient_image)
        report = desc.report(original=synthetic_gradient_image)
        assert "비교대조 검증" in report
        assert "PSNR" in report
