"""
Asura Causal Nexus GLSL Shader Definitions & WebGL2/WebGPU Optimizations

Provides raw GLSL shader strings (standard and WebGL2 branchless LUT-optimized)
along with uniform layout specifications for hardware direct-state causality.
"""

# 1. Standard GLSL Fragment Shader
ASURA_RED_BLACK_FRAGMENT_SHADER_STANDARD = """#version 330 core

in vec2 v_TexCoord;
out vec4 FragColor;

// 1. 유니폼 파라미터 (Nexus 렌더링 파이프라인 연동)
uniform sampler2D u_Texture;           // 메인 스크린 버퍼 또는 도트 스프라이트 텍스처
uniform bool u_EnableRedBlackPalette;  // AsuraCheonmu 적흑 반전 토글 (color_inversion)
uniform float u_AfterimageAlpha;       // 잔상 노드 감쇄율 (0.0 ~ 1.0)
uniform float u_Contrast;              // 명암 대비 강도 (기본값: 2.0~3.0)

// 광도 계산용 표준 가중치 (Rec. 601)
const vec3 LUMINANCE_WEIGHTS = vec3(0.299, 0.587, 0.114);

void main() {
    // 2. 원본 텍스처 컬러 샘플링
    vec4 texColor = texture(u_Texture, v_TexCoord);

    // 알파값이 너무 낮은 픽셀 버림 (Early Exit 성능 최적화)
    if (texColor.a < 0.01) {
        discard;
    }

    vec3 finalRgb = texColor.rgb;

    // 3. RED_BLACK_PALETTE 셰이더 연산
    if (u_EnableRedBlackPalette) {
        // [A] 휘도 산출 (Grayscale 전환)
        float luminance = dot(texColor.rgb, LUMINANCE_WEIGHTS);

        // [B] 고대비 S-Curve 맵핑 (검은색과 붉은색 영역을 선명하게 분리)
        luminance = clamp((luminance - 0.5) * u_Contrast + 0.5, 0.0, 1.0);

        // [C] 칠흑(Pure Black)과 선혈색(Crimson Red) 간 선형 보간 (mix)
        vec3 crimsonRed = vec3(0.9, 0.02, 0.05);
        vec3 deepBlack = vec3(0.01, 0.0, 0.02);

        finalRgb = mix(deepBlack, crimsonRed, luminance);
    }

    // 4. 잔상 감쇄 (Afterimage Alpha Decay) 및 최종 출력
    float finalAlpha = texColor.a * u_AfterimageAlpha;
    FragColor = vec4(finalRgb, finalAlpha);
}
"""

# 2. Optimized WebGL2 / WebGPU High-Performance GLSL Fragment Shader
ASURA_RED_BLACK_FRAGMENT_SHADER_OPTIMIZED = """#version 330 core
precision mediump float; // 웹 환경 메모리 대역폭 절약을 위한 mediump 지정

in vec2 v_TexCoord;
out vec4 FragColor;

// UBO 블록으로 유니폼 묶음 전송 (API 바인딩 오버헤드 O(1)화)
layout (std140) uniform AsuraNexusBlock {
    float u_RedBlackToggle; // 0.0 or 1.0
    float u_AfterimageAlpha;
    float u_Contrast;
};

uniform sampler2D u_Texture;
uniform sampler1D u_PaletteLUT; // 적흑 반전 연산을 미리 구워둔 1D LUT 텍스처

const vec3 LUMINANCE_WEIGHTS = vec3(0.299, 0.587, 0.114);

void main() {
    vec4 texColor = texture(u_Texture, v_TexCoord);

    // [최적화 1] 1D LUT 샘플링으로 S-Curve 및 고대비 연산 대체
    float luminance = dot(texColor.rgb, LUMINANCE_WEIGHTS);
    vec3 redBlackRgb = texture(u_PaletteLUT, luminance).rgb;

    // [최적화 2] Branchless mix: if 조건문 없이 Toggle 값으로 바로 보간
    vec3 finalRgb = mix(texColor.rgb, redBlackRgb, u_RedBlackToggle);

    // [최적화 3] Discard 없이 알파 곱셈으로 안전하게 출력
    FragColor = vec4(finalRgb, texColor.a * u_AfterimageAlpha);
}
"""


def get_asura_shader_code(optimized: bool = True) -> str:
    """Returns standard or optimized GLSL fragment shader string."""
    return ASURA_RED_BLACK_FRAGMENT_SHADER_OPTIMIZED if optimized else ASURA_RED_BLACK_FRAGMENT_SHADER_STANDARD
