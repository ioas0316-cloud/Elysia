Shader "CausalEngine/TerrainDisplacementHLSL"
{
    Properties
    {
        _MainTex ("Terrain Base Texture", 2D) = "white" {}
        _TensionFieldTexture ("Causal Tension Field Texture (V_t)", 2D) = "black" {}
        _DisplacementHeight ("Displacement Height Multiplier", Float) = 4.0
        _DistortionSpeed ("Anomaly Wave Speed", Float) = 3.0
        [HDR] _AnomalyColor ("Anomaly Glow Color", Color) = (2.0, 0.3, 0.1, 1.0)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        Pass
        {
            Name "ForwardLit"
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS   : POSITION;
                float3 normalOS     : NORMAL;
                float2 uv           : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS   : SV_POSITION;
                float2 uv           : TEXCOORD0;
                float3 positionWS   : TEXCOORD1;
                float tensionValue  : TEXCOORD2;
            };

            Texture2D _MainTex;
            SamplerState sampler_MainTex;

            Texture2D _TensionFieldTexture;
            SamplerState sampler_TensionFieldTexture;

            CBUFFER_START(UnityPerMaterial)
                float4 _MainTex_ST;
                float _DisplacementHeight;
                float _DistortionSpeed;
                float4 _AnomalyColor;
            CBUFFER_END

            Varyings vert(Attributes input)
            {
                Varyings output;

                output.uv = TRANSFORM_TEX(input.uv, _MainTex_ST);
                float3 positionWS = TransformObjectToWorld(input.positionOS.xyz);
                float3 normalWS = TransformObjectToWorldNormal(input.normalOS);

                // Sample V_t tension value directly at LOD 0
                float vtTension = _TensionFieldTexture.SampleLevel(sampler_TensionFieldTexture, output.uv, 0).r;
                output.tensionValue = vtTension;

                // Vertex normal displacement and wave modulation
                float wave = sin(_Time.y * _DistortionSpeed + positionWS.x * 1.5 + positionWS.z * 1.5);
                float3 displacementOffset = normalWS * (vtTension * _DisplacementHeight * (0.7 + 0.3 * wave));

                positionWS += displacementOffset;
                output.positionWS = positionWS;
                output.positionCS = TransformWorldToHClip(positionWS);

                return output;
            }

            float4 frag(Varyings input) : SV_Target
            {
                float4 baseColor = _MainTex.Sample(sampler_MainTex, input.uv);
                float vtTension = saturate(input.tensionValue);

                float3 blendedColor = lerp(baseColor.rgb, _AnomalyColor.rgb, vtTension * 0.6);

                // Non-linear emission bloom: V_t^2.5
                float emissionIntensity = pow(vtTension, 2.5) * 2.0;
                float3 finalColor = blendedColor + (_AnomalyColor.rgb * emissionIntensity);

                return float4(finalColor, baseColor.a);
            }
            ENDHLSL
        }
    }
}
