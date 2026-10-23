Shader "Hidden/Custom/PhaseRuptureShift"
{
    Properties
    {
        _MainTex ("Main Texture", 2D) = "white" {}
        _RuptureIntensity ("Rupture Intensity", Range(0, 1)) = 0.0
        _GlitchFrequency ("Glitch Frequency", Float) = 25.0
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        LOD 100
        ZWrite Off ZTest Always Cull Off

        Pass
        {
            Name "PhaseRupturePass"

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"

            struct Attributes
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);

            CBUFFER_START(UnityPerMaterial)
                float _RuptureIntensity;
                float _GlitchFrequency;
            CBUFFER_END

            float Hash12(float2 p)
            {
                float3 p3 = frac(float3(p.xyx) * 0.1031);
                p3 += dot(p3, p3.yzx + 33.33);
                return frac((p3.x + p3.y) * p3.z);
            }

            Varyings Vert(Attributes input)
            {
                Varyings output;
                output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
                output.uv = GetFullScreenTriangleTexCoord(input.vertexID);
                return output;
            }

            float4 Frag(Varyings input) : SV_Target
            {
                float2 uv = input.uv;
                float time = _Time.y * 10.0;

                float lineNoise = Hash12(float2(floor(uv.y * _GlitchFrequency), floor(time)));
                float isGlitchLine = step(0.85, lineNoise);

                float uvOffset = (Hash12(float2(time, uv.y)) - 0.5) * 0.1 * _RuptureIntensity * isGlitchLine;
                float2 distortedUV = uv + float2(uvOffset, 0.0);

                float splitAmount = 0.03 * _RuptureIntensity * (1.0 + isGlitchLine * 2.0);
                float r = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, distortedUV + float2(splitAmount, 0.0)).r;
                float g = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, distortedUV).g;
                float b = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, distortedUV - float2(splitAmount, 0.0)).b;

                float3 baseColor = float3(r, g, b);

                float2 centerUV = uv - 0.5;
                float dist = length(centerUV);
                float pulse = (sin(_Time.y * 15.0) * 0.5 + 0.5) * _RuptureIntensity;
                float vignette = smoothstep(0.3, 0.8, dist) * pulse;

                float3 warningRed = float3(1.0, 0.05, 0.1);
                baseColor = lerp(baseColor, warningRed, vignette * 0.7);

                float wave = sin(uv.y * 50.0 + _Time.y * 20.0) * _RuptureIntensity;
                if (wave > 0.75)
                {
                    baseColor = 1.0 - baseColor;
                }

                return float4(baseColor, 1.0);
            }
            ENDHLSL
        }
    }
}
