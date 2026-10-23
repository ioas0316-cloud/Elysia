Shader "Hidden/CausalEngine/PostProcessAnomaly"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        LOD 100
        ZTest Always ZWrite Off Cull Off

        Pass
        {
            Name "PostProcessAnomalyPass"
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareOpaqueTexture.hlsl"

            struct Attributes
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv         : TEXCOORD0;
            };

            Texture2D _TensionFieldTexture;
            SamplerState sampler_TensionFieldTexture;

            CBUFFER_START(UnityPerMaterial)
                float _RefractionStrength; // Screen refraction vector magnitude
                float _ChromaticOffset;   // Chromatic aberration split width
                float _TearThreshold;     // Pixel tear activation limit (V_critical)
                float _TearAmount;        // Slice displacement amount
            CBUFFER_END

            Varyings Vert(Attributes input)
            {
                Varyings output;
                output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
                output.uv = GetFullScreenTriangleTexCoord(input.vertexID);
                return output;
            }

            float PseudoRandom(float2 st)
            {
                return frac(sin(dot(st.xy, float2(12.9898, 78.233))) * 43758.5453123);
            }

            float4 Frag(Varyings input) : SV_Target
            {
                float2 uv = input.uv;

                float vt = _TensionFieldTexture.Sample(sampler_TensionFieldTexture, uv).r;

                // Screen Space Refraction vector via spatial gradient (ddx, ddy)
                float2 vtGradient = float2(ddx(vt), ddy(vt));
                float2 refractedUV = uv + (vtGradient * _RefractionStrength * vt);

                // Horizontal Slice Glitch Tearing for V_t > V_critical
                if (vt > _TearThreshold)
                {
                    float blockY = floor(uv.y * 120.0);
                    float timeStep = floor(_Time.y * 25.0);
                    float noise = PseudoRandom(float2(blockY, timeStep));

                    float tearShift = (noise - 0.5) * _TearAmount * pow(vt, 2.0);
                    refractedUV.x += tearShift;
                }

                // Chromatic Aberration RGB split
                float2 splitOffset = float2(_ChromaticOffset * vt, 0.0);

                float3 finalColor;
                finalColor.r = SampleSceneColor(refractedUV + splitOffset).r;
                finalColor.g = SampleSceneColor(refractedUV).g;
                finalColor.b = SampleSceneColor(refractedUV - splitOffset).b;

                if (vt > 0.85)
                {
                    float whiteNoise = PseudoRandom(uv + _Time.y);
                    finalColor += whiteNoise * 0.3 * (vt - 0.85);
                }

                return float4(finalColor, 1.0);
            }
            ENDHLSL
        }
    }
}
