// HLSL Compute Shader: HZB Max-Reduction Downsampler (Mip N -> Mip N+1)

Texture2D<float>   g_InputDepthMip  : register(t0);
RWTexture2D<float> g_OutputDepthMip : register(u0);

[numthreads(16, 16, 1)]
void CS_BuildHZBMip(uint3 DTid : SV_DispatchThreadID)
{
    uint2 outputTexel = DTid.xy;
    uint2 inputTexelBase = outputTexel * 2;

    float d0 = g_InputDepthMip.mips[0][inputTexelBase + uint2(0, 0)];
    float d1 = g_InputDepthMip.mips[0][inputTexelBase + uint2(1, 0)];
    float d2 = g_InputDepthMip.mips[0][inputTexelBase + uint2(0, 1)];
    float d3 = g_InputDepthMip.mips[0][inputTexelBase + uint2(1, 1)];

    float maxDepth = max(max(d0, d1), max(d2, d3));

    g_OutputDepthMip[outputTexel] = maxDepth;
}
