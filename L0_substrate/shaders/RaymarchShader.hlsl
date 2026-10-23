// RaymarchShader.hlsl - Volume Raymarching Pass
Texture3D<float4> g_QField : register(t0); // Rotor Field
Texture3D<float4> g_VField : register(t1); // Velocity Field
SamplerState      g_LinearSampler : register(s0);

cbuffer RaymarchConstants : register(b0) {
    float4x4 g_InvViewProj;
    float3   g_CameraPos;
    float    g_StepSize;
};

struct VS_OUTPUT {
    float4 Pos : SV_POSITION;
    float2 UV  : TEXCOORD0;
};

// Full-screen Triangle Generation
VS_OUTPUT VS_Main(uint vertexID : SV_VertexID) {
    VS_OUTPUT output;
    output.UV = float2((vertexID << 1) & 2, vertexID & 2);
    output.Pos = float4(output.UV * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return output;
}

// Ray vs AABB (0,0,0) ~ (1,1,1) Intersection
bool RayBoxIntersection(float3 rayOrigin, float3 rayDir, out float tNear, out float tFar) {
    float3 boxMin = float3(0.0f, 0.0f, 0.0f);
    float3 boxMax = float3(1.0f, 1.0f, 1.0f);

    float3 invDir = 1.0f / (rayDir + 1e-6f);
    float3 tMin = (boxMin - rayOrigin) * invDir;
    float3 tMax = (boxMax - rayOrigin) * invDir;

    float3 t1 = min(tMin, tMax);
    float3 t2 = max(tMin, tMax);

    tNear = max(max(t1.x, t1.y), t1.z);
    tFar  = min(min(t2.x, t2.y), t2.z);

    return tNear < tFar && tFar > 0.0f;
}

float4 PS_Main(VS_OUTPUT input) : SV_TARGET {
    float2 ndc = input.UV * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f);
    float4 target = mul(g_InvViewProj, float4(ndc, 1.0f, 1.0f));
    float3 rayDir = normalize(target.xyz / target.w - g_CameraPos);

    float tNear, tFar;
    if (!RayBoxIntersection(g_CameraPos, rayDir, tNear, tFar)) {
        return float4(0.05f, 0.05f, 0.08f, 1.0f);
    }

    tNear = max(tNear, 0.0f);
    float3 rayPos = g_CameraPos + rayDir * tNear;
    float rayLength = tFar - tNear;

    float4 accumulatedColor = float4(0, 0, 0, 0);
    int maxSteps = 128;
    float stepSize = rayLength / (float)maxSteps;

    [loop]
    for (int i = 0; i < maxSteps; ++i) {
        if (accumulatedColor.a >= 0.95f) break;

        float3 uvw = rayPos;
        float3 vel = g_VField.SampleLevel(g_LinearSampler, uvw, 0).xyz;
        float4 rotor = g_QField.SampleLevel(g_LinearSampler, uvw, 0);

        float speed = length(vel);
        float density = saturate(speed * 2.0f + length(rotor.xyz) * 0.5f);

        if (density > 0.01f) {
            float3 color = normalize(abs(vel) + 1e-3f) * density;
            float alpha = density * 0.05f;

            accumulatedColor.rgb += (1.0f - accumulatedColor.a) * color * alpha;
            accumulatedColor.a   += (1.0f - accumulatedColor.a) * alpha;
        }

        rayPos += rayDir * stepSize;
    }

    return float4(accumulatedColor.rgb + (1.0f - accumulatedColor.a) * float3(0.05f, 0.05f, 0.08f), 1.0f);
}
