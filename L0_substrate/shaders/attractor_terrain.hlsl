// HLSL Compute Shader: Morton3D Decode 및 Instance Transform Build
// C++ 128-byte aligned RotorNodeData 구조체와 1:1 패킹 호환

struct RotorNodeData {
    float4 quaternion;            // q = (x, y, z, w)
    float3 angularVelocity;       // omega = (wx, wy, wz)
    float  dampingBeta;           // 감쇄 계수
    uint2  spatialHashKey;        // 64-bit Morton Code (x: low 32b, y: high 32b)
    uint   usageFrequency;        // LRU Eviction 방어용 사용 빈도
    uint   isAttractor;           // Phase-Lock 달성 여부 (1: Attractor Basin)
    uint2  vramSlabAddress;       // 64-bit VRAM Slab 주소
    uint   reserved0;             // 128B Alignment 패딩
    uint   connectedNodeIndices[8];// 연결 노드 인덱스
    float  gearRatios[8];          // 기어비
};

struct InstanceTransform {
    float4x4 worldMatrix;
    float4   colorEnergy;         // (R, G, B, Phase-Lock Intensity)
};

RWStructuredBuffer<InstanceTransform> g_InstanceBuffer : register(u0);
StructuredBuffer<RotorNodeData>       g_NodePool       : register(t0);

// Morton Code 21-bit Bit-Deinterleaving Helper
uint Dilate1By2(uint x) {
    x &= 0x00092492;
    x = (x ^ (x >> 2))  & 0x030c30c3;
    x = (x ^ (x >> 4))  & 0x0300f00f;
    x = (x ^ (x >> 8))  & 0x030000ff;
    x = (x ^ (x >> 16)) & 0x000003ff;
    return x;
}

[numthreads(64, 1, 1)]
void CS_BuildAttractorTerrain(uint3 DTid : SV_DispatchThreadID) {
    uint nodeIdx = DTid.x;
    RotorNodeData node = g_NodePool[nodeIdx];

    // 1. 64비트 Morton Code에서 3D 격자 좌표 (x, y, z) 복원
    uint lowBits = node.spatialHashKey.x;
    float posX = (float)Dilate1By2(lowBits);
    float posY = (float)Dilate1By2(lowBits >> 1);
    float posZ = (float)Dilate1By2(lowBits >> 2);

    // 2. Quaternion (x, y, z, w) -> 3x3 회전 행렬 변환
    float4 q = node.quaternion;
    float3x3 rotMat = float3x3(
        1.0 - 2.0*(q.y*q.y + q.z*q.z), 2.0*(q.x*q.y - q.z*q.w),       2.0*(q.x*q.z + q.y*q.w),
        2.0*(q.x*q.y + q.z*q.w),       1.0 - 2.0*(q.x*q.x + q.z*q.z), 2.0*(q.y*q.z - q.x*q.w),
        2.0*(q.x*q.z - q.y*q.w),       2.0*(q.y*q.z + q.x*q.w),       1.0 - 2.0*(q.x*q.x + q.y*q.y)
    );

    // 3. 월드 변환 행렬 구성
    float4x4 world = float4x4(
        float4(rotMat[0], 0.0),
        float4(rotMat[1], 0.0),
        float4(rotMat[2], 0.0),
        float4(posX, posY, posZ, 1.0)
    );

    // 4. Attractor 상태에 따른 에너지 시각화 색상
    float4 color = (node.isAttractor == 1)
        ? float4(1.0, 0.85, 0.2, 1.0)  // Golden Phase-Lock (위상 고정 완료)
        : float4(0.2, 0.4, 1.0, 0.3);  // Dynamic Blue (동역학 회전 중)

    InstanceTransform inst;
    inst.worldMatrix = world;
    inst.colorEnergy = color;

    g_InstanceBuffer[nodeIdx] = inst;
}
