#ifndef SIMD_TORQUE_TRANSMISSION_SYSTEM_HPP
#define SIMD_TORQUE_TRANSMISSION_SYSTEM_HPP

#include "simd_component_buffers.hpp"

class FSIMDTorqueTransmissionSystem
{
public:
    // 가변 기어 변속 시스템 (SIMD Vectorized Transmission)
    static void BatchStepTransmission(
        FVariableGearComponentBuffer& GearBuffer,
        float DeltaTime,
        size_t EntityCount);

    // 플라이휠 운동방정식 적분 & 출력 유도 시스템 (SIMD Integration)
    static void BatchIntegrateTorque(
        FFixedGearComponentBuffer& FixedBuffer,
        FVariableGearComponentBuffer& VarBuffer,
        FFlywheelComponentBuffer& FlywheelBuffer,
        float DeltaTime,
        size_t EntityCount);
};

#endif // SIMD_TORQUE_TRANSMISSION_SYSTEM_HPP
