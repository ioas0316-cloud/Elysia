#include "causal_engine/simd_gpu_ecs/simd_torque_transmission_system.hpp"

void FSIMDTorqueTransmissionSystem::BatchStepTransmission(
    FVariableGearComponentBuffer& GearBuffer,
    float DeltaTime,
    size_t EntityCount)
{
    float* __restrict pCurrent = GearBuffer.CurrentRatio.data();
    const float* __restrict pTarget = GearBuffer.TargetRatio.data();
    const float* __restrict pSpeed = GearBuffer.ShiftSpeed.data();
    const float* __restrict pFriction = GearBuffer.Friction.data();

    size_t processed = 0;

#if defined(__AVX2__)
    const size_t SimdBlockCount = EntityCount / 8;
    const __m256 vDt = _mm256_set1_ps(DeltaTime);
    const __m256 vZero = _mm256_set1_ps(0.0f);

    for (size_t i = 0; i < SimdBlockCount; ++i)
    {
        size_t idx = i * 8;
        __m256 vCurrent = _mm256_load_ps(&pCurrent[idx]);
        __m256 vTarget  = _mm256_load_ps(&pTarget[idx]);
        __m256 vSpeed   = _mm256_load_ps(&pSpeed[idx]);
        __m256 vFriction= _mm256_load_ps(&pFriction[idx]);

        // dR = ShiftSpeed * (TargetRatio - CurrentRatio) - (Friction * CurrentRatio)
        __m256 vDeltaRatio = _mm256_sub_ps(vTarget, vCurrent);
        __m256 vShiftForce = _mm256_mul_ps(vSpeed, vDeltaRatio);
        __m256 vFrictionLoss = _mm256_mul_ps(vFriction, vCurrent);
        __m256 vDR = _mm256_sub_ps(vShiftForce, vFrictionLoss);

        // CurrentRatio += dR * DeltaTime
#if defined(__FMA__)
        vCurrent = _mm256_fmadd_ps(vDR, vDt, vCurrent);
#else
        vCurrent = _mm256_add_ps(vCurrent, _mm256_mul_ps(vDR, vDt));
#endif

        // Branchless Clamping: std::max(0.0f, CurrentRatio)
        vCurrent = _mm256_max_ps(vZero, vCurrent);

        _mm256_store_ps(&pCurrent[idx], vCurrent);
    }
    processed = SimdBlockCount * 8;
#endif

    // Remainder / Scalar loop
    for (size_t idx = processed; idx < EntityCount; ++idx)
    {
        float dR = pSpeed[idx] * (pTarget[idx] - pCurrent[idx]) - (pFriction[idx] * pCurrent[idx]);
        pCurrent[idx] = std::max(0.0f, pCurrent[idx] + dR * DeltaTime);
    }
}

void FSIMDTorqueTransmissionSystem::BatchIntegrateTorque(
    FFixedGearComponentBuffer& FixedBuffer,
    FVariableGearComponentBuffer& VarBuffer,
    FFlywheelComponentBuffer& FlywheelBuffer,
    float DeltaTime,
    size_t EntityCount)
{
    const float* __restrict pRFixed  = FixedBuffer.FixedRatio.data();
    const float* __restrict pInertia = FixedBuffer.SystemInertia.data();
    const float* __restrict pRVar    = VarBuffer.CurrentRatio.data();

    const float* __restrict pTIn     = FlywheelBuffer.InputTorque.data();
    float* __restrict pTBrake        = FlywheelBuffer.BrakingTorque.data();
    float* __restrict pOmega         = FlywheelBuffer.AngularVelocity.data();
    const float* __restrict pDamping = FlywheelBuffer.DampingFactor.data();
    float* __restrict pDerivedDamage = FlywheelBuffer.DerivedOutput.data();

    size_t processed = 0;

#if defined(__AVX2__)
    const size_t SimdBlockCount = EntityCount / 8;
    const __m256 vDt = _mm256_set1_ps(DeltaTime);
    const __m256 vZero = _mm256_set1_ps(0.0f);

    for (size_t i = 0; i < SimdBlockCount; ++i)
    {
        size_t idx = i * 8;

        __m256 vRFixed  = _mm256_load_ps(&pRFixed[idx]);
        __m256 vInertia = _mm256_load_ps(&pInertia[idx]);
        __m256 vRVar    = _mm256_load_ps(&pRVar[idx]);
        __m256 vTIn     = _mm256_load_ps(&pTIn[idx]);
        __m256 vTBrake  = _mm256_load_ps(&pTBrake[idx]);
        __m256 vOmega   = _mm256_load_ps(&pOmega[idx]);
        __m256 vDamping = _mm256_load_ps(&pDamping[idx]);

        // 1. OutputTorque = InputTorque * FixedRatio * CurrentRatio
        __m256 vOutputTorque = _mm256_mul_ps(_mm256_mul_ps(vTIn, vRFixed), vRVar);
        _mm256_store_ps(&pDerivedDamage[idx], vOutputTorque);

        // 2. NetTorque = OutputTorque - BrakingTorque - (Damping * Omega)
        __m256 vDampingTorque = _mm256_mul_ps(vDamping, vOmega);
        __m256 vNetTorque = _mm256_sub_ps(_mm256_sub_ps(vOutputTorque, vTBrake), vDampingTorque);

        // 3. SystemInertia Total = Inertia * FixedRatio
        __m256 vTotalInertia = _mm256_mul_ps(vInertia, vRFixed);

        // 4. dw/dt = NetTorque / TotalInertia
        __m256 vAlpha = _mm256_div_ps(vNetTorque, vTotalInertia);

        // 5. Omega += dw/dt * dt
#if defined(__FMA__)
        vOmega = _mm256_fmadd_ps(vAlpha, vDt, vOmega);
#else
        vOmega = _mm256_add_ps(vOmega, _mm256_mul_ps(vAlpha, vDt));
#endif
        vOmega = _mm256_max_ps(vZero, vOmega);

        _mm256_store_ps(&pOmega[idx], vOmega);

        // 6. 충격 제동 토크 소산 (Reset Braking Torque Impulse)
        _mm256_store_ps(&pTBrake[idx], vZero);
    }
    processed = SimdBlockCount * 8;
#endif

    // Remainder / Scalar loop
    for (size_t idx = processed; idx < EntityCount; ++idx)
    {
        float outputTorque = pTIn[idx] * pRFixed[idx] * pRVar[idx];
        pDerivedDamage[idx] = outputTorque;

        float dampingTorque = pDamping[idx] * pOmega[idx];
        float netTorque = outputTorque - pTBrake[idx] - dampingTorque;
        float totalInertia = pInertia[idx] * pRFixed[idx];

        float alpha = netTorque / totalInertia;
        pOmega[idx] = std::max(0.0f, pOmega[idx] + alpha * DeltaTime);
        pTBrake[idx] = 0.0f;
    }
}
