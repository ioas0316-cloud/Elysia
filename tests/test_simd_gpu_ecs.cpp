#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

#include "causal_engine/simd_gpu_ecs/simd_component_buffers.hpp"
#include "causal_engine/simd_gpu_ecs/simd_torque_transmission_system.hpp"
#include "causal_engine/simd_gpu_ecs/gpu_particle_frame_pipeline.hpp"
#include "causal_engine/simd_gpu_ecs/indirect_dispatch_pipeline.hpp"

// Scalar Reference Implementation of Transmission Step
void ScalarStepTransmission(
    FVariableGearComponentBuffer& GearBuffer,
    float DeltaTime,
    size_t EntityCount)
{
    float* pCurrent = GearBuffer.CurrentRatio.data();
    const float* pTarget = GearBuffer.TargetRatio.data();
    const float* pSpeed = GearBuffer.ShiftSpeed.data();
    const float* pFriction = GearBuffer.Friction.data();

    for (size_t idx = 0; idx < EntityCount; ++idx)
    {
        float dR = pSpeed[idx] * (pTarget[idx] - pCurrent[idx]) - (pFriction[idx] * pCurrent[idx]);
        pCurrent[idx] = std::max(0.0f, pCurrent[idx] + dR * DeltaTime);
    }
}

// Scalar Reference Implementation of Torque Integration
void ScalarIntegrateTorque(
    FFixedGearComponentBuffer& FixedBuffer,
    FVariableGearComponentBuffer& VarBuffer,
    FFlywheelComponentBuffer& FlywheelBuffer,
    float DeltaTime,
    size_t EntityCount)
{
    const float* pRFixed = FixedBuffer.FixedRatio.data();
    const float* pInertia = FixedBuffer.SystemInertia.data();
    const float* pRVar = VarBuffer.CurrentRatio.data();

    const float* pTIn = FlywheelBuffer.InputTorque.data();
    float* pTBrake = FlywheelBuffer.BrakingTorque.data();
    float* pOmega = FlywheelBuffer.AngularVelocity.data();
    const float* pDamping = FlywheelBuffer.DampingFactor.data();
    float* pDerivedDamage = FlywheelBuffer.DerivedOutput.data();

    for (size_t idx = 0; idx < EntityCount; ++idx)
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

int main()
{
    std::cout << "Starting SIMD GPU ECS Unit Tests..." << std::endl;

    const size_t entityCounts[] = { 1, 7, 8, 15, 16, 100, 1024, 1027 };
    const float deltaTime = 0.016f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.1f, 10.0f);

    for (size_t count : entityCounts)
    {
        std::cout << "Testing entity count: " << count << "..." << std::endl;

        // Initialize SIMD Buffers
        FVariableGearComponentBuffer varBufferSIMD;
        varBufferSIMD.Resize(count);

        FVariableGearComponentBuffer varBufferScalar;
        varBufferScalar.Resize(count);

        for (size_t i = 0; i < count; ++i)
        {
            float cur = dist(rng);
            float tgt = dist(rng);
            float spd = dist(rng);
            float frc = dist(rng);

            varBufferSIMD.CurrentRatio[i] = cur;
            varBufferSIMD.TargetRatio[i]  = tgt;
            varBufferSIMD.ShiftSpeed[i]   = spd;
            varBufferSIMD.Friction[i]     = frc;

            varBufferScalar.CurrentRatio[i] = cur;
            varBufferScalar.TargetRatio[i]  = tgt;
            varBufferScalar.ShiftSpeed[i]   = spd;
            varBufferScalar.Friction[i]     = frc;
        }

        // Test Transmission
        FSIMDTorqueTransmissionSystem::BatchStepTransmission(varBufferSIMD, deltaTime, count);
        ScalarStepTransmission(varBufferScalar, deltaTime, count);

        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(varBufferSIMD.CurrentRatio[i] - varBufferScalar.CurrentRatio[i]);
            assert(diff < 1e-4f && "Transmission step mismatch between SIMD and Scalar");
        }

        // Initialize Flywheel and Fixed Buffers
        FFixedGearComponentBuffer fixedSIMD, fixedScalar;
        FFlywheelComponentBuffer flywheelSIMD, flywheelScalar;

        fixedSIMD.Resize(count);
        fixedScalar.Resize(count);
        flywheelSIMD.Resize(count);
        flywheelScalar.Resize(count);

        for (size_t i = 0; i < count; ++i)
        {
            float rFix = dist(rng);
            float inert = dist(rng);
            float w = dist(rng);
            float damp = dist(rng);
            float tIn = dist(rng);
            float tBrk = dist(rng);

            fixedSIMD.FixedRatio[i] = fixedScalar.FixedRatio[i] = rFix;
            fixedSIMD.SystemInertia[i] = fixedScalar.SystemInertia[i] = inert;

            flywheelSIMD.AngularVelocity[i] = flywheelScalar.AngularVelocity[i] = w;
            flywheelSIMD.DampingFactor[i] = flywheelScalar.DampingFactor[i] = damp;
            flywheelSIMD.InputTorque[i] = flywheelScalar.InputTorque[i] = tIn;
            flywheelSIMD.BrakingTorque[i] = flywheelScalar.BrakingTorque[i] = tBrk;
        }

        // Test Integration
        FSIMDTorqueTransmissionSystem::BatchIntegrateTorque(fixedSIMD, varBufferSIMD, flywheelSIMD, deltaTime, count);
        ScalarIntegrateTorque(fixedScalar, varBufferScalar, flywheelScalar, deltaTime, count);

        for (size_t i = 0; i < count; ++i)
        {
            float diffDamage = std::abs(flywheelSIMD.DerivedOutput[i] - flywheelScalar.DerivedOutput[i]);
            float diffOmega = std::abs(flywheelSIMD.AngularVelocity[i] - flywheelScalar.AngularVelocity[i]);
            float diffBrake = std::abs(flywheelSIMD.BrakingTorque[i] - flywheelScalar.BrakingTorque[i]);

            assert(diffDamage < 1e-4f && "Derived damage mismatch");
            assert(diffOmega < 1e-4f && "Angular velocity mismatch");
            assert(diffBrake < 1e-4f && "Braking torque reset mismatch");
        }
    }

    // Verify Host Pipeline API instantiations
    GPUParticleSystemManager particleMgr;
    IndirectDispatchPipeline dispatchPipeline;
    (void)particleMgr;
    (void)dispatchPipeline;

    std::cout << "ALL SIMD GPU ECS TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
