// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;

namespace CausalEngine.Unity.Homeostasis
{
    public struct CCNodeHomeostasisComponent : IComponentData
    {
        public float CurrentWeight;     // L_ij current
        public float CoreWeight;        // L_ij core
        public float RawVtGradient;     // Raw V_t gradient
        public float ClampedVtGradient; // Clamped V_t* gradient
        public float LocalPotential;    // U_core local
    }

    public struct HomeostasisConfigSingleton : IComponentData
    {
        public float KappaCore;  // Stiffness kappa_core
        public float UMax;       // U_max threshold
        public float Lambda;     // Damping lambda
        public float PowerP;     // Power p
        public float Gamma;      // Restorative gamma
        public float TotalUCore; // Accumulated U_core
    }

    // Chunk 메모리에 직접 내장되는 10프레임 Debug RingBuffer Component
    public unsafe struct NodeDebugHistoryComponent : IComponentData
    {
        public const int HistoryCapacity = 10;

        // Fixed Buffer: 별도의 DynamicBuffer Heap 할당 없이 IComponentData 내부 직렬화
        public fixed float CurrentWeightHistory[HistoryCapacity];
        public fixed float RawVtHistory[HistoryCapacity];
        public fixed uint FrameHistory[HistoryCapacity];

        public int HeadIndex; // 현재 기록 위치 (0 ~ 9)
        public bool IsFrozen; // 오염 발생 시 true로 전환되어 직전 10프레임 이력 동결

        // 프레임 데이터 기록 (Push)
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Push(float weight, float rawVt, uint currentFrame)
        {
            // 동결 상태에서는 이력을 더 이상 덮어쓰지 않고 오염 직전 상태 보존
            if (IsFrozen) return;

            int idx = HeadIndex;
            CurrentWeightHistory[idx] = weight;
            RawVtHistory[idx] = rawVt;
            FrameHistory[idx] = currentFrame;

            HeadIndex = (HeadIndex + 1) % HistoryCapacity;
        }

        // 오래된 기록(0)부터 최신 기록(9) 순으로 정렬하여 조회하는 헬퍼
        public readonly void GetSampleAt(int relativeIndex, out float weight, out float rawVt, out uint frame)
        {
            // relativeIndex: 0 (10프레임 전) ~ 9 (가장 최근 프레임)
            int actualIdx = (HeadIndex + relativeIndex) % HistoryCapacity;
            weight = CurrentWeightHistory[actualIdx];
            rawVt = RawVtHistory[actualIdx];
            frame = FrameHistory[actualIdx];
        }
    }

    [BurstCompile]
    public partial struct HomeostaticGovernorSystem : ISystem
    {
        private uint _currentFrame;

        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<HomeostasisConfigSingleton>();
            _currentFrame = 0;
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            _currentFrame++;

            var configEntity = SystemAPI.GetSingletonEntity<HomeostasisConfigSingleton>();
            var config = SystemAPI.GetComponent<HomeostasisConfigSingleton>(configEntity);

            var accumulatedUCore = new NativeReference<float>(0.0f, Allocator.TempJob);
            var corruptedNodeCount = new NativeReference<int>(0, Allocator.TempJob);

            var calculatePotentialJob = new CalculateCorePotentialJob
            {
                KappaCore = config.KappaCore,
                TotalUCoreRef = accumulatedUCore
            };
            state.Dependency = calculatePotentialJob.ScheduleParallel(state.Dependency);

            var clampTensionJob = new SafeVtCalculationWithHistoryJob
            {
                CurrentFrame = _currentFrame,
                KappaCore = config.KappaCore,
                UMax = config.UMax,
                Lambda = config.Lambda,
                PowerP = config.PowerP,
                Gamma = config.Gamma,
                TotalUCoreRef = accumulatedUCore,
                CorruptedNodeCountRef = corruptedNodeCount
            };
            state.Dependency = clampTensionJob.ScheduleParallel(state.Dependency);

            accumulatedUCore.Dispose(state.Dependency);
            corruptedNodeCount.Dispose(state.Dependency);
        }
    }

    [BurstCompile]
    public partial struct CalculateCorePotentialJob : IJobEntity
    {
        public float KappaCore;
        [NativeDisableParallelForRestriction] public NativeReference<float> TotalUCoreRef;

        void Execute(ref CCNodeHomeostasisComponent node)
        {
            float safeCurrentWeight = Sanitize(node.CurrentWeight, node.CoreWeight);
            float deltaL = safeCurrentWeight - node.CoreWeight;
            float localU = 0.5f * KappaCore * (deltaL * deltaL);
            node.LocalPotential = localU;

            unsafe
            {
                float* ptr = (float*)TotalUCoreRef.GetUnsafePtr();
                AtomicAddFloat(ptr, localU);
            }
        }

        private static void AtomicAddFloat(unsafe float* ptr, float value)
        {
            int* intPtr = (int*)ptr;
            int oldInt, newInt;
            do
            {
                oldInt = *intPtr;
                float oldValue = math.asfloat(oldInt);
                float newValue = oldValue + value;
                newInt = math.asint(newValue);
            }
            while (System.Threading.Interlocked.CompareExchange(ref *intPtr, newInt, oldInt) != oldInt);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Sanitize(float value, float fallback)
        {
            return math.select(fallback, value, math.isfinite(value));
        }
    }

    [BurstCompile]
    public partial struct SafeClampVtGradientJob : IJobEntity
    {
        public float KappaCore;
        public float UMax;
        public float Lambda;
        public float PowerP;
        public float Gamma;

        [ReadOnly] public NativeReference<float> TotalUCoreRef;
        [NativeDisableParallelForRestriction] public NativeReference<int> CorruptedNodeCountRef; // 에러 감지 디버그 카운터

        void Execute(ref CCNodeHomeostasisComponent node)
        {
            // 1. 글로벌 U_core 수치 오염 검증 및 대치 (NaN/Inf 진입 시 0.0f로 격리)
            float rawTotalUCore = TotalUCoreRef.Value;
            bool isUCoreValid = math.isfinite(rawTotalUCore);
            float safeTotalUCore = math.select(0.0f, rawTotalUCore, isUCoreValid);

            // 2. 노드 입력 가중치 sanitization
            float safeCurrentWeight = Sanitize(node.CurrentWeight, node.CoreWeight);
            float safeRawVt = Sanitize(node.RawVtGradient, 0.0f);

            // 3. 지수 감쇄 및 감쇄 구배 연산
            float normalizedU = math.saturate(safeTotalUCore / math.max(UMax, 0.0001f));
            float sigmaH = math.exp(-Lambda * math.pow(normalizedU, PowerP));
            float gradUCore = KappaCore * (safeCurrentWeight - node.CoreWeight);

            // 4. V_t 장력 산출
            float rawClampedVt = (sigmaH * safeRawVt) - (Gamma * gradUCore);

            // 5. 연산 결과 검증 및 Fail-Safe 복구
            bool isVtValid = math.isfinite(rawClampedVt);

            // 연산 정상: 산출값 유지가 기본, 연산 오염: 0.0f (긴급 억제) 대치
            float finalVt = math.select(0.0f, math.clamp(rawClampedVt, -100.0f, 100.0f), isVtValid);

            // 비상 차단: U_core 한계 돌파 시 외부 장력 완전 절단
            if (safeTotalUCore >= UMax)
            {
                finalVt = -Gamma * Sanitize(gradUCore, 0.0f);
            }

            // 6. 상태 기록 및 오염 발생 시 가중치 강제 원상복구(Core Snap)
            node.ClampedVtGradient = finalVt;

            if (!isVtValid || !isUCoreValid)
            {
                // 오염 발생 시 노드 가중치를 안전한 CoreWeight로 즉시 스냅 백
                node.CurrentWeight = node.CoreWeight;
                node.LocalPotential = 0.0f;

                // Atomic 카운터 증가 (Burst 지원)
                unsafe
                {
                    int* ptr = (int*)CorruptedNodeCountRef.GetUnsafePtr();
                    System.Threading.Interlocked.Increment(ref *ptr);
                }
            }
        }

        // Burst 최적화 인라인 Sanitizer (Branchless 대치)
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Sanitize(float value, float fallback)
        {
            return math.select(fallback, value, math.isfinite(value));
        }
    }

    [BurstCompile]
    public partial struct SafeVtCalculationWithHistoryJob : IJobEntity
    {
        public uint CurrentFrame;
        public float KappaCore;
        public float UMax;
        public float Lambda;
        public float PowerP;
        public float Gamma;

        [ReadOnly] public NativeReference<float> TotalUCoreRef;
        [NativeDisableParallelForRestriction] public NativeReference<int> CorruptedNodeCountRef;

        void Execute(
            ref CCNodeHomeostasisComponent node,
            ref NodeDebugHistoryComponent history)
        {
            // 1. 연산 실행 직전, 현재 노드 상태를 링 버퍼에 기록
            history.Push(node.CurrentWeight, node.RawVtGradient, CurrentFrame);

            // 2. 가중치 및 입력값 Sanitization
            float safeWeight = math.select(node.CoreWeight, node.CurrentWeight, math.isfinite(node.CurrentWeight));
            float safeRawVt = math.select(0.0f, node.RawVtGradient, math.isfinite(node.RawVtGradient));

            float rawTotalUCore = TotalUCoreRef.Value;
            bool isUCoreValid = math.isfinite(rawTotalUCore);
            float safeTotalUCore = math.select(0.0f, rawTotalUCore, isUCoreValid);

            // 3. V_t 기울기 및 포텐셜 계산
            float normalizedU = math.saturate(safeTotalUCore / math.max(UMax, 0.0001f));
            float sigmaH = math.exp(-Lambda * math.pow(normalizedU, PowerP));
            float gradUCore = KappaCore * (safeWeight - node.CoreWeight);

            float calculatedVt = (sigmaH * safeRawVt) - (Gamma * gradUCore);

            // 4. 수치 유효성 검증
            bool isVtValid = math.isfinite(calculatedVt);

            if (!isVtValid || !isUCoreValid)
            {
                // [Fail-Safe 연동]: 오염 발생 시 RingBuffer 즉시 동결 (Freeze)
                // 이를 통해 오염 순간 기준 '직전 10프레임의 궤적'이 덮어씌워지지 않고 보존됨
                history.IsFrozen = true;

                // 노드 상태 안전 초기화 (Core Snap)
                node.CurrentWeight = node.CoreWeight;
                node.LocalPotential = 0.0f;
                node.ClampedVtGradient = 0.0f;

                if (CorruptedNodeCountRef.IsCreated)
                {
                    unsafe
                    {
                        int* ptr = (int*)CorruptedNodeCountRef.GetUnsafePtr();
                        System.Threading.Interlocked.Increment(ref *ptr);
                    }
                }
            }
            else
            {
                node.ClampedVtGradient = math.clamp(calculatedVt, -100.0f, 100.0f);
            }
        }
    }
}
