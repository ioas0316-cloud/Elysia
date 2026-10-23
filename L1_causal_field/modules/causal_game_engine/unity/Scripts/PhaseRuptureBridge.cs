// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using Unity.Entities;
using Unity.Mathematics;
using UnityEngine;

namespace CausalEngine.Unity.Homeostasis
{
    [ExecuteAlways]
    public class PhaseRuptureBridge : MonoBehaviour
    {
        [Header("Shader Settings")]
        [SerializeField] private Material ruptureMaterial;
        [SerializeField] private float smoothSpeed = 10.0f;

        private EntityQuery _configQuery;
        private EntityManager _entityManager;
        private float _currentIntensity = 0.0f;

        private static readonly int RuptureIntensityProperty = Shader.PropertyToID("_RuptureIntensity");

        private void Update()
        {
            if (ruptureMaterial == null) return;

            var defaultWorld = World.DefaultGameObjectInjectionWorld;
            if (defaultWorld == null || !defaultWorld.IsCreated) return;

            _entityManager = defaultWorld.EntityManager;

            if (_configQuery == default)
            {
                _configQuery = _entityManager.CreateEntityQuery(typeof(HomeostasisConfigSingleton));
            }

            if (!_configQuery.HasSingleton<HomeostasisConfigSingleton>()) return;

            var config = _configQuery.GetSingleton<HomeostasisConfigSingleton>();

            float uCore = config.TotalUCore;
            float uMax = math.max(config.UMax, 0.0001f);
            float ratio = uCore / uMax;

            float targetIntensity = 0.0f;
            if (ratio >= 0.8f)
            {
                targetIntensity = math.saturate((ratio - 0.8f) / 0.2f);
                targetIntensity = math.pow(targetIntensity, 1.5f);
            }

            _currentIntensity = math.lerp(_currentIntensity, targetIntensity, Time.deltaTime * smoothSpeed);
            ruptureMaterial.SetFloat(RuptureIntensityProperty, _currentIntensity);
        }
    }
}
