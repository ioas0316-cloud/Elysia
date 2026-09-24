#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#include "elysia/causal_lut_header.h"
#include "elysia/vram_slab_allocator.h"
#include "elysia/phase_lock_evaluator.h"
#include "elysia/sensory_tensor_encoder.h"
#include "elysia/sensory_quantizer.h"
#include "elysia/codebook_online_trainer.h"
#include "elysia/multi_gpu_nvlink_router.h"
#include "elysia/d3d12_causal_page_allocator.h"
#include "elysia/causal_memory_cache_manager.h"
#include "elysia/elysian_engine_main_loop.h"

void TestVRAMSlabAllocator() {
    std::cout << "[1/8] Testing VRAM Slab Allocator...\n";
    uint8_t dummyVram[1024 * 64 * 4]; // 4 slabs of 64KB
    VRAMSlabAllocator allocator(dummyVram, 4, 64 * 1024);

    assert(allocator.GetAvailableSlabs() == 4);

    uint32_t idx0 = 999, idx1 = 999, idx2 = 999, idx3 = 999;
    uint8_t* ptr0 = allocator.AllocateSlab(idx0);
    uint8_t* ptr1 = allocator.AllocateSlab(idx1);
    uint8_t* ptr2 = allocator.AllocateSlab(idx2);
    uint8_t* ptr3 = allocator.AllocateSlab(idx3);

    assert(ptr0 != nullptr && ptr1 != nullptr && ptr2 != nullptr && ptr3 != nullptr);
    assert(allocator.GetAvailableSlabs() == 0);

    uint32_t idxFull = 999;
    uint8_t* ptrFull = allocator.AllocateSlab(idxFull);
    assert(ptrFull == nullptr);

    allocator.FreeSlab(idx0);
    assert(allocator.GetAvailableSlabs() == 1);

    std::cout << "  -> VRAM Slab Allocator PASSED.\n";
}

void TestPhaseLockEvaluator() {
    std::cout << "[2/8] Testing Phase-Lock Evaluator Kernel...\n";
    uint32_t nodeCount = 2;
    uint32_t windowSize = 4;
    std::vector<TrajectoryNode> ringBuffer(nodeCount * windowSize);

    // Node 0: Stationary/Stable (Energy low, velocity variance low)
    for (uint32_t f = 0; f < windowSize; ++f) {
        ringBuffer[0 * windowSize + f].position_phase = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
        ringBuffer[0 * windowSize + f].velocity_energy = make_float4(0.01f, 0.01f, 0.0f, 0.02f);
    }

    // Node 1: Dynamic/High Energy (Energy high)
    for (uint32_t f = 0; f < windowSize; ++f) {
        ringBuffer[1 * windowSize + f].position_phase = make_float4(10.0f, 10.0f, 10.0f, 0.0f);
        ringBuffer[1 * windowSize + f].velocity_energy = make_float4(5.0f, 5.0f, 5.0f, 10.0f);
    }

    std::vector<BakeMetaData> bakeData(nodeCount);

    LaunchPhaseLockEvaluation(ringBuffer.data(), bakeData.data(), 0.1f, 0.05f, nodeCount, windowSize);

    assert(bakeData[0].is_phase_locked == true);
    assert(bakeData[1].is_phase_locked == false);

    std::cout << "  -> Phase-Lock Evaluator PASSED.\n";
}

void TestSensoryTensorEncoder() {
    std::cout << "[3/8] Testing Sensory Tensor Encoder (16-Ch Injection)...\n";
    uint32_t hashSize = 256;
    std::vector<float4> spatialMap(hashSize * 4, make_float4(0, 0, 0, 0));

    UnifiedSensoryFrame frame;
    frame.visual_depth = make_float4(0.5f, 0.6f, 0.7f, 1.0f);
    frame.acoustic_phase = make_float4(0.2f, 440.0f, 0.1f, 0.9f);
    frame.physical_force = make_float4(1.0f, 2.0f, 3.0f, 0.05f);
    frame.contextual_meta = make_float4(0.8f, 42.0f, 0.0f, 0.0f);

    LaunchInjectSensoryTensor(&frame, spatialMap.data(), 1, hashSize);

    // Verify atomic injection took place non-zero
    bool foundNonZero = false;
    for (const auto& f4 : spatialMap) {
        if (f4.x != 0.0f || f4.y != 0.0f || f4.z != 0.0f || f4.w != 0.0f) {
            foundNonZero = true;
            break;
        }
    }
    assert(foundNonZero);

    std::cout << "  -> Sensory Tensor Encoder PASSED.\n";
}

void TestSensoryQuantizerAndOnlineTrainer() {
    std::cout << "[4/8] Testing Sensory Quantizer & Online K-Means Codebook Trainer...\n";
    uint32_t numNodes = 10;
    std::vector<float> spatialTensorMap(numNodes * VECTOR_DIM, 0.5f);
    std::vector<float> codebook(CODEBOOK_SIZE * VECTOR_DIM, 0.0f);

    // Set centroid 5 close to 0.5f
    for (int c = 0; c < VECTOR_DIM; ++c) {
        codebook[5 * VECTOR_DIM + c] = 0.51f;
    }

    std::vector<uint8_t> compressedIndices(numNodes, 0);

    LaunchSensoryQuantizer(spatialTensorMap.data(), compressedIndices.data(), codebook.data(), numNodes);

    for (uint32_t i = 0; i < numNodes; ++i) {
        assert(compressedIndices[i] == 5);
    }

    // Train codebook
    std::vector<float> clusterSums(CODEBOOK_SIZE * VECTOR_DIM, 0.0f);
    std::vector<uint32_t> clusterCounts(CODEBOOK_SIZE, 0);

    ExecuteOnlineCodebookUpdate(
        spatialTensorMap.data(),
        compressedIndices.data(),
        codebook.data(),
        clusterSums.data(),
        clusterCounts.data(),
        numNodes,
        0.1f
    );

    assert(clusterCounts[5] == numNodes);

    std::cout << "  -> Sensory Quantizer & Online Codebook Trainer PASSED.\n";
}

void TestMultiGPUNVLinkCacheRouter() {
    std::cout << "[5/8] Testing Multi-GPU NVLink Cache Router...\n";
    MultiGPUNVLinkCacheRouter router;
    assert(router.GetDeviceCount() >= 1);

    std::vector<float> srcTensor = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> dstTensor(4, 0.0f);

    DistributedPageLocation loc;
    loc.gpuId = 0;
    loc.vramDevicePointer = srcTensor.data();
    loc.usageFrequency = 10;

    router.FetchCausalTensorP2P(0, loc, dstTensor.data(), sizeof(float) * 4);
    assert(dstTensor[0] == 1.0f && dstTensor[3] == 4.0f);

    std::vector<float> masterCodebook(CODEBOOK_SIZE * VECTOR_DIM, 1.5f);
    std::vector<float> slaveCodebook(CODEBOOK_SIZE * VECTOR_DIM, 0.0f);

    router.BroadcastCodebook(masterCodebook.data(), slaveCodebook.data(), CODEBOOK_SIZE * VECTOR_DIM);
    assert(slaveCodebook[0] == 1.5f);

    std::cout << "  -> Multi-GPU NVLink Router PASSED.\n";
}

void TestCausalMemoryCacheManagerAndLRU() {
    std::cout << "[6/8] Testing Causal Memory Cache Manager & LRU Eviction...\n";
    D3D12CausalPageAllocator allocator(nullptr, 2, 64 * 1024); // Only 2 VRAM pages max
    CausalMemoryCacheManager cacheManager(&allocator, 2, 16);

    cacheManager.RegisterEntry(101, 0, 64 * 1024, false);
    cacheManager.RegisterEntry(102, 64 * 1024, 64 * 1024, false);
    cacheManager.RegisterEntry(103, 128 * 1024, 64 * 1024, false);

    uint64_t addr101 = cacheManager.FetchOrPageInCausalPage(101);
    uint64_t addr102 = cacheManager.FetchOrPageInCausalPage(102);
    assert(addr101 != 0 && addr102 != 0);

    // Page in 103 -> Should evict 101 (LRU)
    uint64_t addr103 = cacheManager.FetchOrPageInCausalPage(103);
    assert(addr103 != 0);

    CausalPageEntry* entry101 = cacheManager.GetPageTable().LookupEntry(101);
    assert(entry101->vram_pinned == 0); // Evicted

    std::cout << "  -> Causal Memory Cache Manager & LRU PASSED.\n";
}

void TestElysianEngineLoopIntegration() {
    std::cout << "[7/8] Testing Integrated Elysian Engine Main Loop...\n";
    ElysianEngineLoop engine(nullptr, "test_causal_storage.dat");

    engine.RegisterCausalPage(0xDEADBEEF, 0, 64 * 1024, true);

    std::vector<float> sensoryBatch(16 * 4, 0.5f);
    engine.Tick(0xDEADBEEF, sensoryBatch.data(), 4);

    std::cout << "  -> Elysian Engine Main Loop Integration PASSED.\n";
}

int main() {
    std::cout << "====================================================\n";
    std::cout << " RUNNING ELYSIAN DUAL PILLARS HARDWARE PIPELINE TEST\n";
    std::cout << "====================================================\n";

    TestVRAMSlabAllocator();
    TestPhaseLockEvaluator();
    TestSensoryTensorEncoder();
    TestSensoryQuantizerAndOnlineTrainer();
    TestMultiGPUNVLinkCacheRouter();
    TestCausalMemoryCacheManagerAndLRU();
    TestElysianEngineLoopIntegration();

    std::cout << "====================================================\n";
    std::cout << " ALL ELYSIAN HARDWARE PIPELINE TESTS PASSED!\n";
    std::cout << "====================================================\n";
    return 0;
}
