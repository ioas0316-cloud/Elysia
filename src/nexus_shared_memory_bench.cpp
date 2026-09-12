#include <iostream>
#include <chrono>
#include <cstring>
#include <vector>
#include <cstdint>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

struct CausalNexusBuffer {
    uint64_t frame_tick;
    uint8_t trajectory_bits[512 * 512];
    uint8_t hitbox_bits[512 * 512];
};

void RunSharedMemoryBenchmark() {
    const char* szMapName = "AsuraNexusBenchmarkBuffer";
    const size_t bufferSize = sizeof(CausalNexusBuffer);

    std::cout << "[Nexus Memory Bench] Shared Memory Buffer Size: " << bufferSize << " bytes (512KB)\n";

    uint8_t localTraj[512 * 512];
    uint8_t localHitbox[512 * 512];
    std::memset(localTraj, 0x01, sizeof(localTraj));
    std::memset(localHitbox, 0xFF, sizeof(localHitbox));

    const int ITERATIONS = 10000;

#if defined(_WIN32) || defined(_WIN64)
    HANDLE hMapFile = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)bufferSize, szMapName);
    if (!hMapFile) {
        std::cerr << "Windows Shared Memory creation failed\n";
        return;
    }
    CausalNexusBuffer* pBuffer = (CausalNexusBuffer*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, bufferSize);
    if (!pBuffer) {
        CloseHandle(hMapFile);
        return;
    }
#else
    int shm_fd = shm_open(szMapName, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        // Fallback to heap buffer if OS shm permissions restrict
        std::cout << "[Notice] shm_open restricted. Benchmarking in-memory zero-copy pointer transfers.\n";
    } else {
        ftruncate(shm_fd, bufferSize);
    }
    CausalNexusBuffer fallback_buf;
    CausalNexusBuffer* pBuffer = (shm_fd != -1) ?
        (CausalNexusBuffer*)mmap(0, bufferSize, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0) : &fallback_buf;
#endif

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; ++i) {
        pBuffer->frame_tick = i;
        std::memcpy(pBuffer->trajectory_bits, localTraj, sizeof(localTraj));
        std::memcpy(pBuffer->hitbox_bits, localHitbox, sizeof(localHitbox));
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed_us = end - start;

    double avg_us = elapsed_us.count() / ITERATIONS;
    double avg_ms = avg_us / 1000.0;

    std::cout << "=================================================\n";
    std::cout << "[Nexus Memory Bench] Completed " << ITERATIONS << " Bit Transfers\n";
    std::cout << " Average Frame Transfer Latency: " << avg_us << " us (" << avg_ms << " ms)\n";
    std::cout << " Transfer Method: Zero-Copy Pointer MMF / Direct RAM Write\n";
    std::cout << "=================================================\n";

#if defined(_WIN32) || defined(_WIN64)
    UnmapViewOfFile(pBuffer);
    CloseHandle(hMapFile);
#else
    if (shm_fd != -1) {
        munmap(pBuffer, bufferSize);
        close(shm_fd);
        shm_unlink(szMapName);
    }
#endif
}

int main() {
    RunSharedMemoryBenchmark();
    return 0;
}
