#include "elysia_bridge.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <stdexcept>

static int shm_fd = -1;

CognitionBridge* Bridge_Init(void) {
    std::cout << "[Bridge] Forging the Neural Conduit (" << SHARED_MEM_NAME << ")...\n";

    // Create shared memory object
    shm_fd = shm_open(SHARED_MEM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        throw std::runtime_error("[Bridge] Failed to open shared memory.");
    }

    // Set the size
    if (ftruncate(shm_fd, sizeof(CognitionBridge)) == -1) {
        throw std::runtime_error("[Bridge] Failed to set shared memory size.");
    }

    // Map into process memory
    void* ptr = mmap(0, sizeof(CognitionBridge), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("[Bridge] Failed to map shared memory.");
    }

    CognitionBridge* bridge = static_cast<CognitionBridge*>(ptr);

    // Initialize pointers if it's newly created
    // (In a real system, you'd use a semaphore or mutex to coordinate initialization)
    bridge->head = 0;
    bridge->tail = 0;

    std::cout << "[Bridge] Neural Conduit established and locked.\n";
    return bridge;
}

void Bridge_Push_Essence(CognitionBridge* bridge, uint64_t ts, uint64_t sig, float amp, float phase) {
    uint32_t current_tail = bridge->tail;
    uint32_t next_tail = (current_tail + 1) % COGNITION_QUEUE_SIZE;

    if (next_tail != bridge->head) {
        bridge->essences[current_tail].timestamp_ns = ts;
        bridge->essences[current_tail].signature = sig;
        bridge->essences[current_tail].wave_amplitude = amp;
        bridge->essences[current_tail].phase_angle = phase;

        // Memory barrier to ensure data is written before updating tail
        __sync_synchronize();
        bridge->tail = next_tail;
    }
}

void Bridge_Teardown(CognitionBridge* bridge) {
    if (bridge != MAP_FAILED && bridge != nullptr) {
        munmap(bridge, sizeof(CognitionBridge));
    }
    if (shm_fd != -1) {
        close(shm_fd);
        shm_unlink(SHARED_MEM_NAME);
    }
    std::cout << "[Bridge] Neural Conduit severed.\n";
}
