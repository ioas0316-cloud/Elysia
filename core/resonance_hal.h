#ifndef RESONANCE_HAL_H
#define RESONANCE_HAL_H

#include <stdint.h>
#include <stddef.h>

// -----------------------------------------------------------------------------
// Zero-Copy Ring Buffer Definition (The Essence Queue)
// -----------------------------------------------------------------------------
// This structure is meant to map memory directly accessible by the GPU VRAM.
// In this PoC, we pass physical/virtual addresses without copying data.

#define RING_BUFFER_SIZE 1024

typedef struct {
    uint64_t data_addr;    // Memory address of the raw wave (Essence)
    uint16_t length;       // Length of the valid wave data
    uint64_t timestamp;    // Physical temporal marker (ns)
    uint32_t phase_shift;  // Applied rotation/tension delta
} EssenceWave;

typedef struct {
    EssenceWave waves[RING_BUFFER_SIZE];
    volatile uint32_t head;
    volatile uint32_t tail;
} ZeroCopyRingBuffer;

// -----------------------------------------------------------------------------
// Hardware Abstraction Layer (HAL) Interfaces
// -----------------------------------------------------------------------------

// Initialize the physical network pathway (DPDK / GPUDirect RDMA)
int HAL_Network_Init(int argc, char **argv);

// Allocate or map the zero-copy ring buffer shared with GPU VRAM
ZeroCopyRingBuffer* HAL_Memory_Init(void);

// Poll the NIC for new waves, bypass kernel, and push to the ring buffer
int HAL_Poll_And_Spin(ZeroCopyRingBuffer *ring_buffer);

// Shutdown the physical link gracefully
void HAL_Network_Teardown(void);

#endif // RESONANCE_HAL_H
