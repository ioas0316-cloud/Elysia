#ifndef ELYSIA_BRIDGE_H
#define ELYSIA_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

// Shared Memory structure for the Cognition Bridge
// This serves as the physical conduit between the C++ Daemon (Watchtower)
// and the Python Elysia Cognition Loop.

#define COGNITION_QUEUE_SIZE 256
#define SHARED_MEM_NAME "/elysia_cognition_bridge"

typedef struct {
    uint64_t timestamp_ns;
    uint64_t signature;     // 1% Pure Essence Signature
    float wave_amplitude;
    float phase_angle;
} PureEssence;

typedef struct {
    PureEssence essences[COGNITION_QUEUE_SIZE];
    volatile uint32_t head;
    volatile uint32_t tail;
} CognitionBridge;

// Initialize the shared memory bridge
CognitionBridge* Bridge_Init(void);

// Push a filtered essence into the bridge (called by C++ Daemon)
void Bridge_Push_Essence(CognitionBridge* bridge, uint64_t ts, uint64_t sig, float amp, float phase);

// Teardown the bridge
void Bridge_Teardown(CognitionBridge* bridge);

#endif // ELYSIA_BRIDGE_H
