#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// The pure physical ring buffer size.
// The stream continuously flows and wraps around, creating a topological torus.
#define FIELD_SIZE 1024

// Represents a single point in the variable axis field (the memory ring buffer).
// We do not use floating point numbers, angles, or mathematical averaging.
// The state is purely the physical bit configuration caused by XOR collisions.
typedef struct {
    uint8_t state;
} Node;

typedef struct {
    Node nodes[FIELD_SIZE];
    int head; // The current position of the flowing stream
} TopologyField;

// Initializes the field to a zero-energy state.
void init_field(TopologyField* field) {
    memset(field->nodes, 0, sizeof(Node) * FIELD_SIZE);
    field->head = 0;
}

// Feeds raw ASCII bytes into the field.
// The incoming byte physically collides (XOR) with the current state of the field.
// There are no conditional statements (if/else), thresholds, or parsing logic.
// The collision inherently alters the axis (the bits), naturally recording the trajectory.
void apply_stimulus(TopologyField* field, uint8_t byte_val) {
    // 1. Current state of the axis
    uint8_t current_state = field->nodes[field->head].state;

    // 2. Physical collision (XOR lattice reaction)
    // The incoming raw ASCII bit pattern twists the existing bit pattern.
    // This is the resonance/perturbation happening at the structural level.
    uint8_t next_state = current_state ^ byte_val;

    // 3. Torque (Rotation)
    // The axis rotates physically. We represent this via a bitwise circular shift.
    // The bits physically rotate left by 1.
    next_state = (next_state << 1) | (next_state >> 7);

    // 4. Record the twisted state back into the field
    field->nodes[field->head].state = next_state;

    // 5. The stream flows forward (Ring Buffer)
    // The head naturally advances to the next node in the torus.
    field->head = (field->head + 1) % FIELD_SIZE;
}

// Expose internal state for the Python mirror
// The mirror will strictly READ these values without altering them.
uint8_t get_node_state(const TopologyField* field, int index) {
    if (index >= 0 && index < FIELD_SIZE) {
        return field->nodes[index].state;
    }
    return 0;
}

int get_head_position(const TopologyField* field) {
    return field->head;
}
