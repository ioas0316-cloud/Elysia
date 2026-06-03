#include <math.h>
#include <stdint.h>

#define PHI 1.618033988749895
#define BASE_FREQ 0.5
#define PI 3.14159265358979323846

// States:
// 0=Equilibrium, 1=Meditation, 2=Germination, 3=Explosion, 4=Suppression

typedef struct {
    double freq_0;
    double phase_0;
    double freq_1;
    double phase_1;
    double perturbation;
} RotorState;

void init_rotor(RotorState* state) {
    state->freq_0 = BASE_FREQ;
    state->phase_0 = 0.0;
    state->freq_1 = BASE_FREQ * PHI;
    state->phase_1 = PI / 2.0;
    state->perturbation = 0.0;
}

void apply_stimulus(RotorState* state) {
    state->perturbation += PI / 1.5;
}

void decay_perturbation(RotorState* state) {
    if (state->perturbation > 0.01) {
        state->perturbation *= 0.9;
    } else {
        state->perturbation = 0.0;
    }
}

typedef struct {
    uint8_t b0;
    uint8_t b1;
    uint8_t i_and;
    uint8_t i_xor;
    uint8_t state_code;
} RotorOutput;

RotorOutput tick(RotorState* state, double t) {
    decay_perturbation(state);

    double w0 = sin(2.0 * PI * state->freq_0 * t + state->phase_0);
    double w1 = sin(2.0 * PI * state->freq_1 * t + state->phase_1 + state->perturbation);

    uint8_t b0 = (uint8_t)((w0 + 1.0) * 127.5);
    uint8_t b1 = (uint8_t)((w1 + 1.0) * 127.5);

    uint8_t i_and = b0 & b1;
    uint8_t i_xor = b0 ^ b1;

    uint8_t state_code = 0; // Equilibrium
    if (i_and > 180) {
        state_code = 4; // Suppression
    } else if (i_xor > 200) {
        state_code = 3; // Explosion
    } else if (i_xor > 128) {
        state_code = 2; // Germination
    } else if (i_and < 50) {
        state_code = 1; // Meditation
    }

    RotorOutput out;
    out.b0 = b0;
    out.b1 = b1;
    out.i_and = i_and;
    out.i_xor = i_xor;
    out.state_code = state_code;

    return out;
}
