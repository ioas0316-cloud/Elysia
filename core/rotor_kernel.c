#include <uapi/linux/bpf.h>
#include <linux/in.h>
#include <linux/if_ether.h>
#include <linux/ip.h>

// A BPF map to store the timestamp of the last seen packet.
// Key is 0, Value is the timestamp in nanoseconds.
BPF_ARRAY(last_ts, u64, 1);

// A BPF perf buffer to send events (inter-arrival times) to user space.
struct packet_event_t {
    u64 delta_ns;
    u64 current_ts_ns;
    int phase_shift_applied;
};
BPF_PERF_OUTPUT(events);

// The 100 microseconds (μs) target frequency in nanoseconds
#define TARGET_FREQUENCY_NS 100000

// eBPF XDP Program
int sunlight_prism(struct xdp_md *ctx) {
    u64 current_ts = bpf_ktime_get_ns();
    int key = 0;
    u64 *prev_ts = last_ts.lookup(&key);

    struct packet_event_t event = {};
    event.current_ts_ns = current_ts;
    event.phase_shift_applied = 0;

    if (prev_ts) {
        u64 delta = current_ts - *prev_ts;
        event.delta_ns = delta;

        // "Refraction" logic
        // If the packet arrives too fast (interval < 100μs), we could conceptually
        // apply a delay or a spin-lock here to enforce the 100μs rhythm.
        // However, standard eBPF XDP does not allow blocking sleeps (spin-locks
        // for delaying packet processing are generally forbidden or tightly restricted
        // by the verifier).
        // Instead, we will simulate the "refraction" by logging the tension/deviation
        // and optionally dropping (XDP_DROP) or passing (XDP_PASS) the packet based
        // on whether it hits the exact target resonance.

        // For now, we act as a true prism: we observe the deviation and let it pass,
        // recording if a phase shift *would* be applied to align it to 100μs.
        if (delta < TARGET_FREQUENCY_NS) {
            // "Applying tension": The packet arrived earlier than the 100μs cycle.
            // We record that it needed refraction.
            event.phase_shift_applied = 1;
        } else if (delta > TARGET_FREQUENCY_NS * 2) {
            // "Expansion": The gap is too large, the rhythm is broken.
            event.phase_shift_applied = -1;
        }
    } else {
        event.delta_ns = 0;
    }

    // Update the timestamp for the next packet
    last_ts.update(&key, &current_ts);

    // Send the observation to the user space "Expansion Log"
    events.perf_submit(ctx, &event, sizeof(event));

    // Let the packet flow through the gateway
    return XDP_PASS;
}
