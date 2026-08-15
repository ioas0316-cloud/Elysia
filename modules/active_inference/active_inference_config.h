#ifndef ACTIVE_INFERENCE_CONFIG_H
#define ACTIVE_INFERENCE_CONFIG_H

#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace active_inference {

struct ActiveInferenceConfig {
    // General / Euler Step
    float dt = 0.005f;           // Integration time step (Delta t)

    // 1D Physical parameters
    float alpha = 1.0f;          // Damping coefficient (1D physical environment)

    // 2nd-order (2D generalized coordinates: position & velocity) physical parameters
    float mass = 1.0f;           // Mass (m)
    float gamma = 2.0f;          // Damping coefficient (2D system)

    // 1D Precisions & Learning Rates
    float pi_y = 10.0f;          // Sensory precision
    float pi_p = 2.0f;           // Prior / target precision
    float lr_mu = 2.0f;          // Belief learning rate
    float lr_a = 5.0f;           // Action learning rate

    // 2nd-order (2D position & velocity) Precisions & Learning Rates
    float pi_y0 = 10.0f;         // Position sensory precision
    float pi_y1 = 5.0f;          // Velocity sensory precision
    float pi_p0 = 3.0f;          // Position prior precision
    float pi_p1 = 2.0f;          // Velocity prior precision

    float lr_mu0 = 2.0f;         // Position belief learning rate
    float lr_mu1 = 2.0f;         // Velocity belief learning rate
    float lr_a_2d = 10.0f;       // 2D Action learning rate

    // Clamping thresholds
    bool enable_clamping = true;
    float min_state = -100.0f;   // Lower bound for state and belief (mu, x, v, a)
    float max_state = 100.0f;    // Upper bound for state and belief
    float min_deriv = -50.0f;    // Lower bound for derivatives (d_mu, d_a, dv, dx)
    float max_deriv = 50.0f;     // Upper bound for derivatives

    // Helper for clamping values
    inline float clamp_state(float val) const {
        if (!enable_clamping) return val;
        return std::max(min_state, std::min(max_state, val));
    }

    inline float clamp_deriv(float val) const {
        if (!enable_clamping) return val;
        return std::max(min_deriv, std::min(max_deriv, val));
    }

    // Dynamic config loader from simple key-value file
    bool load_from_file(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[ActiveInferenceConfig] Failed to open file: " << filepath << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            // Ignore comments and empty lines
            if (line.empty() || line[0] == '#' || line[0] == '/') continue;

            std::stringstream ss(line);
            std::string key, val_str;
            if (std::getline(ss, key, '=') && std::getline(ss, val_str)) {
                // Trim key and value
                key.erase(0, key.find_first_not_of(" \t\""));
                key.erase(key.find_last_not_of(" \t\"") + 1);
                val_str.erase(0, val_str.find_first_not_of(" \t\",:"));
                val_str.erase(val_str.find_last_not_of(" \t\",:") + 1);

                try {
                    float val = std::stof(val_str);
                    set_value(key, val);
                } catch (...) {
                    if (key == "enable_clamping") {
                        enable_clamping = (val_str == "true" || val_str == "1");
                    }
                }
            }
        }
        return true;
    }

    void set_value(const std::string& key, float val) {
        if (key == "dt") dt = val;
        else if (key == "alpha") alpha = val;
        else if (key == "mass") mass = val;
        else if (key == "gamma") gamma = val;
        else if (key == "pi_y") pi_y = val;
        else if (key == "pi_p") pi_p = val;
        else if (key == "lr_mu") lr_mu = val;
        else if (key == "lr_a") lr_a = val;
        else if (key == "pi_y0") pi_y0 = val;
        else if (key == "pi_y1") pi_y1 = val;
        else if (key == "pi_p0") pi_p0 = val;
        else if (key == "pi_p1") pi_p1 = val;
        else if (key == "lr_mu0") lr_mu0 = val;
        else if (key == "lr_mu1") lr_mu1 = val;
        else if (key == "lr_a_2d") lr_a_2d = val;
        else if (key == "min_state") min_state = val;
        else if (key == "max_state") max_state = val;
        else if (key == "min_deriv") min_deriv = val;
        else if (key == "max_deriv") max_deriv = val;
    }
};

} // namespace active_inference

#endif // ACTIVE_INFERENCE_CONFIG_H
