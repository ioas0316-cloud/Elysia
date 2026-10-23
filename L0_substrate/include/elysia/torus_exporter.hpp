#ifndef ELYSIA_TORUS_EXPORTER_HPP
#define ELYSIA_TORUS_EXPORTER_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>

namespace elysia {

struct Point3D { float x, y, z; };

struct TorusPoint2D { double theta, phi; };
struct Vector2D { double dtheta, dphi; };

// Exporter for deformed 3D Torus mesh (.obj) and Geodesic trajectory (.csv)
class TorusMeshExporter {
private:
    float R{3.0f};           // Major radius
    float r{1.0f};           // Minor radius
    float theta_0{1.5708f};  // Attractor position theta (pi / 2)
    float phi_0{3.1415f};    // Attractor position phi (pi)
    float k_strength{0.8f};  // Gravity deformation depth
    float alpha{1.5f};       // Localized influence range

public:
    TorusMeshExporter(float major_r = 3.0f, float minor_r = 1.0f, float strength = 0.8f)
        : R(major_r), r(minor_r), k_strength(strength) {}

    Point3D compute_deformed_torus_point(float theta, float phi) const {
        float dist_sq = std::pow(theta - theta_0, 2) + std::pow(phi - phi_0, 2);
        float local_r = r - k_strength * std::exp(-alpha * dist_sq);

        float x = (R + local_r * std::cos(phi)) * std::cos(theta);
        float y = (R + local_r * std::cos(phi)) * std::sin(theta);
        float z = local_r * std::sin(phi);

        return {x, y, z};
    }

    bool export_to_obj(const std::string& filename, int grid_theta = 60, int grid_phi = 30) const {
        std::ofstream obj_file(filename);
        if (!obj_file.is_open()) return false;

        float d_theta = 2.0f * M_PI / grid_theta;
        float d_phi   = 2.0f * M_PI / grid_phi;

        // Write Vertices
        for (int i = 0; i < grid_theta; ++i) {
            float theta = i * d_theta;
            for (int j = 0; j < grid_phi; ++j) {
                float phi = j * d_phi;
                Point3D p = compute_deformed_torus_point(theta, phi);
                obj_file << "v " << p.x << " " << p.y << " " << p.z << "\n";
            }
        }

        // Write Quad Faces
        for (int i = 0; i < grid_theta; ++i) {
            for (int j = 0; j < grid_phi; ++j) {
                int next_i = (i + 1) % grid_theta;
                int next_j = (j + 1) % grid_phi;

                int idx1 = i * grid_phi + j + 1;
                int idx2 = next_i * grid_phi + j + 1;
                int idx3 = next_i * grid_phi + next_j + 1;
                int idx4 = i * grid_phi + next_j + 1;

                obj_file << "f " << idx1 << " " << idx2 << " " << idx3 << " " << idx4 << "\n";
            }
        }

        obj_file.close();
        return true;
    }

    static bool export_geodesic_csv(const std::string& filename, const std::vector<Point3D>& trajectory) {
        std::ofstream csv_file(filename);
        if (!csv_file.is_open()) return false;

        csv_file << "step,x,y,z\n";
        for (size_t i = 0; i < trajectory.size(); ++i) {
            csv_file << i << "," << trajectory[i].x << "," << trajectory[i].y << "," << trajectory[i].z << "\n";
        }

        csv_file.close();
        return true;
    }
};

// Torus Attractor Geodesic Particle Simulator
class TorusAttractorSimulator {
private:
    double R{3.0};
    double r{1.0};
    double theta_0{1.5708}; // pi / 2
    double phi_0{3.1415};   // pi
    double k_strength{2.5};
    double alpha{1.5};

public:
    double g_tt(double theta, double phi) const {
        double dist_sq = std::pow(theta - theta_0, 2) + std::pow(phi - phi_0, 2);
        return std::pow(R + r * std::cos(phi), 2) + k_strength * std::exp(-alpha * dist_sq);
    }

    double g_pp(double theta, double phi) const {
        double dist_sq = std::pow(theta - theta_0, 2) + std::pow(phi - phi_0, 2);
        return std::pow(r, 2) + k_strength * std::exp(-alpha * dist_sq);
    }

    void compute_christoffel(double theta, double phi, double Gamma[2][2][2]) const {
        double eps = 1e-5;

        double dg_tt_dtheta = (g_tt(theta + eps, phi) - g_tt(theta - eps, phi)) / (2.0 * eps);
        double dg_tt_dphi   = (g_tt(theta, phi + eps) - g_tt(theta, phi - eps)) / (2.0 * eps);
        double dg_pp_dtheta = (g_pp(theta + eps, phi) - g_pp(theta - eps, phi)) / (2.0 * eps);
        double dg_pp_dphi   = (g_pp(theta, phi + eps) - g_pp(theta, phi - eps)) / (2.0 * eps);

        double g_tt_inv = 1.0 / g_tt(theta, phi);
        double g_pp_inv = 1.0 / g_pp(theta, phi);

        Gamma[0][0][0] = 0.5 * g_tt_inv * dg_tt_dtheta;
        Gamma[0][0][1] = 0.5 * g_tt_inv * dg_tt_dphi;
        Gamma[0][1][0] = Gamma[0][0][1];
        Gamma[0][1][1] = -0.5 * g_tt_inv * dg_pp_dtheta;

        Gamma[1][0][0] = -0.5 * g_pp_inv * dg_tt_dphi;
        Gamma[1][0][1] = 0.5 * g_pp_inv * dg_pp_dtheta;
        Gamma[1][1][0] = Gamma[1][0][1];
        Gamma[1][1][1] = 0.5 * g_pp_inv * dg_pp_dphi;
    }

    std::vector<Point3D> simulate_trajectory(TorusPoint2D start_pos, Vector2D start_vel, int steps, double dt) const {
        TorusPoint2D pos = start_pos;
        Vector2D vel = start_vel;
        std::vector<Point3D> trajectory;

        TorusMeshExporter exporter(R, r, 0.8f);

        for (int i = 0; i < steps; ++i) {
            double Gamma[2][2][2] = {{{0.0}}};
            compute_christoffel(pos.theta, pos.phi, Gamma);

            double acc_theta = -(Gamma[0][0][0] * vel.dtheta * vel.dtheta +
                                 2.0 * Gamma[0][0][1] * vel.dtheta * vel.dphi +
                                 Gamma[0][1][1] * vel.dphi * vel.dphi);

            double acc_phi   = -(Gamma[1][0][0] * vel.dtheta * vel.dtheta +
                                 2.0 * Gamma[1][0][1] * vel.dtheta * vel.dphi +
                                 Gamma[1][1][1] * vel.dphi * vel.dphi);

            pos.theta += vel.dtheta * dt;
            pos.phi   += vel.dphi * dt;
            vel.dtheta += acc_theta * dt;
            vel.dphi   += acc_phi * dt;

            Point3D p3d = exporter.compute_deformed_torus_point(static_cast<float>(pos.theta), static_cast<float>(pos.phi));
            trajectory.push_back(p3d);
        }
        return trajectory;
    }
};

} // namespace elysia

#endif // ELYSIA_TORUS_EXPORTER_HPP
