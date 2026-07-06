#pragma once

#include <Kokkos_Core.hpp>

struct LBM {
    Kokkos::View<double**> rho;
    Kokkos::View<double***> f;
    Kokkos::View<double***> v;

    Kokkos::View<bool**> wall;

    int rows;
    int cols;

  // LBM(int rows, int cols) : rho("rho", rows, cols), f("f", rows, cols, 9), v("v", rows, cols) {}
};

LBM create_lbm(int rows, int cols);
void print_lbm_message();
void stream_lbm(LBM& lbm);
void compute_density(LBM& lbm);
void compute_velocity(LBM& lbm);
void write_output_f(const LBM& lbm, const std::string& filename);
void write_output_rho(const LBM& lbm, const std::string& filename);
void write_output_velocity(const LBM& lbm, const std::string& filename);
Kokkos::View<double***> compute_equilibrium(LBM& lbm);
void collision_step(LBM& lbm);
void initialize_density_bump(LBM& lbm);
double compute_total_mass(const LBM& lbm);
void initialize_velocity_bump(LBM& lbm);
void initialize_fixed_point(LBM& lbm);
void initialize_shear_wave(LBM &lbm);
void create_walls(LBM &lbm);
void initialize_eq_conditions(LBM& lbm);
void move_top_wall(LBM& lbm, double u_lid);