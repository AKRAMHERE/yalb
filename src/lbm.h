#pragma once

#include <Kokkos_Core.hpp>

// Host-pinned memory space used to stage MPI halo buffers for fast
// device<->host transfers. Kokkos::CudaHostPinnedSpace only exists when
// Kokkos is built with CUDA enabled (e.g. on bwUniCluster's A100 nodes);
// falling back to plain HostSpace keeps the code buildable on CPU-only
// machines (local dev, CI) without changing GPU-build behavior.
#if defined(KOKKOS_ENABLE_CUDA)
using HostPinnedSpace = Kokkos::CudaHostPinnedSpace;
#else
using HostPinnedSpace = Kokkos::HostSpace;
#endif

struct LBM {
    Kokkos::View<double**> rho;
    Kokkos::View<double***, Kokkos::LayoutRight> f;
    Kokkos::View<double***, Kokkos::LayoutRight> f_next;
    Kokkos::View<double***> v;

    Kokkos::View<bool**> wall;

    int rows;
    int cols;

    Kokkos::View<double*> send_lower;
    Kokkos::View<double*> send_upper;
    Kokkos::View<double*> recv_lower;
    Kokkos::View<double*> recv_upper;

    Kokkos::View<double*, HostPinnedSpace> send_lower_host;
    Kokkos::View<double*, HostPinnedSpace> send_upper_host;
    Kokkos::View<double*, HostPinnedSpace> recv_lower_host;
    Kokkos::View<double*, HostPinnedSpace> recv_upper_host;

  // LBM(int rows, int cols) : rho("rho", rows, cols), f("f", rows, cols, 9), v("v", rows, cols) {}
};

LBM create_lbm(int rows, int cols);
void print_lbm_message();
void stream_lbm(LBM& lbm, double u_lid, int local_start, int global_rows);
void compute_density(LBM& lbm);
void compute_velocity(LBM& lbm);
void write_output_f(const LBM& lbm, const std::string& filename);
void write_output_rho(const LBM& lbm, const std::string& filename, int local_start);
void write_output_velocity(const LBM& lbm, const std::string& filename, int local_start);
Kokkos::View<double***, Kokkos::LayoutRight> compute_equilibrium(LBM& lbm);
void collision_step(LBM& lbm);
void initialize_density_bump(LBM& lbm);
double compute_total_mass(const LBM& lbm);
void initialize_velocity_bump(LBM& lbm);
void initialize_fixed_point(LBM& lbm);
void initialize_shear_wave(LBM &lbm);
void create_walls(LBM &lbm, int local_start, int global_rows);
void initialize_eq_conditions(LBM& lbm);
void move_top_wall(LBM& lbm, double u_lid, int local_start, int global_rows);
void exchange_halos(LBM& lbm, int rank, int size);
void exchange_halos_host(LBM& lbm, int rank, int size);
void stream_lbm_pull(LBM& lbm, double u_lid, int local_start, int global_rows);
double compute_local_fluid_mass(const LBM& lbm);
void collision_and_stream(LBM& lbm, double u_lid, int local_start, int global_rows);
