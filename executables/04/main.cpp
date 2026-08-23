#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdlib>

constexpr int NVEL = 9;

int main(int argc, char* argv[]) {
    // -----------------------------------------------------------------
    // From Milestone 03: configurable relaxation parameter tau.
    // Optional CLI override:  ./milestone03 <tau>
    // omega = 1/tau must satisfy 0 < omega < 2  =>  tau > 0.5 for stability.
    // We only accept argv[1] as tau if it parses to a positive number and does not look like a flag (so Kokkos-prefixed args are left alone).
    // -----------------------------------------------------------------
    double tau = 1.0; // default relaxation time (omega = 1.0)
    int mode = 1; //default wavelength


    if (argc > 1)
        tau = std::atof(argv[1]);

    if (argc > 2)
        mode = std::atoi(argv[2]);

    if (tau <= 0.5) {
        std::cerr << "tau must be > 0.5\n";
        return 1;
    }

    Kokkos::initialize(argc, argv);
    {
        const int NX = 64;
        const int NY = 64;
        const int total_steps = 500;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Relaxation time tau = " << tau
                   << "  (omega = " << 1.0 / tau << ")\n";

        // D2Q9 Lattice Velocities
        int h_cx[NVEL] = {0, 1, 0, -1,  0, 1, -1, -1,  1};
        int h_cy[NVEL] = {0, 0, 1,  0, -1, 1,  1, -1, -1};

        // D2Q9 Lattice Weights (From Milestone 03): w0=4/9, w1-w4=1/9, w5-w8=1/36
        double h_w[NVEL] = {4.0 / 9.0,
                             1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                             1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

        Kokkos::View<int[NVEL]>    cx("cx"), cy("cy");
        Kokkos::View<double[NVEL]> w("w");
        auto host_cx = Kokkos::create_mirror_view(cx);
        auto host_cy = Kokkos::create_mirror_view(cy);
        auto host_w  = Kokkos::create_mirror_view(w);
        for (int i = 0; i < NVEL; ++i) {
            host_cx(i) = h_cx[i];
            host_cy(i) = h_cy[i];
            host_w(i)  = h_w[i];
        }
        Kokkos::deep_copy(cx, host_cx);
        Kokkos::deep_copy(cy, host_cy);
        Kokkos::deep_copy(w, host_w);

        using View3D = Kokkos::View<double***, Kokkos::LayoutRight>;
        using View2D = Kokkos::View<double**,  Kokkos::LayoutRight>;

        View3D f_src("f_src", NY, NX, NVEL);
        View3D f_dst("f_dst", NY, NX, NVEL);
        View2D rho("rho", NY, NX);
        View2D ux("ux", NY, NX);
        View2D uy("uy", NY, NX);

        auto f_host   = Kokkos::create_mirror_view(f_src);
        auto rho_host = Kokkos::create_mirror_view(rho);
        auto ux_host  = Kokkos::create_mirror_view(ux);
        auto uy_host  = Kokkos::create_mirror_view(uy);

        // --- Initialization: Omnidirectional distribution seeding changed from milestone 03; these are new ---
        const double rho_init = 1.0;
        const double eps = 0.08;

        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {

                double ux_init =
                    eps * std::sin(2.0 * M_PI * mode * y / NY);

                double uy_init = 0.0;

                for (int i = 0; i < NVEL; ++i) {

                    double cu =
                        h_cx[i] * ux_init + h_cy[i] * uy_init;

                    double usq =
                        ux_init * ux_init + uy_init * uy_init;

                    f_host(y,x,i) = h_w[i] * rho_init * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
                }
            }
        }
        Kokkos::deep_copy(f_src, f_host);

        double total_compute_time   = 0.0;
        double total_collision_time = 0.0;
        double total_stream_time    = 0.0;

        double initial_mass = 0.0;
        double final_mass   = 0.0;

        std::ofstream ampFile("amplitude.txt");

        auto sim_start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step <= total_steps; ++step) {
            // ---------------------------------------------------------
            // STEP 1: Compute rho and u from f_src (Milestone 02 kernel, unchanged)
            // ---------------------------------------------------------
            auto compute_start = std::chrono::high_resolution_clock::now();
            Kokkos::parallel_for("ComputeMoments", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NY, NX}),
            KOKKOS_LAMBDA(const int y, const int x) {
                double local_rho = 0.0;
                double local_ux  = 0.0;
                double local_uy  = 0.0;

                for (int i = 0; i < NVEL; ++i) {
                    double f_val = f_src(y, x, i);
                    local_rho += f_val;
                    local_ux  += f_val * cx(i);
                    local_uy  += f_val * cy(i);
                }

                rho(y, x) = local_rho;
                if (local_rho > 1e-10) {
                    ux(y, x) = local_ux / local_rho;
                    uy(y, x) = local_uy / local_rho;
                } else {
                    ux(y, x) = 0.0; uy(y, x) = 0.0;
                }
            });
            Kokkos::fence();
            double amplitude = 0.0;

            Kokkos::parallel_reduce(
            "Amplitude",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>
            ({0,0},{NY,NX}),
            KOKKOS_LAMBDA(
            const int y,
            const int x,
            double& max_val)
            {
                double val = fabs(ux(y,x));

                if(val > max_val)
                    max_val = val;
            },
            Kokkos::Max<double>(amplitude)
            );

            ampFile<< step << " "<< amplitude<< "\n";

            auto compute_end = std::chrono::high_resolution_clock::now();
            total_compute_time += std::chrono::duration<double>(compute_end - compute_start).count();

            // Mass-conservation bookkeeping (validation aid, see write-up)
            double step_mass = 0.0;
            Kokkos::parallel_reduce("TotalMass", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NY, NX}),
            KOKKOS_LAMBDA(const int y, const int x, double& mass_sum) {
                mass_sum += rho(y, x);
            }, step_mass);
            if (step == 0)           initial_mass = step_mass;
            if (step == total_steps) final_mass   = step_mass;

            Kokkos::deep_copy(rho_host, rho);
            Kokkos::deep_copy(ux_host, ux);
            Kokkos::deep_copy(uy_host, uy);
            /*
            double amplitude = 0.0;

            for (int y = 0; y < NY; ++y)
            {
                for (int x = 0; x < NX; ++x)
                {
                    amplitude =
                        std::max(amplitude,std::abs(ux_host(y,x)));
                }
            }*/

            //ampFile<< step << " " << amplitude<< "\n";

            std::ofstream outFile("lbm_04_step_" + std::to_string(step) + ".txt");
            outFile << "# y x rho ux uy\n";
            for (int y = 0; y < NY; ++y) {
                for (int x = 0; x < NX; ++x) {
                    outFile << y << " " << x << " " << rho_host(y, x) << " " << ux_host(y, x) << " " << uy_host(y, x) << "\n";
                }
            }
            outFile.close();

            if (step == total_steps) break;

            // ---------------------------------------------------------
            // STEP 2 + STEP 3: Equilibrium distribution + BGK collision (Milestone 03)
            //   feq_i  = w_i * rho * (1 + 3(c_i.u) + 4.5(c_i.u)^2 - 1.5|u|^2)
            //   f_post = f - (f - feq) / tau
            // f_src is updated IN PLACE. This is race-free: every (y,x,i)
            // entry depends only on its own current value and on rho/ux/uy
            // at (y,x), which were already fully computed and fenced in
            // Step 1 above. No thread ever reads or writes another
            // thread's f_src element, so there is no data race.
            // ---------------------------------------------------------
            auto collide_start = std::chrono::high_resolution_clock::now();
            Kokkos::parallel_for("Collision", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NY, NX, NVEL}),
            KOKKOS_LAMBDA(const int y, const int x, const int i) {
                double cu  = cx(i) * ux(y, x) + cy(i) * uy(y, x);
                double usq = ux(y, x) * ux(y, x) + uy(y, x) * uy(y, x);
                double feq = w(i) * rho(y, x) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);

                double f_old = f_src(y, x, i);
                f_src(y, x, i) = f_old - (f_old - feq) / tau;
            });
            Kokkos::fence();
            auto collide_end = std::chrono::high_resolution_clock::now();
            total_collision_time += std::chrono::duration<double>(collide_end - collide_start).count();

            // ---------------------------------------------------------
            // STEP 4: Streaming of POST-COLLISION distributions with periodic
            // wrap-around (Milestone 02 kernel, unchanged in code; it now
            // reads the collided f_src instead of the raw, pre-collision f_src)
            // ---------------------------------------------------------
            auto stream_start = std::chrono::high_resolution_clock::now();
            Kokkos::parallel_for("StreamingOp", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NY, NX, NVEL}),
            KOKKOS_LAMBDA(const int y, const int x, const int i) {
                int target_x = (x + cx(i) + NX) % NX;
                int target_y = (y + cy(i) + NY) % NY;
                f_dst(target_y, target_x, i) = f_src(y, x, i);
            });
            Kokkos::fence();
            auto stream_end = std::chrono::high_resolution_clock::now();
            total_stream_time += std::chrono::duration<double>(stream_end - stream_start).count();

            // ---------------------------------------------------------
            // STEP 5: swap buffers
            // ---------------------------------------------------------
            auto temp = f_src; f_src = f_dst; f_dst = temp;
        }
        auto sim_end = std::chrono::high_resolution_clock::now();
        double total_sim_time = std::chrono::duration<double>(sim_end - sim_start).count();

        std::cout << "Total Simulation Time: " << total_sim_time << " seconds\n";
        std::cout << "Average Compute Time per Step: "   << total_compute_time   / total_steps << " seconds\n";
        std::cout << "Average Collision Time per Step: " << total_collision_time / total_steps << " seconds\n";
        std::cout << "Average Stream Time per Step: "    << total_stream_time    / total_steps << " seconds\n";

        double mass_rel_error = (initial_mass > 0.0)
            ? std::fabs(final_mass - initial_mass) / initial_mass
            : 0.0;
        std::cout << "Initial total mass: " << initial_mass
                   << ", Final total mass: " << final_mass
                   << ", Relative change: " << mass_rel_error << "\n";
        ampFile.close();
    }
    
    Kokkos::finalize();
    std::cout << "Successfully executed BGK collision and streaming for Milestone 04." << std::endl;
    return 0;
}