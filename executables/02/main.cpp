/**
 * Lattice Boltzmann Method — D2Q9 Streaming Milestone (FIXED)
 * =============================================================
 * Fix vs. previous version: g_cx/g_cy were declared at NAMESPACE
 * SCOPE (i.e. global Kokkos::View objects). Globals are destroyed
 * AFTER main() returns, which is AFTER Kokkos::finalize() runs.
 * That caused:
 *     Kokkos allocation "g_cy" is being deallocated after
 *     Kokkos::finalize was called
 *
 * Fix: cx/cy are now local Views, declared inside the same
 * { ... } scope block as every other View, and passed explicitly
 * into the functions that need them. This guarantees ALL Kokkos
 * Views are destroyed BEFORE Kokkos::finalize() is called.
 *
 * Everything else (physics, streaming, grid size, spec) is
 * unchanged from the previous version.
 *
 * Grid: NX=15 (x-direction), NY=10 (y-direction)
 *
 * D2Q9 velocity set:
 *   i:  0   1   2   3   4   5   6   7   8
 *  cx:  0   1   0  -1   0   1  -1  -1   1
 *  cy:  0   0   1   0  -1   1   1  -1  -1
 *
 * Build (OpenMP backend):
 *   cmake -DKokkos_ENABLE_OPENMP=ON ..
 *   make -j$(nproc)
 *
 * Build (CUDA backend):
 *   cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON ..
 */

#include <Kokkos_Core.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// ═══════════════════════════════════════════════════════════
// D2Q9 lattice parameters
// ═══════════════════════════════════════════════════════════
static constexpr int Q  = 9;   // number of velocity directions
static constexpr int NX = 15;  // lattice points in x  (width)
static constexpr int NY = 10;  // lattice points in y  (length)

// D2Q9 velocity vectors  c_i = (CX[i], CY[i])
//   0=rest, 1=+x, 2=+y, 3=-x, 4=-y, 5=+x+y, 6=-x+y, 7=-x-y, 8=+x-y
static constexpr int CX[Q] = {  0,  1,  0, -1,  0,  1, -1, -1,  1 };
static constexpr int CY[Q] = {  0,  0,  1,  0, -1,  1,  1, -1, -1 };

// D2Q9 equilibrium weights  w_i  (unused this milestone, kept for reference)
static constexpr double W[Q] = {
    4.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

// ═══════════════════════════════════════════════════════════
// Kokkos View type aliases
// ═══════════════════════════════════════════════════════════
using View3D  = Kokkos::View<double***>;   // f(x, y, i)
using View2D  = Kokkos::View<double**>;    // scalar field (x, y)
using View1Di = Kokkos::View<int*>;        // 1-D device array for lattice vectors

// ═══════════════════════════════════════════════════════════
// compute_density :  rho(x,y) = sum_i f(x,y,i)
// ═══════════════════════════════════════════════════════════
void compute_density(const View3D& f, View2D& rho) {
    Kokkos::parallel_for(
        "compute_density",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NX, NY}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double sum = 0.0;
            for (int i = 0; i < Q; ++i)
                sum += f(x, y, i);
            rho(x, y) = sum;
        }
    );
    Kokkos::fence();
}

// ═══════════════════════════════════════════════════════════
// compute_velocity :  v(x,y) = (1/rho) * sum_i f(x,y,i) * c_i
// cx/cy are now passed IN (no global capture).
// ═══════════════════════════════════════════════════════════
void compute_velocity(const View3D& f, const View2D& rho,
                      View2D& vx, View2D& vy,
                      const View1Di& cx, const View1Di& cy) {
    Kokkos::parallel_for(
        "compute_velocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NX, NY}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double sumx = 0.0, sumy = 0.0;
            for (int i = 0; i < Q; ++i) {
                sumx += f(x, y, i) * cx(i);
                sumy += f(x, y, i) * cy(i);
            }
            const double r = rho(x, y);
            vx(x, y) = (r > 1.0e-12) ? sumx / r : 0.0;
            vy(x, y) = (r > 1.0e-12) ? sumy / r : 0.0;
        }
    );
    Kokkos::fence();
}

// ═══════════════════════════════════════════════════════════
// streaming :  f_i(r + c_i*dt, t+dt) = f_i(r, t)     [periodic BC]
// Push-scheme, double-buffered. cx/cy passed in explicitly.
// ═══════════════════════════════════════════════════════════
void streaming(View3D& f, View3D& f_tmp,
              const View1Di& cx, const View1Di& cy) {
    // Zero the target buffer first (collision = 0 → no in-place update)
    Kokkos::deep_copy(f_tmp, 0.0);

    Kokkos::parallel_for(
        "streaming",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NX, NY, Q}),
        KOKKOS_LAMBDA(const int x, const int y, const int i) {
            // Destination with periodic wrap
            const int dst_x = (x + cx(i) + NX) % NX;
            const int dst_y = (y + cy(i) + NY) % NY;
            f_tmp(dst_x, dst_y, i) = f(x, y, i);
            // C(f) = 0: no collision term added — pure transport
        }
    );
    Kokkos::fence();

    // Swap buffers: f now holds the post-streaming state
    Kokkos::deep_copy(f, f_tmp);
}

// ═══════════════════════════════════════════════════════════
// initialize
//   Localized "explosion": a single lattice point gets a strong
//   density pulse split across all 8 non-rest directions, while
//   the rest of the grid sits at a low uniform baseline. Under
//   pure streaming (collision OFF) with periodic BCs, this pulse
//   spreads outward each step and wraps around the domain edges
//   -- a clean, visually obvious way to see periodic wraparound
//   in the density field.
// ═══════════════════════════════════════════════════════════
void initialize(View3D& f) {
    auto h_f = Kokkos::create_mirror_view(f);

    const int cx0 = NX / 2;   // spike location
    const int cy0 = NY / 2;

    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            for (int i = 0; i < Q; ++i)
                h_f(x, y, i) = 0.1;   // low uniform baseline

            if (x == cx0 && y == cy0) {
                h_f(x, y, 1) = 5.0;   // +x
                h_f(x, y, 2) = 5.0;   // +y
                h_f(x, y, 3) = 5.0;   // -x
                h_f(x, y, 4) = 5.0;   // -y
                h_f(x, y, 5) = 4.0;   // +x+y
                h_f(x, y, 6) = 4.0;   // -x+y
                h_f(x, y, 7) = 4.0;   // -x-y
                h_f(x, y, 8) = 4.0;   // +x-y
                h_f(x, y, 0) = 0.1;   // rest population stays baseline
            }
        }
    }

    Kokkos::deep_copy(f, h_f);
}

// ═══════════════════════════════════════════════════════════
// I/O helpers
// ═══════════════════════════════════════════════════════════
void write_density_csv(const View2D& rho, const std::string& filename) {
    auto h = Kokkos::create_mirror_view(rho);
    Kokkos::deep_copy(h, rho);
    std::ofstream out(filename);
    out << std::fixed << std::setprecision(8);
    out << "x,y,value\n";
    for (int x = 0; x < NX; ++x)
        for (int y = 0; y < NY; ++y)
            out << x << "," << y << "," << h(x, y) << "\n";
}

void write_velocity_csv(const View2D& vx_v, const View2D& vy_v,
                        const std::string& filename) {
    auto hx = Kokkos::create_mirror_view(vx_v);
    auto hy = Kokkos::create_mirror_view(vy_v);
    Kokkos::deep_copy(hx, vx_v);
    Kokkos::deep_copy(hy, vy_v);
    std::ofstream out(filename);
    out << std::fixed << std::setprecision(8);
    out << "x,y,vx,vy\n";
    for (int x = 0; x < NX; ++x)
        for (int y = 0; y < NY; ++y)
            out << x << "," << y << "," << hx(x,y) << "," << hy(x,y) << "\n";
}

void write_f_csv(const View3D& f, const std::string& filename) {
    auto h = Kokkos::create_mirror_view(f);
    Kokkos::deep_copy(h, f);
    std::ofstream out(filename);
    out << std::fixed << std::setprecision(8);
    out << "x,y,i,f\n";
    for (int x = 0; x < NX; ++x)
        for (int y = 0; y < NY; ++y)
            for (int i = 0; i < Q; ++i)
                if (h(x, y, i) != 0.0)
                    out << x << "," << y << "," << i << "," << h(x,y,i) << "\n";
}

// ═══════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int N_STEPS  = 30;
        const int OUT_FREQ = 1;

        // ── Lattice vectors: LOCAL Views, scoped with everything else ──
        View1Di cx("cx", Q), cy("cy", Q);
        auto h_cx = Kokkos::create_mirror_view(cx);
        auto h_cy = Kokkos::create_mirror_view(cy);
        for (int i = 0; i < Q; ++i) { h_cx(i) = CX[i]; h_cy(i) = CY[i]; }
        Kokkos::deep_copy(cx, h_cx);
        Kokkos::deep_copy(cy, h_cy);

        // ── Field Views ──
        View3D f    ("f",     NX, NY, Q);
        View3D f_tmp("f_tmp", NX, NY, Q);
        View2D rho  ("rho",   NX, NY);
        View2D vx   ("vx",    NX, NY);
        View2D vy   ("vy",    NX, NY);

        initialize(f);

        std::cout << "LBM D2Q9 | Streaming milestone\n"
                  << "  Grid    : " << NX << " x " << NY << "\n"
                  << "  Steps   : " << N_STEPS << "\n"
                  << "  C(f)    : 0  (pure streaming, no collision)\n"
                  << "  BC      : periodic\n\n";

        for (int t = 0; t <= N_STEPS; ++t) {
            compute_density(f, rho);
            compute_velocity(f, rho, vx, vy, cx, cy);

            if (t % OUT_FREQ == 0) {
                const std::string tag = std::to_string(t);
                write_density_csv (rho,    "rho_t" + tag + ".csv");
                write_velocity_csv(vx, vy, "vel_t" + tag + ".csv");
                write_f_csv       (f,      "f_t"   + tag + ".csv");
                std::cout << "  t=" << std::setw(3) << t
                          << "  -> rho, vel, f CSVs written\n";
            }

            if (t < N_STEPS)
                streaming(f, f_tmp, cx, cy);
        }

        std::cout << "\nDone.\n";
        // ── Scope closes here: cx, cy, f, f_tmp, rho, vx, vy all
        //    destroyed BEFORE Kokkos::finalize() below runs. ──
    }
    Kokkos::finalize();
    return 0;
}