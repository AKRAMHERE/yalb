// m05_2906.cpp — Kokkos D2Q9 LBM Lid-Driven Cavity (Milestone 05)
//
// Structure mirrors the reference implementation (separate functions per step):
//   stream()        — pull scheme, clamp at boundaries
//   bounce_back()   — stationary walls: left, right, bottom
//   move_top_wall() — moving lid using rho reconstructed from known populations
//   collide()       — BGK in-place
//
// BC order: stream → bounce_back → move_top_wall → collide
// This matches the reference: streaming first, then BCs, then collision.
//
// Moving wall formula (confirmed by reference code):
//   rho_w = f0+f1+f3 + 2*(f2+f5+f6)   ← reconstruct from known post-stream pops
//   f4 = f2
//   f7 = f5 - (1/6)*rho_w*u_lid
//   f8 = f6 + (1/6)*rho_w*u_lid

#include <Kokkos_Core.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <chrono>

// ── Parameters ────────────────────────────────────────────────────────────────
static constexpr int    NX           = 256;
static constexpr int    NY           = 256;
static constexpr double OMEGA        = 1.7;
static constexpr double U_LID        = 0.1;
static constexpr double CONV         = 1e-6;
static constexpr int    MAXSTEP      = 300000;
static constexpr int    LOG_INTERVAL  = 5000;
static constexpr int    SNAP_INTERVAL = 2000;

// ── D2Q9 ──────────────────────────────────────────────────────────────────────
//  i:  0    1    2    3    4    5    6    7    8
// cx:  0    1    0   -1    0    1   -1   -1    1
// cy:  0    0    1    0   -1    1    1   -1   -1
KOKKOS_INLINE_FUNCTION constexpr int    CX (int i){const int    v[]={0,1,0,-1,0,1,-1,-1,1};return v[i];}
KOKKOS_INLINE_FUNCTION constexpr int    CY (int i){const int    v[]={0,0,1,0,-1,1,1,-1,-1};return v[i];}
KOKKOS_INLINE_FUNCTION constexpr int    OPP(int i){const int    v[]={0,3,4,1,2,7,8,5,6};  return v[i];}
KOKKOS_INLINE_FUNCTION constexpr double W  (int i){const double v[]={4./9,1./9,1./9,1./9,1./9,
                                                                      1./36,1./36,1./36,1./36};return v[i];}

KOKKOS_INLINE_FUNCTION double feq(int i, double rho, double ux, double uy){
    double cu  = CX(i)*ux + CY(i)*uy;
    double usq = ux*ux + uy*uy;
    return W(i) * rho * (1. + 3.*cu + 4.5*cu*cu - 1.5*usq);
}

using View3D = Kokkos::View<double***>;  // [NY][NX][9]
using View2D = Kokkos::View<double**>;   // [NY][NX]

// ═════════════════════════════════════════════════════════════════════════════
// stream — pull scheme
// f_next(y,x,i) = f_curr(y-cy, x-cx, i)
// Boundary cells are clamped here; BCs below will overwrite them.
// ═════════════════════════════════════════════════════════════════════════════
void stream(const View3D& f_curr, const View3D& f_next){
    Kokkos::parallel_for("stream",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{NY,NX}),
        KOKKOS_LAMBDA(int y, int x){
            for(int i=0;i<9;i++){
                int xs = x - CX(i);
                int ys = y - CY(i);
                // Clamp — boundary nodes will be overwritten by BC functions
                xs = (xs < 0) ? 0 : (xs >= NX ? NX-1 : xs);
                ys = (ys < 0) ? 0 : (ys >= NY ? NY-1 : ys);
                f_next(y,x,i) = f_curr(ys,xs,i);
            }
        });
    Kokkos::fence();
}

// ═════════════════════════════════════════════════════════════════════════════
// bounce_back — stationary walls (left, right, bottom)
// Eq.2: f_ī(x_b) = f_i*(x_b)
// Reads from f_curr (pre-streaming = f_i*), writes into f_next.
// ═════════════════════════════════════════════════════════════════════════════
void bounce_back(const View3D& f_curr, const View3D& f_next){
    // Bottom wall y=0: unknown incoming = channels 2,5,6 (cy > 0)
    Kokkos::parallel_for("bb_bottom", Kokkos::RangePolicy<>(0,NX),
        KOKKOS_LAMBDA(int x){
            f_next(0,x,2) = f_curr(0,x,OPP(2));  // f_curr(0,x,4)
            f_next(0,x,5) = f_curr(0,x,OPP(5));  // f_curr(0,x,7)
            f_next(0,x,6) = f_curr(0,x,OPP(6));  // f_curr(0,x,8)
        });

    // Left wall x=0: unknown incoming = channels 1,5,8 (cx > 0)
    Kokkos::parallel_for("bb_left", Kokkos::RangePolicy<>(1,NY-1),
        KOKKOS_LAMBDA(int y){
            f_next(y,0,1) = f_curr(y,0,OPP(1));  // f_curr(y,0,3)
            f_next(y,0,5) = f_curr(y,0,OPP(5));  // f_curr(y,0,7)
            f_next(y,0,8) = f_curr(y,0,OPP(8));  // f_curr(y,0,6)
        });

    // Right wall x=NX-1: unknown incoming = channels 3,6,7 (cx < 0)
    Kokkos::parallel_for("bb_right", Kokkos::RangePolicy<>(1,NY-1),
        KOKKOS_LAMBDA(int y){
            f_next(y,NX-1,3) = f_curr(y,NX-1,OPP(3));  // f_curr(y,NX-1,1)
            f_next(y,NX-1,6) = f_curr(y,NX-1,OPP(6));  // f_curr(y,NX-1,8)
            f_next(y,NX-1,7) = f_curr(y,NX-1,OPP(7));  // f_curr(y,NX-1,5)
        });

    Kokkos::fence();
}

// ═════════════════════════════════════════════════════════════════════════════
// move_top_wall — moving lid (y=NY-1), u_w = (U_LID, 0)
//
// Mirrors reference implementation exactly:
//   rho reconstructed from known post-stream populations at the boundary
//   (Zou-He density formula — more accurate than using collision-step density)
//
//   f4 = f2                          (vertical: no x-momentum correction)
//   f7 = f5 - (1/6)*rho*u_lid        (northeast→southwest: lid adds rightward momentum)
//   f8 = f6 + (1/6)*rho*u_lid        (northwest→southeast: lid removes rightward momentum)
//
// Corner nodes (x=0, x=NX-1): plain bounce-back only — they belong to both
// the stationary side wall and the moving lid. At low Mach, either choice
// gives the same flow field; bounce-back is simpler and consistent.
// ═════════════════════════════════════════════════════════════════════════════
void move_top_wall(const View3D& f_next){
    Kokkos::parallel_for("move_top_wall", Kokkos::RangePolicy<>(1,NX-1),
        KOKKOS_LAMBDA(int x){
            // Reconstruct wall density from known post-stream populations.
            // At y=NY-1 after streaming: f0,f1,f2,f3,f5,f6 are known.
            // f4,f7,f8 are unknown (they would come from outside the domain).
            // Zou-He formula: rho = (f0+f1+f3 + 2*(f2+f5+f6)) / (1 - u_y)
            // With u_y=0: rho = f0+f1+f3 + 2*(f2+f5+f6)
            double rho_w = f_next(NY-1,x,0)
                         + f_next(NY-1,x,1)
                         + f_next(NY-1,x,3)
                         + 2.0*(f_next(NY-1,x,2)
                               +f_next(NY-1,x,5)
                               +f_next(NY-1,x,6));

            // f4: vertical reflection — no horizontal correction
            f_next(NY-1,x,4) = f_next(NY-1,x,2);

            // f7: reflects f5 (northeast) → southwest
            // Lid moves right → f7 loses momentum: negative correction
            f_next(NY-1,x,7) = f_next(NY-1,x,5) - (1.0/6.0)*rho_w*U_LID;

            // f8: reflects f6 (northwest) → southeast
            // Lid moves right → f8 gains momentum: positive correction
            f_next(NY-1,x,8) = f_next(NY-1,x,6) + (1.0/6.0)*rho_w*U_LID;
        });

    // Top-left corner (0, NY-1): plain bounce-back for all crossing channels
    // Top-right corner (NX-1, NY-1): same
    Kokkos::parallel_for("corner_bb", Kokkos::RangePolicy<>(0,1),
        KOKKOS_LAMBDA(int){
            for(int i=0;i<9;i++){
                f_next(NY-1,   0,i) = f_next(NY-1,   0,OPP(i));
                f_next(NY-1,NX-1,i) = f_next(NY-1,NX-1,OPP(i));
            }
        });

    Kokkos::fence();
}

// ═════════════════════════════════════════════════════════════════════════════
// collide — BGK collision in-place on f
// f_i += -omega * (f_i - f_eq(rho, u))
// Also writes macroscopic fields vel_x, vel_y, dens for output + convergence.
// ═════════════════════════════════════════════════════════════════════════════
void collide(const View3D& f,
             const View2D& vel_x, const View2D& vel_y, const View2D& dens){
    Kokkos::parallel_for("collide",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{NY,NX}),
        KOKKOS_LAMBDA(int y, int x){
            double r=0, mx=0, my=0;
            for(int i=0;i<9;i++){
                double fi = f(y,x,i);
                r += fi; mx += fi*CX(i); my += fi*CY(i);
            }
            double uvx = mx/r, uvy = my/r;
            vel_x(y,x) = uvx;
            vel_y(y,x) = uvy;
            dens (y,x) = r;
            for(int i=0;i<9;i++)
                f(y,x,i) += -OMEGA*(f(y,x,i) - feq(i,r,uvx,uvy));
        });
    Kokkos::fence();
}

// ═════════════════════════════════════════════════════════════════════════════
// write_csv — helper for snapshot + final output
// ═════════════════════════════════════════════════════════════════════════════
void write_csv(const std::string& path,
               const Kokkos::View<double**,Kokkos::HostSpace>& hx,
               const Kokkos::View<double**,Kokkos::HostSpace>& hy){
    std::ofstream o(path);
    o << "x,y,ux,uy\n";
    for(int y=0;y<NY;y++) for(int x=0;x<NX;x++)
        o << x<<","<<y<<","<<hx(y,x)<<","<<hy(y,x)<<"\n";
}

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]){
    Kokkos::initialize(argc, argv);
    {
        View3D f_curr("f_curr", NY, NX, 9);
        View3D f_next("f_next", NY, NX, 9);
        View2D vel_x ("vel_x",  NY, NX);
        View2D vel_y ("vel_y",  NY, NX);
        View2D dens  ("dens",   NY, NX);
        View2D vel_x_prev("vel_x_prev", NY, NX);
        View2D vel_y_prev("vel_y_prev", NY, NX);

        // Initialize: rho=1, u=0, f=f_eq
        Kokkos::parallel_for("init",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{NY,NX}),
            KOKKOS_LAMBDA(int y, int x){
                for(int i=0;i<9;i++) f_curr(y,x,i) = feq(i,1.,0.,0.);
                vel_x(y,x)=0.; vel_y(y,x)=0.; dens(y,x)=1.;
                vel_x_prev(y,x)=0.; vel_y_prev(y,x)=0.;
            });
        Kokkos::fence();

        double nu = (1./3.)*(1./OMEGA - 0.5);
        std::cout << "Grid: "<<NX<<"x"<<NY
                  << "  omega="<<OMEGA
                  << "  nu="<<nu
                  << "  Re="<<(U_LID*NX/nu)<<"\n";

        auto t_start = std::chrono::steady_clock::now();
        int  final_step = MAXSTEP;

        std::ofstream conv_log("convergence_log.csv");
        conv_log << "step,max_du\n";

        for(int step = 0; step < MAXSTEP; step++){

            // Save previous velocity for convergence check
            Kokkos::deep_copy(vel_x_prev, vel_x);
            Kokkos::deep_copy(vel_y_prev, vel_y);

            // ── Step 1: Stream ───────────────────────────────────────────
            stream(f_curr, f_next);

            // ── Step 2: Bounce-back on stationary walls ──────────────────
            // Reads f_curr (pre-stream = f_i*), writes f_next
            bounce_back(f_curr, f_next);

            // ── Step 3: Moving wall on top lid ───────────────────────────
            // Reads and writes f_next (post-stream populations are now correct
            // at interior; top wall unknowns reconstructed from known f_next values)
            move_top_wall(f_next);

            // ── Step 4: Collide (BGK, in-place on f_next) ────────────────
            // f_next becomes the post-collision distribution for next step
            collide(f_next, vel_x, vel_y, dens);

            // Swap: f_next (post-collision) becomes f_curr for next step
            std::swap(f_curr, f_next);

            // ── Convergence check ────────────────────────────────────────
            double max_du = 0.;
            Kokkos::parallel_reduce("conv",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{NY-1,NX-1}),
                KOKKOS_LAMBDA(int y, int x, double& lmax){
                    double du = Kokkos::abs(vel_x(y,x) - vel_x_prev(y,x))
                              + Kokkos::abs(vel_y(y,x) - vel_y_prev(y,x));
                    if(du > lmax) lmax = du;
                },
                Kokkos::Max<double>(max_du));
            Kokkos::fence();

            conv_log << step << "," << max_du << "\n";

            // Snapshot for animation
            if(step % SNAP_INTERVAL == 0){
                auto sx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vel_x);
                auto sy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vel_y);
                char name[64];
                std::snprintf(name, sizeof(name), "snapshot_%06d.csv", step);
                write_csv(name, sx, sy);
            }

            if(step > 100 && max_du < CONV){
                std::cout << "Converged at step "<<step
                          << "  max|du|="<<max_du<<"\n";
                // Final snapshot at convergence
                auto sx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vel_x);
                auto sy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vel_y);
                char name[64];
                std::snprintf(name, sizeof(name), "snapshot_%06d.csv", step);
                write_csv(name, sx, sy);
                final_step = step;
                break;
            }

            if(step % LOG_INTERVAL == 0)
                std::cout << "Step "<<step<<"  max|du|="<<max_du<<"\n";
        }

        auto t_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        double mlups = (double)NX * NY * final_step / elapsed / 1e6;

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                  << "Grid:    " << NX << " x " << NY << "\n"
                  << "Steps:   " << final_step << "\n"
                  << "Elapsed: " << elapsed << " s\n"
                  << "MLUPS:   " << mlups << "\n"
                  << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

        // Write final velocity field and centerline
        auto hvel_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vel_x);
        auto hvel_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vel_y);
        write_csv("velocity_field.csv", hvel_x, hvel_y);
        {
            std::ofstream o("centerline.csv");
            o << "y_norm,ux_norm\n";
            for(int y=0;y<NY;y++)
                o << (double)y/(NY-1) << "," << hvel_x(y,NX/2)/U_LID << "\n";
        }
        std::cout << "Wrote velocity_field.csv, centerline.csv\n";
    }
    Kokkos::finalize();
    return 0;
}