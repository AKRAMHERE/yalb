#include "lbm.h"
#include <iostream>
#include <fstream>
#include <numbers>

LBM create_lbm(int rows, int cols) {
    LBM grid;
    grid.rows = rows;
    grid.cols = cols;

    grid.rho = Kokkos::View<double**>("rho", rows, cols);

    grid.f = Kokkos::View<double***>(
        "f",
        rows,
        cols,
        9
    );

    grid.v = Kokkos::View<double***>(
        "v",
        rows,
        cols,
        2
    );

    grid.wall = Kokkos::View<bool**>(
        "wall",
        rows,
        cols
    );

    return grid;
}

void print_lbm_message() {
    std::cout << "LBM initialized\n";
}

void create_walls(LBM& lbm) {

    Kokkos::parallel_for(
        "CreateWall",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {lbm.rows, lbm.cols}),
        KOKKOS_LAMBDA(int x, int y) {

            if (x == 0 || x == lbm.rows - 1 || y == 0)
                lbm.wall(x, y) = true;
            else
                lbm.wall(x, y) = false;

        }   
    );

}

void move_top_wall(LBM& lbm, double u_lid) {

    int y = lbm.cols - 1; // top boundary row


    // loop over every position on along the lid
    Kokkos::parallel_for(
        "MoveTopWall",
        lbm.rows,
        KOKKOS_LAMBDA(int x) {

            /*
                Reconstruct density at the top boundary.

                At the top wall, some populations are known after streaming
                and some are missing. This formula estimates rho from the
                populations available at the boundary.

                The factor 2 appears because the unknown opposite-direction
                populations are paired with the known populations.
            */

            double rho =
                lbm.f(x, y, 0) +
                lbm.f(x, y, 1) +
                lbm.f(x, y, 3) +
                2.0 * (
                    lbm.f(x, y, 2) +
                    lbm.f(x, y, 5) +
                    lbm.f(x, y, 6)
                );
        /*
                Vertical bounce-back part.

                f2 points upward toward the lid.
                f4 points downward back into the fluid.

                For the straight vertical direction, the moving lid does not
                add horizontal correction, so we simply reflect:
                    f4 = f2
        */        
        lbm.f(x, y, 4) = lbm.f(x, y, 2);

        /*
                Diagonal bounce back with moving wall correction.

                f5 points northeast.
                Its opposite direction is f7, southwest.

                Because the lid moves to the right, the reflected diagonal
                populations are adjusted so the fluid receives positive
                x-momentum from the lid.

                This term:
                    (1/6) * rho * u_lid

                is the D2Q9 moving-wall correction for the diagonal directions.
        */
        lbm.f(x, y, 7) = lbm.f (x, y, 5) - (1.0/6.0) * rho * u_lid;
        
        /*
                f6 points northwest.
                Its opposite direction is f8, southeast.

                This gets the opposite sign from f7 because f8 has positive
                x-direction while f7 has negative x-direction.
        */
        lbm.f(x, y, 8) = lbm.f (x, y, 6) + (1.0/6.0) * rho * u_lid;



        }
    );

}

void stream_lbm(LBM& lbm) {


    Kokkos::View<double***> f_next("f_next", lbm.rows, lbm.cols, 9);
    
    /*
        Initialize f_next to zero.

        This matters because not every f_next entry is necessarily written
        during every streaming step, especially near walls.
    */
    Kokkos::deep_copy(f_next, 0.0);

    Kokkos::parallel_for(
        "StreamLBM",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {lbm.rows, lbm.cols, 9}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1}; // Example velocity directions
            int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

            int opposite[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};


            /*
                Wall nodes are not normal fluid nodes.

                We do not stream populations out of solid wall cells.
                Only fluid cells participate in collision/streaming.
            */
            if (lbm.wall(x, y)) {
                return;
            }

            
            int x_next = x + cx[i];
            int y_next = y + cy[i];


            /*
                Check whether the destination is outside the grid.

                For a 300x300 grid, valid indices are:
                    0 ... 299

                So x_next == 300 is outside.
                y_next == 300 is outside.
            */
            bool outside = x_next < 0 || x_next >= lbm.rows ||
                           y_next < 0 || y_next >= lbm.cols;
            

            /*
                Case 1:
                The population would leave the domain or enter a wall.

                Then it bounces back.

                Instead of:
                    f_next(x_next, y_next, i)

                we write:
                    f_next(x, y, opposite[i])

                Meaning:
                    stay at the current fluid node,
                    but reverse direction.
            */
            if (outside || lbm.wall(x_next, y_next)) {
                f_next(x, y, opposite[i]) = lbm.f(x, y, i);
            }

             /*
                Case 2:
                The destination is a normal fluid node.

                Then stream normally:
                    f_i(x,y) moves to f_i(x_next,y_next)
            */
            else {
                f_next(x_next, y_next, i) = lbm.f(x, y, i);
            }
        }
    );

    lbm.f = f_next; // Update the distribution functions after streaming
}

void compute_density(LBM& lbm) {
    // Example computation: compute density from distribution functions
    Kokkos::parallel_for(
        "ComputeDensity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {lbm.rows, lbm.cols}),
        KOKKOS_LAMBDA(int x, int y) {
            double local_rho = 0.0;
            for (int i = 0; i < 9; ++i) {
                local_rho += lbm.f(x, y, i);
            }
            lbm.rho(x, y) = local_rho;
        }
    );
}

void compute_velocity(LBM& lbm) {
    // Example computation: compute velocity from distribution functions
    Kokkos::parallel_for(
        "ComputeVelocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {lbm.rows, lbm.cols}),
        KOKKOS_LAMBDA(int x, int y) {
            double local_vx = 0.0;
            double local_vy = 0.0;
            int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1}; // Example velocity directions
            int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
            for (int i = 0; i < 9; ++i) {
                local_vx += lbm.f(x, y, i) * cx[i];
                local_vy += lbm.f(x, y, i) * cy[i];
            }
            if (lbm.rho(x, y) > 0) { // Avoid division by zero
                lbm.v(x, y, 0) = local_vx / lbm.rho(x, y);
                lbm.v(x, y, 1) = local_vy / lbm.rho(x, y);
            } else {
                lbm.v(x, y, 0) = 0.0;
                lbm.v(x, y, 1) = 0.0;
            }
        }
    );
}

void write_output_f(const LBM& lbm, const std::string& filename) {
    Kokkos::View<double***>::HostMirror f_host = Kokkos::create_mirror_view(lbm.f);
    Kokkos::deep_copy(f_host, lbm.f);

    std::ofstream output_file(filename);

    for (int x = 0; x < lbm.rows; ++x) {
        for (int y = 0; y < lbm.cols; ++y) {
            for (int i = 0; i < 9; ++i) {
                if (f_host(x, y, i) != 0.0) {
                    output_file << x << " "
                                << y << " "
                                << i << " "
                                << f_host(x, y, i) << "\n";
                }
            }
            output_file << "\n";
        }
    }
    output_file.close();
}

void write_output_rho(const LBM& lbm, const std::string& filename) {
    Kokkos::View<double**>::HostMirror rho_host = Kokkos::create_mirror_view(lbm.rho);
    Kokkos::deep_copy(rho_host, lbm.rho);

    std::ofstream output_file(filename);

    for (int x = 0; x < lbm.rows; ++x) {
        for (int y = 0; y < lbm.cols; ++y) {
            output_file << x << " "
                        << y << " "
                        << rho_host(x, y) << "\n";
        }
        output_file << "\n";
    }
    output_file.close();
}

void write_output_velocity(const LBM& lbm, const std::string& filename) {
    Kokkos::View<double***>::HostMirror v_host = Kokkos::create_mirror_view(lbm.v);
    Kokkos::deep_copy(v_host, lbm.v);

    std::ofstream output_file(filename);

    for (int x = 0; x < lbm.rows; ++x) {
        for (int y = 0; y < lbm.cols; ++y) {
            output_file << x << " "
                        << y << " "
                        << v_host(x, y, 0) << " "
                        << v_host(x, y, 1) << "\n";
        }
        output_file << "\n";
    }
    output_file.close();
}

Kokkos::View<double***> compute_equilibrium(LBM& lbm) {
    Kokkos::View<double***> feq("feq", lbm.rows, lbm.cols, 9);

    double w[9] = {
        4.0/9.0,
        1.0/9.0,
        1.0/9.0,
        1.0/9.0,
        1.0/9.0,
        1.0/36.0,
        1.0/36.0,
        1.0/36.0,
        1.0/36.0
    };

    int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

    Kokkos::parallel_for(
        "ComputeEquilibrium",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {lbm.rows, lbm.cols}),
        KOKKOS_LAMBDA(int x, int y) {
            double rho = lbm.rho(x, y);
            double vx = lbm.v(x, y, 0);
            double vy = lbm.v(x, y, 1);

            for (int i = 0; i < 9; ++i) {
                double cu = cx[i] * vx + cy[i] * vy;
                double v2 = vx*vx + vy*vy;
                feq(x, y, i) = w[i] * rho *
                    (1 + 3*cu + 4.5*cu*cu - 1.5*v2);
            }
        }
    );

    return feq;
}

void collision_step(LBM& lbm) {
    double tau = 0.6; // Relaxation time
    auto feq = compute_equilibrium(lbm);

    Kokkos::parallel_for(
        "CollisionStep",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {lbm.rows, lbm.cols, 9}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            lbm.f(x, y, i) += -(lbm.f(x, y, i) - feq(x, y, i)) / tau;
        }
    );
}

void initialize_density_bump(LBM& lbm) {
    /*
    Want:
    rho = uniform background
    rho(centre) = slightly higher
    v = 0 everywhere
    f = feq(rho, v) everywhere
    */

    double rho0 = 0.1; // Background density
    double bump = 0.05; // Density bump at the center
    int x_center = lbm.rows / 2;
    int y_center = lbm.cols / 2;

    Kokkos::parallel_for(
        "InitializeDensityBump",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {lbm.rows, lbm.cols}),
        KOKKOS_LAMBDA(int x, int y) {
            double local_rho = rho0;

            if (x == x_center && y == y_center) {
                local_rho += bump; // Add bump at the center
            }

            lbm.rho(x, y) = local_rho;
            lbm.v(x, y, 0) = 0.0; // vx
            lbm.v(x, y, 1) = 0.0; // vy
        }
    );

    auto feq = compute_equilibrium(lbm);

    Kokkos::parallel_for(
        "InitializeDistributionFunctions",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {lbm.rows, lbm.cols, 9}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            lbm.f(x, y, i) = feq(x, y, i);
        }
    );
    
    
}

double compute_total_mass(const LBM& lbm) {
    auto rho_host = Kokkos::create_mirror_view(lbm.rho);
    Kokkos::deep_copy(rho_host, lbm.rho);

    double mass = 0.0;
    for (int x = 0; x < lbm.rows; ++x) {
        for (int y = 0; y < lbm.cols; ++y) {
            mass += rho_host(x, y);
        }
    }

    return mass;
}

void initialize_velocity_bump(LBM& lbm){
          Kokkos::parallel_for(
            "InitializeVelocityBump",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {lbm.rows, lbm.cols}),
            KOKKOS_LAMBDA(int x, int y) {
                lbm.rho(x, y) = 0.1; // uniform density
                if (x < lbm.rows / 2) {
                    lbm.v(x, y, 0) = 0.05; // vx
                    lbm.v(x, y, 1) = 0.0;  // vy
                } else {
                    lbm.v(x, y, 0) = 0.0; // vx
                    lbm.v(x, y, 1) = 0.0; // vy
                }
              }
          );
        auto feq = compute_equilibrium(lbm);

        Kokkos::parallel_for(
            "InitializeDistributionFunctions",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {lbm.rows, lbm.cols, 9}),
            KOKKOS_LAMBDA(int x, int y, int i) {
                lbm.f(x, y, i) = feq(x, y, i);
            }
        );
}

void initialize_fixed_point(LBM& lbm){
    // rho = 0.1 everywhere
    // v = 0 everywhere
    // f = feq(rho, v) everywhere
    Kokkos::parallel_for(
        "InitializeFixedPoint",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {lbm.rows, lbm.cols}),
        KOKKOS_LAMBDA(int x, int y) {
            lbm.rho(x, y) = 0.7; // uniform density
            lbm.v(x, y, 0) = 0.0; // vx
            lbm.v(x, y, 1) = 0.0; // vy
        }
    );
    auto feq = compute_equilibrium(lbm);


    Kokkos::parallel_for(
        "InitializeDistributionFunctions",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {lbm.rows, lbm.cols, 9}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            lbm.f(x, y, i) = feq(x, y, i);
        }
    );
    

}

void initialize_eq_conditions(LBM& lbm){
    // rho = 1.0 everywhere
    // v = 0 everywhere
    // f = feq(rho, v) everywhere
    Kokkos::parallel_for(
        "InitializeFixedPoint",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {lbm.rows, lbm.cols}),
        KOKKOS_LAMBDA(int x, int y) {
            lbm.rho(x, y) = 1.0; // uniform density
            lbm.v(x, y, 0) = 0.0; // vx
            lbm.v(x, y, 1) = 0.0; // vy
        }
    );
    auto feq = compute_equilibrium(lbm);


    Kokkos::parallel_for(
        "InitializeDistributionFunctions",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {lbm.rows, lbm.cols, 9}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            lbm.f(x, y, i) = feq(x, y, i);
        }
    );
    

}

void initialize_shear_wave(LBM &lbm) {

    // rho = 1
    // ux = epsilon * sin((2piy) / ny)
    // uy = 0

    double epsilon = 0.01;
    // double pi = std::numbers::pi;
    int ny = lbm.cols;


    Kokkos::parallel_for(
        "InitializeShearWave",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {lbm.rows, lbm.cols}),
        KOKKOS_LAMBDA(int x, int y) {
            
            double rho = 0.5;
            double ux = epsilon * sin((2.0 * 3.1415 * y) / ny);
            double uy = 0;
            
            lbm.rho(x, y) = rho;
            lbm.v(x, y, 0) = ux;
            lbm.v(x, y, 1) = uy;


        }
    );

    auto feq = compute_equilibrium(lbm);


    Kokkos::parallel_for(
        "InitializeDistributionFunctions",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {lbm.rows, lbm.cols, 9}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            lbm.f(x, y, i) = feq(x, y, i);
        }
    );


}