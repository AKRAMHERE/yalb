#include "lbm.h"
#include <iostream>
#include <fstream>
#include <numbers>
#include <mpi.h>

LBM create_lbm(int rows, int cols) {
    LBM grid;
    grid.rows = rows;
    grid.cols = cols;

    grid.rho = Kokkos::View<double**>("rho", rows, cols);

    grid.f = Kokkos::View<double***, Kokkos::LayoutRight>(
        "f",
        rows,
        cols,
        9
    );

    grid.f_next = Kokkos::View<double***, Kokkos::LayoutRight>(
        "f_next",
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

    const int halo_size = cols * 9; // Each row has cols * 9 distribution functions

    grid.send_lower =
        Kokkos::View<double*>(
            "send_lower",
            halo_size
        );

    grid.send_upper =
        Kokkos::View<double*>(
            "send_upper",
            halo_size
        );

    grid.recv_lower =
        Kokkos::View<double*>(
            "recv_lower",
            halo_size
        );

    grid.recv_upper =
        Kokkos::View<double*>(
            "recv_upper",
            halo_size
        );

    

    return grid;
}

double compute_local_fluid_mass(const LBM& lbm)
{
    double local_mass = 0.0;

    Kokkos::parallel_reduce(
        "FluidMass",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),
        KOKKOS_LAMBDA(int row, int col, double& mass) {
            if (!lbm.wall(row, col)) {
                mass += lbm.rho(row, col);
            }
        },
        local_mass
    );

    return local_mass;
}

void print_lbm_message() {
    std::cout << "LBM initialized\n";
}

void create_walls(LBM& lbm, int local_start, int global_rows) {
    // Set walls on the left, right, and bottom boundaries of the grid
    // Need to ensure that each rank sets its own walls correctly based on its local grid portion
    Kokkos::parallel_for(
        "CreateWalls",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {lbm.rows, lbm.cols}),
        KOKKOS_LAMBDA(int row, int col) {
            
            const int global_row = local_start + row - 1;

            const bool ghost_row =
                row == 0 || row == lbm.rows - 1;

            const bool bottom_wall = 
                global_row == 0;

            const bool top_wall =
                global_row == global_rows - 1;

            const bool left_wall = 
                col == 0;

            const bool right_wall =
                col == lbm.cols - 1;
            
            if (ghost_row) {
                /*
                    Ghost rows represent neighboring rows, but their
                    first and last columns are still side walls.
                */
               lbm.wall(row, col) = 
                    left_wall || right_wall;
            } else {
                lbm.wall(row, col) =
                    bottom_wall ||
                    top_wall ||
                    left_wall ||
                    right_wall;
            }
        }
    );

}

void move_top_wall(LBM& lbm, double u_lid, int local_start, int global_rows) {

    // int y = lbm.cols - 1; // top boundary row

    const int top_fluid_global_x = global_rows - 2; // The last fluid row in the global grid (just below the top wall)
    const int local_x = top_fluid_global_x - local_start + 1; // Convert to local index, +1 for ghost cell offset   


    if (local_x < 1 || local_x >= lbm.rows - 1) {
        // This rank does not own the top fluid row, so return early
        return;
    }

    // loop over every position on along the lid
    Kokkos::parallel_for(
        "MoveTopWall",
        Kokkos::RangePolicy<>(1, lbm.rows - 1), // Exclude ghost cells
        KOKKOS_LAMBDA(int y) {

            /*
                Coordinate convention in this code:
                    first array index, x: vertical direction
                    second array index, y: horizontal direction

                D2Q9 directions crossing the top wall have cx = +1:

                    i = 1: (+1,  0)
                    i = 5: (+1, +1)
                    i = 8: (+1, -1)

                stream_lbm() has already reflected these into:

                    1 -> 3
                    5 -> 7
                    8 -> 6

                The lid moves in the positive second-index direction,
                so the reflected diagonal populations require the
                standard moving-wall correction.
            */
            
            double rho = 0.0;

            for (int i = 0; i < 9; ++i) {
                rho += lbm.f(local_x, y, i);
            }

            const double correction = rho * u_lid / 6.0;

            /*
                Population 5 had positive horizontal velocity before
                hitting the wall. Its reflected population is 7.
            */

            lbm.f(local_x, y, 7) -= correction;

            /*
                Population 8 had negative horizontal velocity before
                hitting the wall. Its reflected population is 6.
            */
            lbm.f(local_x, y, 6) += correction;

        }
    );

}

void stream_lbm(LBM& lbm, double u_lid, int local_start, int global_rows) {


    Kokkos::View<double***, Kokkos::LayoutRight> f_next("f_next", lbm.rows, lbm.cols, 9);
    
    /*
        Initialize f_next to zero.

        This matters because not every f_next entry is necessarily written
        during every streaming step, especially near walls.
    */

    Kokkos::parallel_for(
        "StreamLBM",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0}, {lbm.rows - 1, lbm.cols, 9}),
        KOKKOS_LAMBDA(int row, int col, int i) {
            int drow[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1}; // Example velocity directions
            int dcol[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

            int opposite[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};


            /*
                Wall nodes are not normal fluid nodes.

                We do not stream populations out of solid wall cells.
                Only fluid cells participate in collision/streaming.
            */
            if (lbm.wall(row, col)) {
                return;
            }

            
            const int next_row = row + drow[i];
            const int next_col = col + dcol[i];


            /*
                Check whether the destination is outside the grid.

                For a 300x300 grid, valid indices are:
                    0 ... 299

                So x_next == 300 is outside.
                y_next == 300 is outside.
            */
            bool outside = next_row < 0 || next_row >= lbm.rows ||
                           next_col < 0 || next_col >= lbm.cols;
            

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
            if (outside) {
                f_next(row, col, opposite[i]) = lbm.f(row, col, i);
                return;
            }

            if (lbm.wall(next_row, next_col)) {
                /*
                    Convert the destinations local vertical index
                    to a global vertical index

                    Local fluid index 1 corresponds to global row
                    local_start
                */
                const int destination_global_x = local_start + (next_row - 1); // Convert to global index, -1 for ghost cell offset
                /*
                    The top boundary is global_rows - 1

                    Exclude the two corner nodes because the moving lid
                    normally does not include the stationary side-wall corners
                */
                const bool hits_moving_lid =
                    destination_global_x == global_rows - 1 &&
                    next_col > 0 &&
                    next_col < lbm.cols - 1;

                if (hits_moving_lid) {
                   /*
                        Moving wall bounce back:

                        f_opposite =
                            f_incoming - 6 * w_i * rho (c_i dot u_wall)

                        for diagonal D2Q9 populations, w_i = 1/36, and c_i dot u_wall = +/- u_lid

                        so 6 * w_i = 1/6

                        The lid velocity is in the positive horizontal, second-index direction:

                        u_wall = (0, u_lid)
                   
                   */ 
                    const double rho = lbm.rho(row, col);

                    const double wall_correction = 
                        6.0 *
                        (1.0 / 36.0) *
                        rho *
                        dcol[i] *
                        u_lid;
                    
                    f_next(row, col, opposite[i]) = lbm.f(row, col, i) - wall_correction;

                } else {
                    // ordinary stationary-wall bounce back
                    f_next(row, col, opposite[i]) = lbm.f(row, col, i);
                }

                return;
            }
            f_next(next_row, next_col, i) = lbm.f(row, col, i);                  
        }
    );

    lbm.f = f_next; // Update the distribution functions after streaming
}


void stream_lbm_pull(LBM& lbm, double u_lid, int local_start, int global_rows) {

    /*
        Initialize f_next to zero.

        This matters because not every f_next entry is necessarily written
        during every streaming step, especially near walls.
    */
    Kokkos::deep_copy(lbm.f_next, 0.0);

    Kokkos::parallel_for(
        "StreamLBM",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0}, {lbm.rows - 1, lbm.cols, 9}),
        KOKKOS_LAMBDA(int row, int col, int i) {
            int drow[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1}; // Example velocity directions
            int dcol[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

            int opposite[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

            const double weights[9] = {
                4.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 36.0,
                1.0 / 36.0,
                1.0 / 36.0,
                1.0 / 36.0
            };

            
            /*
                Do not update physical wall nodes as fluid destinations
            */
            if (lbm.wall(row, col)){
                return;
            }

            /*
                Pull population i from the source cell

                Push:
                    destination = source + c_i

                Pull:
                    source = destination - c_i
            */
            const int source_row = row - drow[i];
            const int source_col = col - dcol[i];

            const bool source_outside =
                source_row < 0 ||
                source_row >= lbm.rows ||
                source_col < 0 ||
                source_col >= lbm.cols;

            if (source_outside) {
                lbm.f_next(row, col, i) = lbm.f(row, col, opposite[i]);
                return;
            }

            /*
                If the source is a physical wall, reconstruct the incoming
                population using bounce back
            */
           if (lbm.wall(source_row, source_col)) {
                const int source_global_row = local_start + source_row - 1;

                const bool hits_moving_lid =
                    source_global_row == global_rows - 1 &&
                    col > 0 &&
                    col < lbm.cols - 1;

                if (hits_moving_lid) {
                    const double rho = lbm.rho(row, col);
                    
                    const double wall_correction =
                        6.0 *
                        weights[i] *
                        rho *
                        dcol[i] *
                        u_lid;

                    lbm.f_next(row, col, i) = lbm.f(row, col, opposite[i]) + wall_correction;
                } else {
                    lbm.f_next(row, col, i) = lbm.f(row, col, opposite[i]);
                }
            return;
           }
           lbm.f_next(row, col, i) = lbm.f(source_row, source_col, i);
        }
    );

    std::swap(lbm.f, lbm.f_next); // Update the distribution functions after streaming

}



void compute_density(LBM& lbm) {
    // Example computation: compute density from distribution functions
    Kokkos::parallel_for(
        "ComputeDensity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),
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
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),
        KOKKOS_LAMBDA(int row, int col) {
            double velocity_vertical = 0.0;
            double velocity_horizontal = 0.0;
            int drow[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1}; // Example velocity directions
            int dcol[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
            for (int i = 0; i < 9; ++i) {
                velocity_vertical += lbm.f(row, col, i) * drow[i];
                velocity_horizontal += lbm.f(row, col, i) * dcol[i];
            }
            if (lbm.rho(row, col) > 0) { // Avoid division by zero
                lbm.v(row, col, 0) = velocity_vertical / lbm.rho(row, col);
                lbm.v(row, col, 1) = velocity_horizontal / lbm.rho(row, col);
            } else {
                lbm.v(row, col, 0) = 0.0;
                lbm.v(row, col, 1) = 0.0;
            }
        }
    );
}

void write_output_f(const LBM& lbm, const std::string& filename) {
    Kokkos::View<double***, Kokkos::LayoutRight>::HostMirror f_host = Kokkos::create_mirror_view(lbm.f);
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

void write_output_rho(const LBM& lbm, const std::string& filename, int local_start) {
    Kokkos::View<double**>::HostMirror rho_host = Kokkos::create_mirror_view(lbm.rho);
    Kokkos::deep_copy(rho_host, lbm.rho);

    std::ofstream output_file(filename);

    for (int x = 1; x < lbm.rows - 1; ++x) {
        int global_x = local_start + (x - 1); // Calculate the global x-coordinate based on the local start index
        for (int y = 0; y < lbm.cols; ++y) {
            output_file << global_x << " "
                        << y << " "
                        << rho_host(x, y) << "\n";
        }
        output_file << "\n";
    }
    output_file.close();
}

void write_output_velocity(const LBM& lbm, const std::string& filename, int local_start) {
    auto v_host = Kokkos::create_mirror_view(lbm.v);
    Kokkos::deep_copy(v_host, lbm.v);

    std::ofstream output_file(filename);

    for (int row = 1; row < lbm.rows - 1; ++row) {
        const int global_row = local_start + (row - 1); // Calculate the global x-coordinate based on the local start index
        
        for (int col = 0; col < lbm.cols; ++col) {
            
            const double vertical_velocity = v_host(row, col, 0);

            const double horizontal_velocity = v_host(row, col, 1);

            output_file << global_row << " "
                        << col << " "
                        << horizontal_velocity << " "
                        << vertical_velocity << "\n";
        }
        output_file << "\n";
    }
    output_file.close();
}

Kokkos::View<double***, Kokkos::LayoutRight> compute_equilibrium(LBM& lbm) {
    Kokkos::View<double***, Kokkos::LayoutRight> feq("feq", lbm.rows, lbm.cols, 9);

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
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),
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

void collision_step(LBM& lbm)
{
    constexpr double tau = 0.596;
    constexpr double omega = 1.0 / tau;

    Kokkos::parallel_for(
        "Collision",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {1, 0},
            {lbm.rows - 1, lbm.cols}
        ),
        KOKKOS_LAMBDA(int row, int col) {

            if (lbm.wall(row, col)) {
                return;
            }

            double f[9];

            for (int i = 0; i < 9; ++i) {
                f[i] = lbm.f(row, col, i);
            }

            const double rho =
                f[0] + f[1] + f[2] +
                f[3] + f[4] + f[5] +
                f[6] + f[7] + f[8];

            const double ux =
                (
                    f[1] - f[3] +
                    f[5] - f[6] -
                    f[7] + f[8]
                ) / rho;

            const double uy =
                (
                    f[2] - f[4] +
                    f[5] + f[6] -
                    f[7] - f[8]
                ) / rho;

            const double u2 =
                ux * ux + uy * uy;

            constexpr double w[9] = {
                4.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 36.0,
                1.0 / 36.0,
                1.0 / 36.0,
                1.0 / 36.0
            };

            constexpr int cx[9] =
                {0,1,0,-1,0,1,-1,-1,1};

            constexpr int cy[9] =
                {0,0,1,0,-1,1,1,-1,-1};

            for (int i = 0; i < 9; ++i) {

                const double cu =
                    cx[i] * ux +
                    cy[i] * uy;

                const double feq =
                    w[i] *
                    rho *
                    (
                        1.0 +
                        3.0 * cu +
                        4.5 * cu * cu -
                        1.5 * u2
                    );

                lbm.f(row, col, i) =
                    f[i] -
                    omega * (f[i] - feq);
            }

            lbm.rho(row, col) = rho;
            lbm.v(row, col, 0) = ux;
            lbm.v(row, col, 1) = uy;
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
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0}, {lbm.rows - 1, lbm.cols, 9}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            lbm.f(x, y, i) = feq(x, y, i);
        }
    );
    
    
}

double compute_total_mass(const LBM& lbm) {
    auto rho_host = Kokkos::create_mirror_view(lbm.rho);
    Kokkos::deep_copy(rho_host, lbm.rho);
    
    double mass = 0.0;
    for (int x = 1; x < lbm.rows - 1; ++x) {
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
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0}, {lbm.rows - 1, lbm.cols, 9}),
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
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0}, {lbm.rows - 1, lbm.cols, 9}),
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
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),
        KOKKOS_LAMBDA(int x, int y) {
            lbm.rho(x, y) = 1.0; // uniform density
            lbm.v(x, y, 0) = 0.0; // vx
            lbm.v(x, y, 1) = 0.0; // vy
        }
    );
    auto feq = compute_equilibrium(lbm);


    Kokkos::parallel_for(
        "InitializeDistributionFunctions",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0}, {lbm.rows - 1, lbm.cols, 9}),
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
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0}, {lbm.rows - 1, lbm.cols, 9}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            lbm.f(x, y, i) = feq(x, y, i);
        }
    );


}



void exchange_halos(LBM& lbm, int rank, int size) {
    
    const int lower_rank =
        (rank == 0)
            ? MPI_PROC_NULL
            : rank - 1;

    const int upper_rank =
        (rank == size - 1)
            ? MPI_PROC_NULL
            : rank + 1;

    const int values_per_row =
        lbm.cols * 9;


    // Pack boundary directly on the GPU

    // Owned rows: 1 ... lbm.rows - 2
    // Ghost rows: 0 and lbm.rows - 1

    Kokkos::parallel_for(
        "PackHalos",
        Kokkos::RangePolicy<>(0, values_per_row),
        KOKKOS_LAMBDA(const int idx) {

            const int col = idx / 9;
            const int i   = idx % 9;

            lbm.send_lower(idx) =
                lbm.f(1, col, i);

            lbm.send_upper(idx) =
                lbm.f(lbm.rows - 2, col, i);
        }
    );

    Kokkos::fence(); // Ensure packing is complete before MPI communication

    // Send lower owned row downward. Receive upper lower owned row.

    MPI_Sendrecv(
        lbm.send_lower.data(),
        values_per_row,
        MPI_DOUBLE,
        lower_rank,
        100,

        lbm.recv_upper.data(),
        values_per_row,
        MPI_DOUBLE,
        upper_rank,
        100,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // Send upper owned row upward. Receive lower upper owned row.

    MPI_Sendrecv(
        lbm.send_upper.data(),
        values_per_row,
        MPI_DOUBLE,
        upper_rank,
        200,

        lbm.recv_lower.data(),
        values_per_row,
        MPI_DOUBLE,
        lower_rank,
        200,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // Unpack received halos directly on the GPU

    Kokkos::parallel_for(
        "UnpackHalos",
        Kokkos::RangePolicy<>(0, values_per_row),
        KOKKOS_LAMBDA(const int idx) {

            const int col = idx / 9;
            const int i   = idx % 9;

            if (lower_rank != MPI_PROC_NULL) {
                lbm.f(0, col, i) =
                    lbm.recv_lower(idx);
            }

            if (upper_rank != MPI_PROC_NULL) {
                lbm.f(lbm.rows - 1, col, i) =
                    lbm.recv_upper(idx);
            }
        }
    );

    // Ensure unpacking is complete
    Kokkos::fence();
}