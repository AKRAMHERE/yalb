#include "hello.h"
#include "lbm.h"
#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <Kokkos_Core.hpp>


int main(int argc, char *argv[]) {
    int rank = 0, size = 1;

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);


    {
        LBM lbm_grid = create_lbm(128, 128);
        print_lbm_message();

        create_walls(lbm_grid);

        // initialize_density_bump(lbm_grid);
        // initialize_velocity_bump(lbm_grid);
        //initialize_fixed_point(lbm_grid);
        //initialize_shear_wave(lbm_grid);
        initialize_eq_conditions(lbm_grid);

        for (int step = 0; step < 20000; ++step) {
            compute_density(lbm_grid);
            //Compute total mass and print every 10 steps
            //if (step % 10 == 0)
            //  printf("Total mass at step %d: %f\n", step, compute_total_mass(lbm_grid));
            
            compute_velocity(lbm_grid);

            // Confirm momentum conservation by checking the total momentum
            //compare sum_i, cx[i], and f[i]
            //printf("momentum before collision at step %d: %f\n", step, lbm_grid.v(lbm_grid.rows/2, lbm_grid.cols/2, 0) + lbm_grid.v(lbm_grid.rows/2, lbm_grid.cols/2, 1));
            collision_step(lbm_grid);
            //printf("momentum after collision at step %d: %f\n", step, lbm_grid.v(lbm_grid.rows/2, lbm_grid.cols/2, 0) + lbm_grid.v(lbm_grid.rows/2, lbm_grid.cols/2, 1));

            
            stream_lbm(lbm_grid);
            move_top_wall(lbm_grid, 0.1);
            
            if (step % 1000 == 0){
              write_output_rho(lbm_grid, "data/rho/output_rho_" + std::to_string(step) + ".txt");
              write_output_velocity(lbm_grid, "data/v/output_velocity_" + std::to_string(step) + ".txt");
            }
            if (step % 100 == 0){
              printf("done step: %d\n", step);
            }
        }

    }

    // Retrieve process infos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Hello I am rank " << rank << " of " << size << "\n";

    if (rank == 0)
      hello_world();

    auto input_path = "./simulation_test_input.txt";

    if (not std::filesystem::exists(input_path))
      std::cerr << "warning: could not find input file " << input_path << "\n";

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
