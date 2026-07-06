add_library(lib STATIC
    hello.cpp
    #lj.cpp
    lbm.cpp
)
target_include_directories(lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lib PUBLIC MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
