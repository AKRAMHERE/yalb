add_subdirectory(02)
add_subdirectory(03)
add_subdirectory(04)
add_subdirectory(05)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE lib)
