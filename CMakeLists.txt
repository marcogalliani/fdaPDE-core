cmake_minimum_required(VERSION 3.14)
project(fdaPDE-core)

add_subdirectory(test/)

add_subdirectory(experiments/test_randEVD/)
add_subdirectory(experiments/test_randSVD/)

add_subdirectory(experiments/new_rsvd/)
add_subdirectory(experiments/new_revd/)
add_subdirectory(experiments/rpchol/)
add_subdirectory(experiments/python_interface/)




